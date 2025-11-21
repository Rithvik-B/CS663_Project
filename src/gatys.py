"""
Baseline Gatys NST wrapper that safely reuses existing implementation.
"""

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import sys
from typing import Dict, Optional, Tuple
from collections import namedtuple

# Add parent directory to import existing utils
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import utils.utils as utils_parent
except ImportError:
    utils_parent = None

from .vgg_features import VGGFeatureExtractor
from .io_utils import prepare_img, save_image, tensor_to_image
from .config import DEFAULT_CONFIG, get_project_root
from .save_utils import save_result_with_metadata, append_to_metrics_csv
from .metrics import MetricsComputer
from pathlib import Path
import time


def gram_matrix(x: torch.Tensor, should_normalize: bool = True) -> torch.Tensor:
    """
    Compute Gram matrix (reimplementation if parent utils not available).
    
    Args:
        x: Feature tensor of shape (B, C, H, W)
        should_normalize: If True, normalize by C*H*W
    
    Returns:
        Gram matrix of shape (B, C, C)
    """
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y: torch.Tensor) -> torch.Tensor:
    """Compute total variation loss."""
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


def build_gatys_loss(
    feature_extractor: VGGFeatureExtractor,
    optimizing_img: torch.Tensor,
    target_content: torch.Tensor,
    target_grams: Dict[str, torch.Tensor],
    config: Dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build Gatys-style loss function.
    
    Args:
        feature_extractor: VGGFeatureExtractor
        optimizing_img: Image being optimized (1, 3, H, W)
        target_content: Target content features
        target_grams: Dictionary of target Gram matrices per layer
        config: Configuration dict
    
    Returns:
        Tuple of (total_loss, content_loss, style_loss, tv_loss)
    """
    # Extract features from optimizing image (needs gradients!)
    all_features = feature_extractor.extract_features(optimizing_img, requires_grad=True)
    content_layer_name = feature_extractor.layer_names[feature_extractor.content_idx]
    current_content = all_features[content_layer_name].squeeze(0)
    
    # Content loss
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content, current_content)
    
    # Style loss (Gram matrices) - match original implementation
    style_loss = 0.0
    style_layer_names = feature_extractor.get_style_layer_names()
    
    for layer_name in style_layer_names:
        if layer_name not in target_grams:
            continue
        
        current_features = all_features[layer_name]
        current_gram = gram_matrix(current_features)
        target_gram = target_grams[layer_name]
        
        # MSE loss on Gram matrices - use 'sum' reduction as in original
        style_loss += torch.nn.MSELoss(reduction='sum')(target_gram[0], current_gram[0])
    
    # Average over number of style layers (as in original)
    if len(target_grams) > 0:
        style_loss /= len(target_grams)
    
    # Total variation loss
    tv_loss = total_variation(optimizing_img)
    
    # Total loss
    total_loss = (config['content_weight'] * content_loss + 
                  config['style_weight'] * style_loss + 
                  config['tv_weight'] * tv_loss)
    
    return total_loss, content_loss, style_loss, tv_loss


def gatys_style_transfer(
    content_img_path: str,
    style_img_path: str,
    output_path: Optional[str] = None,
    config: Optional[Dict] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Run baseline Gatys style transfer.
    
    Args:
        content_img_path: Path to content image
        style_img_path: Path to style image
        output_path: Optional path to save result
        config: Configuration dict (uses defaults if None)
        progress_callback: Optional callback(iteration, loss_dict) for progress
    
    Returns:
        Tuple of (result_image_tensor, metrics_dict)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load images
    content_img = prepare_img(content_img_path, config['height'], device)
    style_img = prepare_img(style_img_path, config['height'], device)
    
    # Initialize feature extractor
    feature_extractor = VGGFeatureExtractor(model_name=config['model'], device=device)
    
    # Extract target representations
    target_content = feature_extractor.get_content_features(content_img).squeeze(0)
    style_features = feature_extractor.get_style_features(style_img)
    
    # Compute target Gram matrices
    target_grams = {}
    for layer_name, features in style_features.items():
        target_grams[layer_name] = gram_matrix(features)
    
    # Initialize optimizing image
    init_method = config.get('init_method', 'content')
    if init_method == 'random':
        gaussian_noise = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise).float().to(device)
    elif init_method == 'content':
        init_img = content_img.clone()
    else:  # style
        style_resized = prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_resized
    
    optimizing_img = Variable(init_img, requires_grad=True)
    
    # Setup optimizer
    optimizer_name = config.get('optimizer', 'lbfgs')
    iterations = config.get('iterations', 1000) if optimizer_name == 'lbfgs' else config.get('adam_iterations', 3000)
    
    metrics = {
        'content_loss': [],
        'style_loss': [],
        'tv_loss': [],
        'total_loss': [],
        'start_time': time.time()
    }
    
    if optimizer_name == 'adam':
        optimizer = Adam((optimizing_img,), lr=config.get('lr', 1e1))
        
        for cnt in range(iterations):
            optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_gatys_loss(
                feature_extractor, optimizing_img, target_content, target_grams, config
            )
            total_loss.backward()
            optimizer.step()
            
            # Clamp values
            with torch.no_grad():
                optimizing_img.data.clamp_(0, 255)
            
            metrics['total_loss'].append(total_loss.item())
            metrics['content_loss'].append(content_loss.item())
            metrics['style_loss'].append(style_loss.item())
            metrics['tv_loss'].append(tv_loss.item())
            
            if progress_callback:
                progress_callback(cnt, {
                    'total_loss': total_loss.item(),
                    'content_loss': content_loss.item(),
                    'style_loss': style_loss.item(),
                    'tv_loss': tv_loss.item()
                })
    
    elif optimizer_name == 'lbfgs':
        optimizer = LBFGS((optimizing_img,), max_iter=iterations, line_search_fn='strong_wolfe')
        cnt = 0
        
        def closure():
            nonlocal cnt
            optimizer.zero_grad()
            
            total_loss, content_loss, style_loss, tv_loss = build_gatys_loss(
                feature_extractor, optimizing_img, target_content, target_grams, config
            )
            
            total_loss.backward()
            
            # Log metrics (outside gradient computation)
            with torch.no_grad():
                metrics['total_loss'].append(total_loss.item())
                metrics['content_loss'].append(content_loss.item())
                metrics['style_loss'].append(style_loss.item())
                metrics['tv_loss'].append(tv_loss.item())
                
                if progress_callback:
                    progress_callback(cnt, {
                        'total_loss': total_loss.item(),
                        'content_loss': content_loss.item(),
                        'style_loss': style_loss.item(),
                        'tv_loss': tv_loss.item()
                    })
            
            cnt += 1
            return total_loss
        
        optimizer.step(closure)
    
    # Final result
    result = optimizing_img.detach()
    
    # Auto-save final image and metadata
    runtime = time.time() - (metrics.get('start_time', time.time()))
    
    # Prepare metadata
    metadata = {
        'mode': 'baseline',
        'method': 'gatys',
        'content_image': content_img_path,
        'style_image': style_img_path,
        'hyperparameters': {
            'content_weight': config.get('content_weight', 1e5),
            'style_weight': config.get('style_weight', 3e4),
            'tv_weight': config.get('tv_weight', 1e0),
            'optimizer': config.get('optimizer', 'lbfgs'),
            'iterations': len(metrics.get('total_loss', [])),
            'init_method': config.get('init_method', 'content'),
            'height': config.get('height', 400)
        },
        'final_losses': {
            'total_loss': metrics['total_loss'][-1] if metrics['total_loss'] else None,
            'content_loss': metrics['content_loss'][-1] if metrics['content_loss'] else None,
            'style_loss': metrics['style_loss'][-1] if metrics['style_loss'] else None,
            'tv_loss': metrics['tv_loss'][-1] if metrics['tv_loss'] else None
        },
        'runtime_seconds': runtime
    }
    
    # Compute additional metrics if possible
    try:
        metrics_computer = MetricsComputer(device=device, model_name=config.get('model', 'vgg19'))
        content_img_tensor = prepare_img(content_img_path, config.get('height', 400), device)
        style_img_tensor = prepare_img(style_img_path, config.get('height', 400), device)
        all_metrics = metrics_computer.compute_all_metrics(result, content_img_tensor, style_img_tensor, None, runtime)
        metadata.update(all_metrics)
    except Exception as e:
        # If metrics fail, continue without them
        pass
    
    # Auto-save with deterministic filename
    image_path, json_path = save_result_with_metadata(result, metadata)
    
    # Append to CSV
    try:
        append_to_metrics_csv(metadata)
    except Exception as e:
        # Log but don't fail
        print(f"Warning: Could not append to CSV: {e}")
    
    # Legacy support: if output_path provided, also save there
    if output_path:
        save_image(result, output_path, denormalize=True)
    
    return result, metrics

