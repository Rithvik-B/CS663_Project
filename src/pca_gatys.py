"""
PCA-Gatys style transfer: uses mixed covariance matrices as style target.
"""

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import argparse
import os
from typing import Dict, Optional, Tuple, List

from .vgg_features import VGGFeatureExtractor
from .pca_code import extract_pca_codes, PCACode
from .mixing import mix_style_codes, mix_gram_matrices, gram_matrix
from .io_utils import prepare_img, save_image
from .config import DEFAULT_CONFIG, get_project_root
from .save_utils import save_result_with_metadata, append_to_metrics_csv
from .metrics import MetricsComputer
from .utils import get_device
from pathlib import Path
import time


def total_variation(y: torch.Tensor) -> torch.Tensor:
    """Compute total variation loss."""
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


def build_pca_loss(
    feature_extractor: VGGFeatureExtractor,
    optimizing_img: torch.Tensor,
    target_content: torch.Tensor,
    target_covariances: Dict[str, torch.Tensor],  # Mixed covariance matrices
    target_means: Dict[str, torch.Tensor],  # Mixed means
    config: Dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build loss function using covariance matrices as style target.
    
    Args:
        feature_extractor: VGGFeatureExtractor
        optimizing_img: Image being optimized
        target_content: Target content features
        target_covariances: Dictionary of target covariance matrices per layer
        target_means: Dictionary of target means per layer
        config: Configuration dict
    
    Returns:
        Tuple of (total_loss, content_loss, style_loss, tv_loss)
    """
    from .pca_code import compute_covariance
    
    # Extract features from optimizing image (needs gradients!)
    all_features = feature_extractor.extract_features(optimizing_img, requires_grad=True)
    content_layer_name = feature_extractor.layer_names[feature_extractor.content_idx]
    current_content = all_features[content_layer_name].squeeze(0)
    
    # Content loss
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content, current_content)
    
    # Style loss (covariance matrices)
    style_loss = 0.0
    style_layer_names = feature_extractor.get_style_layer_names()
    
    num_layers = 0
    for layer_name in style_layer_names:
        if layer_name not in target_covariances:
            continue
        
        current_features = all_features[layer_name]
        
        # Compute current covariance
        current_cov, current_mean = compute_covariance(current_features, center=True)
        target_cov = target_covariances[layer_name]
        target_mean = target_means[layer_name]
        
        # Covariance loss: MSE (sum of squared differences)
        cov_diff = current_cov - target_cov
        cov_loss = torch.sum(cov_diff ** 2)
        
        # Mean loss (optional, usually small)
        mean_diff = current_mean - target_mean
        mean_loss = torch.sum(mean_diff ** 2)
        
        # Add to style loss (similar to Gram matrix loss structure)
        style_loss += cov_loss + mean_loss
        num_layers += 1
    
    # Average over number of style layers (matching Gram matrix loss structure)
    if num_layers > 0:
        style_loss /= num_layers
    
    # Total variation loss
    tv_loss = total_variation(optimizing_img)
    
    # Total loss
    total_loss = (config['content_weight'] * content_loss + 
                  config['style_weight'] * style_loss + 
                  config['tv_weight'] * tv_loss)
    
    return total_loss, content_loss, style_loss, tv_loss


def pca_gatys_style_transfer(
    content_img_path: str,
    style1_img_path: str,
    style2_img_path: str,
    alpha: float,
    mixing_method: str = 'joint',
    output_path: Optional[str] = None,
    config: Optional[Dict] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Run PCA-Gatys style transfer with mixed styles.
    
    Args:
        content_img_path: Path to content image
        style1_img_path: Path to first style image
        style2_img_path: Path to second style image
        alpha: Mixing coefficient (0.0 = style2, 1.0 = style1)
        mixing_method: 'simple', 'joint', 'covariance-linear', or 'gram-linear'
        output_path: Optional path to save result
        config: Configuration dict
        progress_callback: Optional callback for progress
    
    Returns:
        Tuple of (result_image_tensor, metrics_dict)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Use get_device utility which properly handles CUDA availability
    device = get_device(config.get('device'))
    
    # Load images
    content_img = prepare_img(content_img_path, config['height'], device)
    style1_img = prepare_img(style1_img_path, config['height'], device)
    style2_img = prepare_img(style2_img_path, config['height'], device)
    
    # Initialize feature extractor
    feature_extractor = VGGFeatureExtractor(model_name=config['model'], device=device)
    
    # Extract target content
    target_content = feature_extractor.get_content_features(content_img).squeeze(0)
    
    # Initialize optimizing image first (needed for all methods)
    init_method = config.get('init_method', 'content')
    if init_method == 'random':
        gaussian_noise = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise).float().to(device)
    elif init_method == 'content':
        init_img = content_img.clone()
    else:  # style
        style_resized = prepare_img(style1_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_resized
    
    optimizing_img = Variable(init_img, requires_grad=True)
    
    # Handle different mixing methods
    if mixing_method == 'gram-linear':
        # Use Gram matrix mixing (baseline)
        style1_features = feature_extractor.get_style_features(style1_img)
        style2_features = feature_extractor.get_style_features(style2_img)
        
        grams1 = {name: gram_matrix(features) for name, features in style1_features.items()}
        grams2 = {name: gram_matrix(features) for name, features in style2_features.items()}
        target_grams = mix_gram_matrices(grams1, grams2, alpha, config.get('per_layer_alpha'))
        
        # Use Gram-based loss (similar to Gatys)
        return _optimize_with_gram_target(
            feature_extractor, optimizing_img, target_content, target_grams, config, progress_callback
        )
    
    else:
        # PCA-based mixing
        # Extract PCA codes (with caching for performance)
        cache_dir = os.path.join(get_project_root(), 'data', 'pca_cache')
        codes1 = extract_pca_codes(feature_extractor, style1_img, cache_dir=cache_dir, style_img_path=style1_img_path)
        codes2 = extract_pca_codes(feature_extractor, style2_img, cache_dir=cache_dir, style_img_path=style2_img_path)
        
        # Mix codes
        per_layer_alpha = config.get('per_layer_alpha')
        mixed_codes = mix_style_codes(codes1, codes2, alpha, mixing_method, per_layer_alpha)
        
        # Extract target covariances and means
        target_covariances = {name: code.C for name, code in mixed_codes.items()}
        target_means = {name: code.mean for name, code in mixed_codes.items()}
    
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
            total_loss, content_loss, style_loss, tv_loss = build_pca_loss(
                feature_extractor, optimizing_img, target_content, 
                target_covariances, target_means, config
            )
            total_loss.backward()
            optimizer.step()
            
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
            
            total_loss, content_loss, style_loss, tv_loss = build_pca_loss(
                feature_extractor, optimizing_img, target_content,
                target_covariances, target_means, config
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
    
    result = optimizing_img.detach()
    
    # Auto-save final image and metadata
    runtime = time.time() - metrics.get('start_time', time.time())
    
    # Determine mode and method name
    if mixing_method == 'gram-linear':
        mode = 'pca_mix'
        method_display = 'gram-linear'
    else:
        mode = 'pca_mix'
        method_display = mixing_method
    
    # Prepare metadata
    metadata = {
        'mode': mode,
        'method': method_display,
        'content_image': content_img_path,
        'style1_image': style1_img_path,
        'style2_image': style2_img_path,
        'alpha': alpha,
        'hyperparameters': {
            'content_weight': config.get('content_weight', 1e5),
            'style_weight': config.get('style_weight', 3e4),
            'tv_weight': config.get('tv_weight', 1e0),
            'optimizer': config.get('optimizer', 'lbfgs'),
            'iterations': len(metrics.get('total_loss', [])),
            'init_method': config.get('init_method', 'content'),
            'height': config.get('height', 400),
            'mixing_method': mixing_method
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
        style1_img_tensor = prepare_img(style1_img_path, config.get('height', 400), device)
        style2_img_tensor = prepare_img(style2_img_path, config.get('height', 400), device)
        all_metrics = metrics_computer.compute_all_metrics(
            result, content_img_tensor, style1_img_tensor, style2_img_tensor, runtime
        )
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


def _optimize_with_gram_target(
    feature_extractor: VGGFeatureExtractor,
    optimizing_img: torch.Tensor,
    target_content: torch.Tensor,
    target_grams: Dict[str, torch.Tensor],
    config: Dict,
    progress_callback: Optional[callable]
) -> Tuple[torch.Tensor, Dict]:
    """Helper to optimize with Gram matrix targets (for gram-linear baseline)."""
    from .gatys import build_gatys_loss
    
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
            
            with torch.no_grad():
                optimizing_img.data.clamp_(0, 255)
            
            metrics['total_loss'].append(total_loss.item())
            metrics['content_loss'].append(content_loss.item())
            metrics['style_loss'].append(style_loss.item())
            metrics['tv_loss'].append(tv_loss.item())
            
            if progress_callback:
                progress_callback(cnt, metrics)
    
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
    
    return optimizing_img.detach(), metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA-Gatys Style Transfer")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style1", type=str, required=True, help="Path to first style image")
    parser.add_argument("--style2", type=str, required=True, help="Path to second style image")
    parser.add_argument("--alpha", type=float, default=0.5, help="Mixing coefficient (0.0-1.0)")
    parser.add_argument("--method", type=str, default="joint", 
                       choices=["simple", "joint", "covariance-linear", "gram-linear"],
                       help="Mixing method")
    parser.add_argument("--out", type=str, help="Output image path")
    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--height", type=int, default=400, help="Image height")
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "adam"])
    parser.add_argument("--content_weight", type=float, default=1e5)
    parser.add_argument("--style_weight", type=float, default=3e4)
    parser.add_argument("--tv_weight", type=float, default=1e0)
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        'iterations': args.iters if args.optimizer == 'lbfgs' else None,
        'adam_iterations': args.iters if args.optimizer == 'adam' else None,
        'height': args.height,
        'optimizer': args.optimizer,
        'content_weight': args.content_weight,
        'style_weight': args.style_weight,
        'tv_weight': args.tv_weight
    })
    
    result, metrics = pca_gatys_style_transfer(
        args.content, args.style1, args.style2,
        alpha=args.alpha,
        mixing_method=args.method,
        output_path=args.out,
        config=config
    )
    
    if args.out:
        print(f"Result saved to: {args.out}")
    print(f"Final loss: {metrics['total_loss'][-1]:.4f}")
