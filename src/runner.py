"""
Runner API: executes NST jobs with automatic saving of images, metrics, and metadata.
"""

import os
import json
import time
import pathlib
import tempfile
import shutil
from typing import Dict, Optional, Tuple
from datetime import datetime
import torch
import pandas as pd
import numpy as np

from .io_utils import save_image, prepare_img, tensor_to_image
from .metrics import MetricsComputer
from .config import DEFAULT_CONFIG
from .utils import set_seed


def _atomic_write(filepath: str, content: str, mode: str = 'w'):
    """Write file atomically using temp file + os.replace."""
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    fd, tmp_path = tempfile.mkstemp(dir=dirname, suffix='.tmp')
    try:
        with os.fdopen(fd, mode) as f:
            f.write(content)
        os.replace(tmp_path, filepath)
    except Exception:
        os.unlink(tmp_path)
        raise


def _atomic_write_json(filepath: str, data: dict):
    """Write JSON file atomically."""
    content = json.dumps(data, indent=2, default=str)
    _atomic_write(filepath, content)


def _create_run_folder(config: Dict) -> pathlib.Path:
    """Create deterministic run folder name."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    content_name = pathlib.Path(config['content_img_path']).stem if config.get('content_img_path') else 'unknown'
    style1_name = pathlib.Path(config['style1_img_path']).stem if config.get('style1_img_path') else 'unknown'
    style2_name = pathlib.Path(config['style2_img_path']).stem if config.get('style2_img_path') else 'unknown'
    
    method = config.get('mixing_method', 'gatys')
    alpha = config.get('alpha', 0.5)
    seed = config.get('seed', 42)
    
    # Normalize method name for folder
    method_normalized = method
    if method == 'pca_joint':
        method_normalized = 'pca_joint'
    elif method == 'pca_simple':
        method_normalized = 'pca_simple'
    elif method == 'joint':
        method_normalized = 'pca_joint'
    elif method == 'simple':
        method_normalized = 'pca_simple'
    
    folder_name = f"{timestamp}__{content_name}__{style1_name}__{style2_name}__{method_normalized}_a{alpha:.2f}_s{seed}"
    
    base_dir = pathlib.Path("data/outputs/runs")
    run_folder = base_dir / folder_name
    
    run_folder.mkdir(parents=True, exist_ok=True)
    (run_folder / "images").mkdir(exist_ok=True)
    (run_folder / "metrics").mkdir(exist_ok=True)
    
    return run_folder


def _save_meta_start(run_folder: pathlib.Path, config: Dict):
    """Save initial metadata."""
    meta = {
        'content_img_path': config.get('content_img_path'),
        'style1_img_path': config.get('style1_img_path'),
        'style2_img_path': config.get('style2_img_path'),
        'mixing_method': config.get('mixing_method', 'gatys'),
        'alpha': config.get('alpha', 0.5),
        'seed': config.get('seed', 42),
        'iterations_requested': config.get('iterations', 1000),
        'snapshot_interval': config.get('snapshot_interval', 10),
        'device': str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
        'start_ts_utc': datetime.utcnow().isoformat() + 'Z',
        'end_ts_utc': None,
        'iterations_completed': None
    }
    
    _atomic_write_json(run_folder / "meta.json", meta)
    return meta


def _update_meta_end(run_folder: pathlib.Path, iterations_completed: int):
    """Update metadata with completion info."""
    meta_path = run_folder / "meta.json"
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    meta['end_ts_utc'] = datetime.utcnow().isoformat() + 'Z'
    meta['iterations_completed'] = iterations_completed
    
    _atomic_write_json(meta_path, meta)


def _save_iteration_metrics(
    run_folder: pathlib.Path,
    iteration: int,
    elapsed_s: float,
    step_time_s: float,
    content_loss: float,
    style_loss: float,
    total_loss: float,
    lpips: Optional[float] = None,
    ssim: Optional[float] = None,
    gram_dist: Optional[float] = None
):
    """Append row to metrics CSV."""
    csv_path = run_folder / "metrics" / "metrics_summary.csv"
    
    row = {
        'iteration': iteration,
        'elapsed_s': elapsed_s,
        'step_time_s': step_time_s,
        'content_loss': content_loss,
        'style_loss': style_loss,
        'total_loss': total_loss,
        'lpips': lpips if lpips is not None else '',
        'ssim': ssim if ssim is not None else '',
        'gram_dist': gram_dist if gram_dist is not None else ''
    }
    
    # Read existing CSV or create new
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    # Write atomically
    tmp_path = csv_path.with_suffix('.tmp')
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)


def _save_snapshot(run_folder: pathlib.Path, iteration: int, image: torch.Tensor):
    """Save iteration snapshot image."""
    filename = f"iter_{iteration:03d}.png"
    image_path = run_folder / "images" / filename
    
    # Save atomically
    tmp_path = image_path.with_suffix('.tmp.png')
    save_image(image, str(tmp_path), denormalize=True)
    os.replace(tmp_path, image_path)


def _save_final_metrics(run_folder: pathlib.Path, metrics: Dict):
    """Save final metrics JSON."""
    final_metrics = {
        'iteration': metrics.get('iteration', 0),
        'elapsed_s': metrics.get('elapsed_s', 0),
        'content_loss': metrics.get('content_loss', 0),
        'style_loss': metrics.get('style_loss', 0),
        'total_loss': metrics.get('total_loss', 0),
        'lpips': metrics.get('lpips'),
        'ssim': metrics.get('ssim'),
        'gram_dist': metrics.get('gram_dist'),
        'summary': f"Final loss: {metrics.get('total_loss', 0):.4f}"
    }
    
    _atomic_write_json(run_folder / "metrics" / "final_metrics.json", final_metrics)


class IterationCallback:
    """Callback for tracking iterations and saving metrics."""
    
    def __init__(self, run_folder: pathlib.Path, config: Dict, metrics_computer: Optional[MetricsComputer] = None):
        self.run_folder = run_folder
        self.config = config
        self.metrics_computer = metrics_computer
        self.snapshot_interval = config.get('snapshot_interval', 10)
        self.max_iters = config.get('iterations', 1000)
        self.start_time = time.time()
        self.last_iter_time = time.time()
        self.content_img = None
        self.style1_img = None
        self.style2_img = None
        
    def set_reference_images(self, content_img, style1_img=None, style2_img=None):
        """Set reference images for metrics computation."""
        self.content_img = content_img
        self.style1_img = style1_img
        self.style2_img = style2_img
    
    def __call__(self, iteration: int, loss_dict: Dict, current_image: torch.Tensor):
        """Called each iteration."""
        current_time = time.time()
        elapsed_s = current_time - self.start_time
        step_time_s = current_time - self.last_iter_time
        self.last_iter_time = current_time
        
        # Compute metrics if available
        lpips = None
        ssim = None
        gram_dist = None
        
        if self.metrics_computer and self.content_img is not None and iteration == self.max_iters:
            # Only compute expensive metrics on final iteration
            try:
                all_metrics = self.metrics_computer.compute_all_metrics(
                    current_image, self.content_img, self.style1_img, self.style2_img
                )
                lpips = all_metrics.get('lpips_content')
                ssim = all_metrics.get('ssim_content')
                gram_dist = all_metrics.get('gram_dist_style1_avg')
            except Exception:
                pass  # Metrics computation failed, continue
        
        # Save metrics
        _save_iteration_metrics(
            self.run_folder, iteration, elapsed_s, step_time_s,
            loss_dict.get('content_loss', 0),
            loss_dict.get('style_loss', 0),
            loss_dict.get('total_loss', 0),
            lpips, ssim, gram_dist
        )
        
        # Save snapshot if needed
        if iteration % self.snapshot_interval == 0 or iteration == self.max_iters:
            _save_snapshot(self.run_folder, iteration, current_image)


def run_once(config: Dict) -> pathlib.Path:
    """
    Runs a single NST job with given config.
    
    Args:
        config: Configuration dictionary with:
            - content_img_path, style1_img_path, style2_img_path (paths)
            - mixing_method: 'gatys', 'joint', 'simple', 'gram-linear', 'covariance-linear'
            - alpha: mixing coefficient (0.0-1.0)
            - iterations: max iterations
            - snapshot_interval: save snapshot every N iterations
            - seed: random seed
            - device: 'cuda' or 'cpu'
            - All other DEFAULT_CONFIG options
    
    Returns:
        pathlib.Path to the created run folder
    """
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Create run folder
    run_folder = _create_run_folder(config)
    
    # Save initial metadata
    meta = _save_meta_start(run_folder, config)
    
    try:
        # Prepare config for style transfer
        transfer_config = DEFAULT_CONFIG.copy()
        transfer_config.update({
            'iterations': config.get('iterations', 1000),
            'height': config.get('height', 400),
            'device': config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            'optimizer': config.get('optimizer', 'lbfgs'),
            'content_weight': config.get('content_weight', 1e5),
            'style_weight': config.get('style_weight', 3e4),
            'tv_weight': config.get('tv_weight', 1e0),
        })
        
        # Initialize metrics computer
        device = torch.device(transfer_config['device'])
        metrics_computer = MetricsComputer(device=device, model_name=transfer_config['model'])
        
        # Create callback
        callback = IterationCallback(run_folder, config, metrics_computer)
        
        # Load reference images for metrics
        content_img = prepare_img(config['content_img_path'], transfer_config['height'], device)
        style1_img = prepare_img(config['style1_img_path'], transfer_config['height'], device) if config.get('style1_img_path') else None
        style2_img = prepare_img(config['style2_img_path'], transfer_config['height'], device) if config.get('style2_img_path') else None
        
        callback.set_reference_images(content_img, style1_img, style2_img)
        
        # Determine method (handle UI naming: pca_joint -> joint, pca_simple -> simple)
        mixing_method = config.get('mixing_method', 'gatys')
        if mixing_method == 'pca_joint':
            mixing_method = 'joint'
        elif mixing_method == 'pca_simple':
            mixing_method = 'simple'
        
        # Run style transfer with exact iteration control
        if mixing_method == 'gatys' or not config.get('style2_img_path'):
            result, metrics = _run_gatys_with_callback(
                config['content_img_path'],
                config['style1_img_path'],
                callback,
                transfer_config
            )
        else:
            result, metrics = _run_pca_with_callback(
                config['content_img_path'],
                config['style1_img_path'],
                config['style2_img_path'],
                config.get('alpha', 0.5),
                mixing_method,
                callback,
                transfer_config
            )
        
        # Save final image (always saved as iter_{max_iters:03d}.png)
        max_iters = config.get('iterations', 1000)
        _save_snapshot(run_folder, max_iters, result)
        
        # Compute and save final metrics
        final_metrics = {
            'iteration': max_iters,
            'elapsed_s': time.time() - callback.start_time,
            'content_loss': metrics.get('content_loss', [0])[-1] if metrics.get('content_loss') else 0,
            'style_loss': metrics.get('style_loss', [0])[-1] if metrics.get('style_loss') else 0,
            'total_loss': metrics.get('total_loss', [0])[-1] if metrics.get('total_loss') else 0,
        }
        
        # Compute final quality metrics
        try:
            all_metrics = metrics_computer.compute_all_metrics(
                result, content_img, style1_img, style2_img
            )
            final_metrics['lpips'] = all_metrics.get('lpips_content')
            final_metrics['ssim'] = all_metrics.get('ssim_content')
            final_metrics['gram_dist'] = all_metrics.get('gram_dist_style1_avg')
        except Exception:
            pass
        
        _save_final_metrics(run_folder, final_metrics)
        
        # Update metadata
        _update_meta_end(run_folder, max_iters)
        
    except Exception as e:
        # Update metadata with error
        meta = _save_meta_start(run_folder, config)
        meta['error'] = str(e)
        _atomic_write_json(run_folder / "meta.json", meta)
        raise
    
    return run_folder


def _run_gatys_with_callback(
    content_path: str,
    style_path: str,
    callback: IterationCallback,
    config: Dict
) -> Tuple[torch.Tensor, Dict]:
    """Run Gatys with iteration callback and exact iteration control."""
    from .gatys import build_gatys_loss
    from .vgg_features import VGGFeatureExtractor
    from .io_utils import prepare_img
    from .mixing import gram_matrix
    from torch.optim import Adam, LBFGS
    from torch.autograd import Variable
    import numpy as np
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load images
    content_img = prepare_img(content_path, config['height'], device)
    style_img = prepare_img(style_path, config['height'], device)
    
    # Initialize feature extractor
    feature_extractor = VGGFeatureExtractor(model_name=config['model'], device=device)
    
    # Extract targets (once)
    with torch.no_grad():
        target_content = feature_extractor.get_content_features(content_img).squeeze(0)
        style_features = feature_extractor.get_style_features(style_img)
        target_grams = {name: gram_matrix(features) for name, features in style_features.items()}
    
    # Initialize optimizing image
    init_method = config.get('init_method', 'content')
    if init_method == 'random':
        gaussian_noise = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise).float().to(device)
    elif init_method == 'content':
        init_img = content_img.clone()
    else:
        init_img = prepare_img(style_path, config['height'], device)
    
    optimizing_img = Variable(init_img, requires_grad=True)
    
    # Setup optimizer
    optimizer_name = config.get('optimizer', 'lbfgs')
    iterations = config.get('iterations', 1000)
    
    metrics = {'content_loss': [], 'style_loss': [], 'tv_loss': [], 'total_loss': []}
    
    if optimizer_name == 'adam':
        optimizer = Adam((optimizing_img,), lr=config.get('lr', 1e1))
        
        for it in range(1, iterations + 1):
            optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_gatys_loss(
                feature_extractor, optimizing_img, target_content, target_grams, config
            )
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                optimizing_img.data.clamp_(0, 255)
            
            loss_dict = {
                'content_loss': content_loss.item(),
                'style_loss': style_loss.item(),
                'tv_loss': tv_loss.item(),
                'total_loss': total_loss.item()
            }
            
            metrics['total_loss'].append(loss_dict['total_loss'])
            metrics['content_loss'].append(loss_dict['content_loss'])
            metrics['style_loss'].append(loss_dict['style_loss'])
            metrics['tv_loss'].append(loss_dict['tv_loss'])
            
            callback(it, loss_dict, optimizing_img.detach())
    
    elif optimizer_name == 'lbfgs':
        # LBFGS with exact iteration control: one step per outer iteration
        optimizer = LBFGS((optimizing_img,), max_iter=1, line_search_fn='strong_wolfe')
        
        for it in range(1, iterations + 1):
            def closure():
                optimizer.zero_grad()
                total_loss, content_loss, style_loss, tv_loss = build_gatys_loss(
                    feature_extractor, optimizing_img, target_content, target_grams, config
                )
                total_loss.backward()
                return total_loss
            
            optimizer.step(closure)
            
            with torch.no_grad():
                optimizing_img.data.clamp_(0, 255)
            
            # Compute loss for metrics (no grad)
            with torch.no_grad():
                total_loss, content_loss, style_loss, tv_loss = build_gatys_loss(
                    feature_extractor, optimizing_img, target_content, target_grams, config
                )
            
            loss_dict = {
                'content_loss': content_loss.item(),
                'style_loss': style_loss.item(),
                'tv_loss': tv_loss.item(),
                'total_loss': total_loss.item()
            }
            
            metrics['total_loss'].append(loss_dict['total_loss'])
            metrics['content_loss'].append(loss_dict['content_loss'])
            metrics['style_loss'].append(loss_dict['style_loss'])
            metrics['tv_loss'].append(loss_dict['tv_loss'])
            
            callback(it, loss_dict, optimizing_img.detach())
    
    return optimizing_img.detach(), metrics


def _run_pca_with_callback(
    content_path: str,
    style1_path: str,
    style2_path: str,
    alpha: float,
    mixing_method: str,
    callback: IterationCallback,
    config: Dict
) -> Tuple[torch.Tensor, Dict]:
    """Run PCA-Gatys with iteration callback and exact iteration control."""
    from .pca_gatys import build_pca_loss
    from .vgg_features import VGGFeatureExtractor
    from .pca_code import extract_pca_codes
    from .mixing import mix_style_codes, mix_gram_matrices, gram_matrix
    from .io_utils import prepare_img
    from torch.optim import Adam, LBFGS
    from torch.autograd import Variable
    import numpy as np
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load images
    content_img = prepare_img(content_path, config['height'], device)
    style1_img = prepare_img(style1_path, config['height'], device)
    style2_img = prepare_img(style2_path, config['height'], device)
    
    # Initialize feature extractor
    feature_extractor = VGGFeatureExtractor(model_name=config['model'], device=device)
    
    # Extract targets (once)
    with torch.no_grad():
        target_content = feature_extractor.get_content_features(content_img).squeeze(0)
        
        if mixing_method == 'gram-linear':
            style1_features = feature_extractor.get_style_features(style1_img)
            style2_features = feature_extractor.get_style_features(style2_img)
            grams1 = {name: gram_matrix(features) for name, features in style1_features.items()}
            grams2 = {name: gram_matrix(features) for name, features in style2_features.items()}
            target_grams = mix_gram_matrices(grams1, grams2, alpha, None)
            target_covariances = None
            target_means = None
        else:
            codes1 = extract_pca_codes(feature_extractor, style1_img, use_cache=False)
            codes2 = extract_pca_codes(feature_extractor, style2_img, use_cache=False)
            mixed_codes = mix_style_codes(codes1, codes2, alpha, mixing_method, None)
            target_covariances = {name: code.C.detach() for name, code in mixed_codes.items()}
            target_means = {name: code.mean.detach() for name, code in mixed_codes.items()}
            target_grams = None
    
    # Initialize optimizing image
    init_method = config.get('init_method', 'content')
    if init_method == 'random':
        gaussian_noise = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise).float().to(device)
    elif init_method == 'content':
        init_img = content_img.clone()
    else:
        init_img = prepare_img(style1_path, config['height'], device)
    
    optimizing_img = Variable(init_img, requires_grad=True)
    
    # Setup optimizer
    optimizer_name = config.get('optimizer', 'lbfgs')
    iterations = config.get('iterations', 1000)
    
    metrics = {'content_loss': [], 'style_loss': [], 'tv_loss': [], 'total_loss': []}
    
    if optimizer_name == 'adam':
        optimizer = Adam((optimizing_img,), lr=config.get('lr', 1e1))
        
        for it in range(1, iterations + 1):
            optimizer.zero_grad()
            
            if target_grams is not None:
                from .gatys import build_gatys_loss
                total_loss, content_loss, style_loss, tv_loss = build_gatys_loss(
                    feature_extractor, optimizing_img, target_content, target_grams, config
                )
            else:
                total_loss, content_loss, style_loss, tv_loss = build_pca_loss(
                    feature_extractor, optimizing_img, target_content,
                    target_covariances, target_means, config
                )
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                optimizing_img.data.clamp_(0, 255)
            
            loss_dict = {
                'content_loss': content_loss.item(),
                'style_loss': style_loss.item(),
                'tv_loss': tv_loss.item(),
                'total_loss': total_loss.item()
            }
            
            metrics['total_loss'].append(loss_dict['total_loss'])
            metrics['content_loss'].append(loss_dict['content_loss'])
            metrics['style_loss'].append(loss_dict['style_loss'])
            metrics['tv_loss'].append(loss_dict['tv_loss'])
            
            callback(it, loss_dict, optimizing_img.detach())
    
    elif optimizer_name == 'lbfgs':
        # LBFGS with exact iteration control
        optimizer = LBFGS((optimizing_img,), max_iter=1, line_search_fn='strong_wolfe')
        
        for it in range(1, iterations + 1):
            def closure():
                optimizer.zero_grad()
                if target_grams is not None:
                    from .gatys import build_gatys_loss
                    total_loss, _, _, _ = build_gatys_loss(
                        feature_extractor, optimizing_img, target_content, target_grams, config
                    )
                else:
                    total_loss, _, _, _ = build_pca_loss(
                        feature_extractor, optimizing_img, target_content,
                        target_covariances, target_means, config
                    )
                total_loss.backward()
                return total_loss
            
            optimizer.step(closure)
            
            with torch.no_grad():
                optimizing_img.data.clamp_(0, 255)
            
            # Compute loss for metrics
            with torch.no_grad():
                if target_grams is not None:
                    from .gatys import build_gatys_loss
                    total_loss, content_loss, style_loss, tv_loss = build_gatys_loss(
                        feature_extractor, optimizing_img, target_content, target_grams, config
                    )
                else:
                    total_loss, content_loss, style_loss, tv_loss = build_pca_loss(
                        feature_extractor, optimizing_img, target_content,
                        target_covariances, target_means, config
                    )
            
            loss_dict = {
                'content_loss': content_loss.item(),
                'style_loss': style_loss.item(),
                'tv_loss': tv_loss.item(),
                'total_loss': total_loss.item()
            }
            
            metrics['total_loss'].append(loss_dict['total_loss'])
            metrics['content_loss'].append(loss_dict['content_loss'])
            metrics['style_loss'].append(loss_dict['style_loss'])
            metrics['tv_loss'].append(loss_dict['tv_loss'])
            
            callback(it, loss_dict, optimizing_img.detach())
    
    return optimizing_img.detach(), metrics

