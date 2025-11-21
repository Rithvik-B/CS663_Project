"""
Orchestrates batch runs for grids and metric CSV output.
"""

import os
import time
from typing import List, Dict, Optional, Tuple, Callable
import pandas as pd
from pathlib import Path

from .pca_gatys import pca_gatys_style_transfer
from .gatys import gatys_style_transfer
from .io_utils import create_alpha_grid, save_image, load_image, prepare_img, tensor_to_image
from .metrics import MetricsComputer
from .config import DEFAULT_CONFIG, get_project_root
from .save_utils import log_batch_run
import time


def run_alpha_grid(
    content_img_path: str,
    style1_img_path: str,
    style2_img_path: str,
    alphas: List[float],
    methods: List[str],
    output_dir: str,
    config: Optional[Dict] = None,
    compute_metrics: bool = True,
    progress_callback: Optional[Callable] = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Run style transfer across multiple alpha values and methods, create grids.
    
    Args:
        content_img_path: Path to content image
        style1_img_path: Path to first style image
        style2_img_path: Path to second style image
        alphas: List of alpha values to try
        methods: List of methods ['pca-joint', 'pca-simple', 'gram-linear', 'cov-linear', 'gatys-style1', 'gatys-style2']
        output_dir: Directory to save results
        config: Configuration dict
        compute_metrics: If True, compute and save metrics
    
    Returns:
        Tuple of (metrics DataFrame, dict mapping method -> grid image path)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract names for grid filename
    content_name = Path(content_img_path).stem
    style1_name = Path(style1_img_path).stem
    style2_name = Path(style2_img_path).stem
    
    # Load images once
    from .io_utils import prepare_img
    from .utils import get_device
    device = get_device(config.get('device'))
    
    content_img = prepare_img(content_img_path, config['height'], device)
    style1_img = prepare_img(style1_img_path, config['height'], device)
    style2_img = prepare_img(style2_img_path, config['height'], device)
    
    # Convert to numpy for grid display
    from .io_utils import tensor_to_image
    content_np = tensor_to_image(content_img, denormalize=True)
    style1_np = tensor_to_image(style1_img, denormalize=True)
    style2_np = tensor_to_image(style2_img, denormalize=True)
    
    all_results = []
    grid_paths = {}
    
    metrics_computer = MetricsComputer(device=device, model_name=config['model']) if compute_metrics else None
    
    total_runs = len(methods) * len(alphas)
    current_run = 0
    
    for method in methods:
        method_results = []
        
        if progress_callback:
            progress_callback(f"Starting method: {method}", current_run, total_runs, None)
        
        for alpha in alphas:
            current_run += 1
            print(f"Running {method} with alpha={alpha:.2f} ({current_run}/{total_runs})")
            
            if progress_callback:
                progress_callback(f"{method} (α={alpha:.2f})", current_run, total_runs, None)
            
            start_time = time.time()
            
            # Generate result (auto-saves image + JSON + CSV)
            try:
                log_batch_run(f"Starting {method} with alpha={alpha:.2f} for {Path(content_img_path).name}")
                
                # Progress callback for individual runs
                def run_progress_callback(iteration, loss_dict):
                    if progress_callback:
                        progress_callback(
                            f"{method} (α={alpha:.2f}) - Iter {iteration}",
                            current_run, total_runs,
                            loss_dict
                        )
                
                if method == 'gatys-style1':
                    result, _ = gatys_style_transfer(
                        content_img_path, style1_img_path,
                        output_path=None, config=config,
                        progress_callback=run_progress_callback
                    )
                elif method == 'gatys-style2':
                    result, _ = gatys_style_transfer(
                        content_img_path, style2_img_path,
                        output_path=None, config=config,
                        progress_callback=run_progress_callback
                    )
                else:
                    # Map method names
                    mixing_method_map = {
                        'pca-joint': 'joint',
                        'pca-simple': 'simple',
                        'gram-linear': 'gram-linear',
                        'cov-linear': 'covariance-linear'
                    }
                    mixing_method = mixing_method_map.get(method, 'joint')
                    
                    result, _ = pca_gatys_style_transfer(
                        content_img_path, style1_img_path, style2_img_path,
                        alpha=alpha, mixing_method=mixing_method,
                        output_path=None, config=config,
                        progress_callback=run_progress_callback
                    )
                
                runtime = time.time() - start_time
                
                # Find the saved file (most recent)
                output_base_dir = os.path.join(get_project_root(), 'data', 'outputs')
                output_files = sorted(Path(output_base_dir).glob("*.jpg"), key=os.path.getmtime, reverse=True)
                
                if output_files:
                    saved_path = str(output_files[0])
                    log_batch_run(f"Completed {method} alpha={alpha:.2f}: {saved_path}")
                else:
                    log_batch_run(f"Warning: Could not find saved file for {method} alpha={alpha:.2f}", level="WARNING")
                
                # Compute additional metrics for grid display (if needed)
                if compute_metrics and metrics_computer:
                    metrics = metrics_computer.compute_all_metrics(
                        result, content_img, style1_img, style2_img, runtime
                    )
                    metrics['method'] = method
                    metrics['alpha'] = alpha
                    metrics['content_image'] = Path(content_img_path).name
                    metrics['style1_image'] = Path(style1_img_path).name
                    metrics['style2_image'] = Path(style2_img_path).name
                    all_results.append(metrics)
                
            except Exception as e:
                error_msg = f"Error in {method} alpha={alpha:.2f}: {str(e)}"
                log_batch_run(error_msg, level="ERROR")
                print(error_msg)
                continue
            
            method_results.append((alpha, result))
        
        # Create grid for this method
        grid_filename = f"{content_name}_{style1_name}_{style2_name}_{method}_grid.jpg"
        grid_path = os.path.join(output_dir, grid_filename)
        
        create_alpha_grid(
            content_np, style1_np, style2_np,
            method_results, method_name=method,
            save_path=grid_path, denormalize=False
        )
        grid_paths[method] = grid_path
    
    # Create combined metrics DataFrame (additional metrics for grid, main CSV is auto-updated)
    if all_results:
        df = pd.DataFrame(all_results)
        metrics_csv = os.path.join(output_dir, 'metrics_summary_local.csv')
        df.to_csv(metrics_csv, index=False)
        log_batch_run(f"Grid experiment complete. Local metrics saved to {metrics_csv}")
    else:
        df = pd.DataFrame()
    
    log_batch_run(f"Grid experiment finished: {len(methods)} methods × {len(alphas)} alphas")
    
    return df, grid_paths


def run_batch_experiment(
    content_dir: str,
    style_dir: str,
    content_images: List[str],
    style_pairs: List[Tuple[str, str]],
    alphas: List[float],
    methods: List[str],
    output_dir: str,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Run batch experiment across multiple content images and style pairs.
    Each run auto-saves image + JSON + appends to CSV.
    """
    """
    Run batch experiment across multiple content images and style pairs.
    
    Args:
        content_dir: Directory containing content images
        style_dir: Directory containing style images
        content_images: List of content image filenames
        style_pairs: List of (style1, style2) tuples
        alphas: List of alpha values
        methods: List of methods
        output_dir: Output directory
        config: Configuration dict
    
    Returns:
        Combined metrics DataFrame
    """
    all_metrics = []
    
    log_batch_run(f"Starting batch experiment: {len(content_images)} contents × {len(style_pairs)} pairs × {len(alphas)} alphas × {len(methods)} methods")
    
    total_runs = len(content_images) * len(style_pairs) * len(alphas) * len(methods)
    run_count = 0
    
    for content_img_name in content_images:
        content_img_path = os.path.join(content_dir, content_img_name)
        if not os.path.exists(content_img_path):
            print(f"Warning: Content image not found: {content_img_path}")
            continue
        
        for style1_name, style2_name in style_pairs:
            style1_path = os.path.join(style_dir, style1_name)
            style2_path = os.path.join(style_dir, style2_name)
            
            if not os.path.exists(style1_path) or not os.path.exists(style2_path):
                print(f"Warning: Style images not found: {style1_path} or {style2_path}")
                continue
            
            # Create subdirectory for this combination
            combo_name = f"{Path(content_img_name).stem}_{Path(style1_name).stem}_{Path(style2_name).stem}"
            combo_output_dir = os.path.join(output_dir, combo_name)
            
            log_batch_run(f"Processing: {combo_name}")
            print(f"\nProcessing: {combo_name}")
            
            try:
                df, _ = run_alpha_grid(
                    content_img_path, style1_path, style2_path,
                    alphas, methods, combo_output_dir, config, compute_metrics=True
                )
                
                run_count += len(methods) * len(alphas)
                log_batch_run(f"Completed {combo_name}: {run_count}/{total_runs} runs done")
                
                if not df.empty:
                    all_metrics.append(df)
            except Exception as e:
                error_msg = f"Error processing {combo_name}: {str(e)}"
                log_batch_run(error_msg, level="ERROR")
                print(error_msg)
                continue
    
    # Combine all metrics (main CSV is auto-updated by individual runs)
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        combined_csv = os.path.join(output_dir, 'metrics_summary_all.csv')
        combined_df.to_csv(combined_csv, index=False)
        log_batch_run(f"Batch experiment complete. Combined metrics: {combined_csv}")
        log_batch_run(f"Total runs completed: {run_count}/{total_runs}")
        return combined_df
    else:
        log_batch_run("Batch experiment finished with no metrics collected", level="WARNING")
        return pd.DataFrame()

