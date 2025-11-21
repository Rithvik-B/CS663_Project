"""
Utilities for auto-saving images with metadata and deterministic filenames.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import torch
import pandas as pd

from .io_utils import save_image, tensor_to_image
from .config import get_project_root, get_data_path


def generate_output_filename(
    mode: str,
    method: str,
    content_name: str,
    style1_name: Optional[str] = None,
    style2_name: Optional[str] = None,
    alpha: Optional[float] = None,
    timestamp: Optional[str] = None
) -> str:
    """
    Generate deterministic filename following pattern:
    YYYYMMDD_HHMMSS__mode_METHOD__contentNAME__styleANAME__styleBNAME__alpha_X.XX.jpg
    
    Args:
        mode: 'baseline' or 'pca_mix'
        method: Method name (e.g., 'joint', 'simple', 'gram-linear', 'gatys')
        content_name: Content image name (stem, no extension)
        style1_name: Style 1 image name (stem)
        style2_name: Style 2 image name (stem)
        alpha: Mixing coefficient
        timestamp: Optional timestamp string (YYYYMMDD_HHMMSS), if None uses current time
    
    Returns:
        Filename string
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean names (remove special chars, spaces -> underscores)
    def clean_name(name: str) -> str:
        return name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    
    content_clean = clean_name(content_name)
    
    parts = [timestamp, f"mode_{mode}", f"METHOD_{method}", f"content{content_clean}"]
    
    if style1_name:
        style1_clean = clean_name(style1_name)
        parts.append(f"styleA{style1_clean}")
    
    if style2_name:
        style2_clean = clean_name(style2_name)
        parts.append(f"styleB{style2_clean}")
    
    if alpha is not None:
        parts.append(f"alpha_{alpha:.2f}")
    
    filename = "__".join(parts) + ".jpg"
    return filename


def save_result_with_metadata(
    result_img: torch.Tensor,
    metadata: Dict,
    output_dir: Optional[str] = None,
    custom_filename: Optional[str] = None
) -> Tuple[str, str]:
    """
    Save result image and JSON metadata with deterministic filename.
    
    Args:
        result_img: Result image tensor (1, 3, H, W)
        metadata: Dictionary containing all metadata (hyperparameters, metrics, etc.)
        output_dir: Output directory (default: final_project/data/outputs/)
        custom_filename: Optional custom filename (if None, generates from metadata)
    
    Returns:
        Tuple of (image_path, json_path)
    """
    if output_dir is None:
        output_dir = get_data_path("outputs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if custom_filename is None:
        mode = metadata.get('mode', 'pca_mix')
        method = metadata.get('method', 'unknown')
        content_name = Path(metadata.get('content_image', 'unknown')).stem
        style1_name = Path(metadata.get('style1_image', '')).stem if metadata.get('style1_image') else None
        style2_name = Path(metadata.get('style2_image', '')).stem if metadata.get('style2_image') else None
        alpha = metadata.get('alpha')
        
        filename = generate_output_filename(
            mode=mode,
            method=method,
            content_name=content_name,
            style1_name=style1_name,
            style2_name=style2_name,
            alpha=alpha
        )
    else:
        filename = custom_filename
        if not filename.endswith('.jpg'):
            filename += '.jpg'
    
    # Save image
    image_path = os.path.join(output_dir, filename)
    save_image(result_img, image_path, denormalize=True)
    
    # Save JSON metadata
    json_path = os.path.join(output_dir, Path(filename).stem + '.json')
    
    # Ensure all values are JSON-serializable
    metadata_serializable = _make_json_serializable(metadata)
    
    with open(json_path, 'w') as f:
        json.dump(metadata_serializable, f, indent=2)
    
    return image_path, json_path


def _make_json_serializable(obj):
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return str(obj)


def append_to_metrics_csv(
    metadata: Dict,
    csv_path: Optional[str] = None
) -> None:
    """
    Append a single row to metrics_summary.csv.
    
    Args:
        metadata: Dictionary containing metrics and run info
        csv_path: Path to CSV file (default: final_project/results/metrics_summary.csv)
    """
    if csv_path is None:
        csv_path = os.path.join(get_project_root(), 'results', 'metrics_summary.csv')
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Flatten nested dictionaries for CSV
    flat_metadata = _flatten_dict(metadata)
    
    # Create DataFrame with single row
    df_new = pd.DataFrame([flat_metadata])
    
    # Append to existing CSV if it exists
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # Save
    df_combined.to_csv(csv_path, index=False)


def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary for CSV."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], dict):
            # Handle list of dicts (e.g., per-layer metrics)
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(_flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_batch_run(
    message: str,
    log_path: Optional[str] = None,
    level: str = "INFO"
) -> None:
    """
    Append a log entry to batch_run_log.txt.
    
    Args:
        message: Log message
        log_path: Path to log file (default: final_project/results/batch_run_log.txt)
        level: Log level (INFO, ERROR, WARNING)
    """
    if log_path is None:
        log_path = os.path.join(get_project_root(), 'results', 'batch_run_log.txt')
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}\n"
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)

