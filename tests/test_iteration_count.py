"""
Test exact iteration counts - no overshoot.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.runner import run_once
from src.io_utils import prepare_img
import torch


def test_exact_iteration_count():
    """Test that exactly the requested number of iterations are executed."""
    # Create tiny test images
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal test images (1x1 pixels)
    import numpy as np
    from PIL import Image
    
    content_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    style_img = Image.new('RGB', (64, 64), color=(255, 0, 0))
    
    content_path = test_dir / "test_content.png"
    style_path = test_dir / "test_style.png"
    
    content_img.save(content_path)
    style_img.save(style_path)
    
    # Run with exactly 2 iterations
    max_iters = 2
    config = {
        'content_img_path': str(content_path),
        'style1_img_path': str(style_path),
        'style2_img_path': str(style_path),
        'mixing_method': 'gatys',
        'alpha': 0.5,
        'iterations': max_iters,
        'snapshot_interval': 1,
        'height': 64,
        'optimizer': 'adam',  # Use Adam for deterministic iteration count
        'seed': 42,
        'device': 'cpu'  # Use CPU for consistency
    }
    
    run_folder = run_once(config)
    
    # Check metrics CSV has exactly max_iters rows
    metrics_path = run_folder / "metrics" / "metrics_summary.csv"
    assert metrics_path.exists(), "Metrics CSV should exist"
    
    df = pd.read_csv(metrics_path)
    assert len(df) == max_iters, f"Expected {max_iters} rows, got {len(df)}"
    
    # Check iterations are 1, 2, ..., max_iters
    assert list(df['iteration']) == list(range(1, max_iters + 1)), "Iterations should be 1, 2, ..., max_iters"
    
    # Check meta.json
    meta_path = run_folder / "meta.json"
    assert meta_path.exists(), "meta.json should exist"
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    assert meta['iterations_completed'] == max_iters, f"Expected {max_iters} iterations completed, got {meta['iterations_completed']}"
    assert meta['iterations_requested'] == max_iters, f"Expected {max_iters} iterations requested, got {meta['iterations_requested']}"
    
    print("✓ Iteration count test passed")


def test_lbfgs_exact_iterations():
    """Test that LBFGS doesn't overshoot iterations."""
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    from PIL import Image
    
    content_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    style_img = Image.new('RGB', (64, 64), color=(255, 0, 0))
    
    content_path = test_dir / "test_content_lbfgs.png"
    style_path = test_dir / "test_style_lbfgs.png"
    
    content_img.save(content_path)
    style_img.save(style_path)
    
    max_iters = 3
    config = {
        'content_img_path': str(content_path),
        'style1_img_path': str(style_path),
        'style2_img_path': str(style_path),
        'mixing_method': 'gatys',
        'alpha': 0.5,
        'iterations': max_iters,
        'snapshot_interval': 1,
        'height': 64,
        'optimizer': 'lbfgs',  # Test LBFGS specifically
        'seed': 42,
        'device': 'cpu'
    }
    
    run_folder = run_once(config)
    
    # Check exact iteration count
    metrics_path = run_folder / "metrics" / "metrics_summary.csv"
    df = pd.read_csv(metrics_path)
    
    assert len(df) == max_iters, f"LBFGS: Expected {max_iters} rows, got {len(df)}"
    assert list(df['iteration']) == list(range(1, max_iters + 1)), "LBFGS: Iterations should be sequential"
    
    meta_path = run_folder / "meta.json"
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    assert meta['iterations_completed'] == max_iters, f"LBFGS: Expected {max_iters} iterations completed"
    
    print("✓ LBFGS iteration count test passed")


if __name__ == "__main__":
    print("Running iteration count tests...")
    test_exact_iteration_count()
    test_lbfgs_exact_iterations()
    print("\nAll iteration count tests passed! ✓")

