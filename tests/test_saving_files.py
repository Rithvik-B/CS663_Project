"""
Test that all required files are saved correctly.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.runner import run_once


def test_saving_files():
    """Test that all required files are saved."""
    # Create tiny test images
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    from PIL import Image
    
    content_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    style_img = Image.new('RGB', (64, 64), color=(255, 0, 0))
    
    content_path = test_dir / "test_content_save.png"
    style_path = test_dir / "test_style_save.png"
    
    content_img.save(content_path)
    style_img.save(style_path)
    
    # Run with snapshot_interval=1 to ensure all snapshots are saved
    max_iters = 3
    snapshot_interval = 1
    
    config = {
        'content_img_path': str(content_path),
        'style1_img_path': str(style_path),
        'style2_img_path': str(style_path),
        'mixing_method': 'gatys',
        'alpha': 0.5,
        'iterations': max_iters,
        'snapshot_interval': snapshot_interval,
        'height': 64,
        'optimizer': 'adam',
        'seed': 42,
        'device': 'cpu'
    }
    
    run_folder = run_once(config)
    
    # Check directory structure
    assert run_folder.exists(), "Run folder should exist"
    assert (run_folder / "images").exists(), "images/ directory should exist"
    assert (run_folder / "metrics").exists(), "metrics/ directory should exist"
    
    # Check images
    # Should have iter_001.png, iter_002.png, iter_003.png (final)
    for i in range(1, max_iters + 1):
        if i % snapshot_interval == 0 or i == max_iters:
            img_path = run_folder / "images" / f"iter_{i:03d}.png"
            assert img_path.exists(), f"Image iter_{i:03d}.png should exist"
    
    # Check final.png (saved as iter_{max_iters:03d}.png)
    final_path = run_folder / "images" / f"iter_{max_iters:03d}.png"
    assert final_path.exists(), "Final image should exist"
    
    # Check metrics CSV
    metrics_csv = run_folder / "metrics" / "metrics_summary.csv"
    assert metrics_csv.exists(), "metrics_summary.csv should exist"
    
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    assert len(df) == max_iters, f"Metrics CSV should have {max_iters} rows"
    assert 'iteration' in df.columns, "Metrics CSV should have 'iteration' column"
    assert 'total_loss' in df.columns, "Metrics CSV should have 'total_loss' column"
    
    # Check final_metrics.json
    final_metrics = run_folder / "metrics" / "final_metrics.json"
    assert final_metrics.exists(), "final_metrics.json should exist"
    
    with open(final_metrics, 'r') as f:
        final_data = json.load(f)
    
    assert 'total_loss' in final_data, "final_metrics.json should have 'total_loss'"
    assert 'iteration' in final_data, "final_metrics.json should have 'iteration'"
    
    # Check meta.json
    meta_path = run_folder / "meta.json"
    assert meta_path.exists(), "meta.json should exist"
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    assert 'start_ts_utc' in meta, "meta.json should have 'start_ts_utc'"
    assert 'end_ts_utc' in meta, "meta.json should have 'end_ts_utc'"
    assert meta['end_ts_utc'] is not None, "end_ts_utc should be set after completion"
    assert meta['iterations_completed'] == max_iters, "iterations_completed should match"
    
    print("✓ File saving test passed")


if __name__ == "__main__":
    print("Running file saving tests...")
    test_saving_files()
    print("\nAll file saving tests passed! ✓")

