"""
Smoke test for Streamlit run functionality via runner API.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.runner import run_once


def test_streamlit_run_smoke():
    """Test that run_once works with config similar to what Streamlit UI would provide."""
    # Create tiny test images
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    from PIL import Image
    
    content_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    style1_img = Image.new('RGB', (64, 64), color=(255, 0, 0))
    style2_img = Image.new('RGB', (64, 64), color=(0, 255, 0))
    
    content_path = test_dir / "test_content_ui.png"
    style1_path = test_dir / "test_style1_ui.png"
    style2_path = test_dir / "test_style2_ui.png"
    
    content_img.save(content_path)
    style1_img.save(style1_path)
    style2_img.save(style2_path)
    
    # Config similar to what Streamlit UI would create
    config = {
        'content_img_path': str(content_path),
        'style1_img_path': str(style1_path),
        'style2_img_path': str(style2_path),
        'mixing_method': 'pca_joint',  # UI method name
        'alpha': 0.5,
        'iterations': 5,
        'snapshot_interval': 2,
        'height': 64,
        'optimizer': 'adam',
        'content_weight': 1e5,
        'style_weight': 3e4,
        'tv_weight': 1e0,
        'seed': 42,
        'device': 'cpu'
    }
    
    # Run via runner API (what UI would call)
    run_folder = run_once(config)
    
    # Verify run folder was created and returned
    assert run_folder.exists(), "Run folder should exist"
    assert isinstance(run_folder, Path), "run_once should return Path object"
    
    # Verify basic structure
    assert (run_folder / "images").exists(), "images/ directory should exist"
    assert (run_folder / "metrics").exists(), "metrics/ directory should exist"
    assert (run_folder / "meta.json").exists(), "meta.json should exist"
    
    # Verify some images were saved
    images_dir = run_folder / "images"
    image_files = list(images_dir.glob("*.png"))
    assert len(image_files) > 0, "At least one image should be saved"
    
    # Verify metrics were saved
    metrics_csv = run_folder / "metrics" / "metrics_summary.csv"
    assert metrics_csv.exists(), "Metrics CSV should exist"
    
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    assert len(df) > 0, "Metrics CSV should have rows"
    
    print("✓ Streamlit run smoke test passed")


if __name__ == "__main__":
    print("Running Streamlit run smoke tests...")
    test_streamlit_run_smoke()
    print("\nAll Streamlit run smoke tests passed! ✓")

