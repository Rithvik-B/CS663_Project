"""
Quick test script with low iterations to verify auto-save, JSON metadata, and CSV logging.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.pca_gatys import pca_gatys_style_transfer
from src.gatys import gatys_style_transfer
from src.config import DEFAULT_CONFIG, get_project_root, get_data_path
import json
import pandas as pd


def test_baseline_gatys():
    """Test baseline Gatys with low iterations."""
    print("\n" + "="*60)
    print("TEST 1: Baseline Gatys (10 iterations)")
    print("="*60)
    
    # Use example images
    content_path = os.path.join(get_data_path("content_examples"), "taj_mahal.jpg")
    style_path = os.path.join(get_data_path("style_examples"), "vg_starry_night.jpg")
    
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print(f"⚠️  Skipping: Images not found")
        print(f"   Content: {content_path}")
        print(f"   Style: {style_path}")
        return False
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        'iterations': 10,  # Very low for testing
        'optimizer': 'lbfgs',
        'height': 256  # Smaller for speed
    })
    
    print(f"Running Gatys style transfer...")
    result, metrics = gatys_style_transfer(
        content_path, style_path,
        output_path=None,  # Auto-save will handle it
        config=config
    )
    
    # Check outputs
    outputs_dir = get_data_path("outputs")
    jpg_files = list(Path(outputs_dir).glob("*.jpg"))
    json_files = list(Path(outputs_dir).glob("*.json"))
    
    if jpg_files and json_files:
        latest_jpg = max(jpg_files, key=os.path.getmtime)
        latest_json = max(json_files, key=os.path.getmtime)
        
        print(f"✅ Image saved: {latest_jpg.name}")
        print(f"✅ JSON saved: {latest_json.name}")
        
        # Verify JSON
        with open(latest_json, 'r') as f:
            metadata = json.load(f)
        print(f"✅ JSON contains: mode={metadata.get('mode')}, method={metadata.get('method')}")
        print(f"   Final loss: {metadata.get('final_losses', {}).get('total_loss', 'N/A')}")
        
        return True
    else:
        print("❌ Files not found!")
        return False


def test_pca_mix():
    """Test PCA mixing with low iterations."""
    print("\n" + "="*60)
    print("TEST 2: PCA Mix (joint, 10 iterations)")
    print("="*60)
    
    content_path = os.path.join(get_data_path("content_examples"), "taj_mahal.jpg")
    style1_path = os.path.join(get_data_path("style_examples"), "vg_starry_night.jpg")
    style2_path = os.path.join(get_data_path("style_examples"), "candy.jpg")
    
    if not all(os.path.exists(p) for p in [content_path, style1_path, style2_path]):
        print(f"⚠️  Skipping: Images not found")
        return False
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        'iterations': 10,
        'optimizer': 'lbfgs',
        'height': 256
    })
    
    print(f"Running PCA-Gatys style transfer (alpha=0.5)...")
    result, metrics = pca_gatys_style_transfer(
        content_path, style1_path, style2_path,
        alpha=0.5,
        mixing_method='joint',
        output_path=None,
        config=config
    )
    
    # Check outputs
    outputs_dir = get_data_path("outputs")
    jpg_files = list(Path(outputs_dir).glob("*.jpg"))
    json_files = list(Path(outputs_dir).glob("*.json"))
    
    if jpg_files and json_files:
        latest_jpg = max(jpg_files, key=os.path.getmtime)
        latest_json = max(json_files, key=os.path.getmtime)
        
        print(f"✅ Image saved: {latest_jpg.name}")
        print(f"✅ JSON saved: {latest_json.name}")
        
        # Verify JSON
        with open(latest_json, 'r') as f:
            metadata = json.load(f)
        print(f"✅ JSON contains: mode={metadata.get('mode')}, method={metadata.get('method')}")
        print(f"   Alpha: {metadata.get('alpha')}")
        print(f"   Final loss: {metadata.get('final_losses', {}).get('total_loss', 'N/A')}")
        
        return True
    else:
        print("❌ Files not found!")
        return False


def test_csv_logging():
    """Test CSV logging."""
    print("\n" + "="*60)
    print("TEST 3: CSV Logging Verification")
    print("="*60)
    
    csv_path = os.path.join(get_project_root(), 'results', 'metrics_summary.csv')
    
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV file not found: {csv_path}")
        print("   This is OK if no runs have completed yet")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"✅ CSV file exists: {csv_path}")
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
    
    if len(df) > 0:
        print(f"\n   Latest entry:")
        latest = df.iloc[-1]
        print(f"   - Mode: {latest.get('mode', 'N/A')}")
        print(f"   - Method: {latest.get('method', 'N/A')}")
        print(f"   - Runtime: {latest.get('runtime_seconds', 'N/A')}s")
        return True
    
    return False


def main():
    """Run all quick tests."""
    print("\n" + "="*60)
    print("QUICK TEST SUITE - Low Iterations (10 iter)")
    print("="*60)
    print("\nThis will test:")
    print("  1. Baseline Gatys auto-save")
    print("  2. PCA Mix auto-save")
    print("  3. CSV logging")
    print("\nNote: Using only 10 iterations for speed!")
    print("="*60)
    
    results = []
    
    # Test 1: Baseline
    try:
        results.append(("Baseline Gatys", test_baseline_gatys()))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("Baseline Gatys", False))
    
    # Test 2: PCA Mix
    try:
        results.append(("PCA Mix", test_pca_mix()))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("PCA Mix", False))
    
    # Test 3: CSV
    try:
        results.append(("CSV Logging", test_csv_logging()))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("CSV Logging", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed or were skipped")
    print("="*60)
    
    # Show output locations
    print("\n📁 Check these locations:")
    print(f"   Images: {get_data_path('outputs')}")
    print(f"   CSV: {os.path.join(get_project_root(), 'results', 'metrics_summary.csv')}")
    print(f"   Logs: {os.path.join(get_project_root(), 'results', 'batch_run_log.txt')}")


if __name__ == "__main__":
    main()

