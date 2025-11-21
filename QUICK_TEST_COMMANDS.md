# Quick Test Commands - Low Iterations

Use these commands to quickly test the implementation with minimal iterations.

## Prerequisites

Make sure you have example images in:
- `data/content_examples/` (e.g., `taj_mahal.jpg`)
- `data/style_examples/` (e.g., `vg_starry_night.jpg`, `candy.jpg`)

## Option 1: Run Test Script (Easiest)

```bash
# From project root
python test_quick.py
```

This will:
- Run baseline Gatys (10 iterations)
- Run PCA mix (10 iterations)
- Verify CSV logging
- Show summary of all tests

## Option 2: Manual CLI Commands

### Test Baseline Gatys (10 iterations)

```bash
python -m src.gatys \
    --content data/content_examples/taj_mahal.jpg \
    --style data/style_examples/vg_starry_night.jpg \
    --iters 10 \
    --height 256
```

**What to check:**
- `data/outputs/` should have a new `.jpg` file
- `data/outputs/` should have a corresponding `.json` file
- `results/metrics_summary.csv` should have a new row

### Test PCA Mix (10 iterations)

```bash
python -m src.pca_gatys \
    --content data/content_examples/taj_mahal.jpg \
    --style1 data/style_examples/vg_starry_night.jpg \
    --style2 data/style_examples/candy.jpg \
    --alpha 0.5 \
    --method joint \
    --iters 10 \
    --height 256
```

**What to check:**
- `data/outputs/` should have a new `.jpg` file with "pca_mix" and "joint" in filename
- `data/outputs/` should have a corresponding `.json` file
- `results/metrics_summary.csv` should have a new row

### Test Multiple Methods (Quick)

```bash
# Gram-linear (10 iter)
python -m src.pca_gatys \
    --content data/content_examples/taj_mahal.jpg \
    --style1 data/style_examples/vg_starry_night.jpg \
    --style2 data/style_examples/candy.jpg \
    --alpha 0.5 \
    --method gram-linear \
    --iters 10 \
    --height 256

# Covariance-linear (10 iter)
python -m src.pca_gatys \
    --content data/content_examples/taj_mahal.jpg \
    --style1 data/style_examples/vg_starry_night.jpg \
    --style2 data/style_examples/candy.jpg \
    --alpha 0.5 \
    --method covariance-linear \
    --iters 10 \
    --height 256
```

## Option 3: Python API (Quick Test)

```python
from src.gatys import gatys_style_transfer
from src.pca_gatys import pca_gatys_style_transfer
from src.config import DEFAULT_CONFIG

# Quick config
config = DEFAULT_CONFIG.copy()
config.update({
    'iterations': 10,  # Very low!
    'height': 256,     # Smaller for speed
    'optimizer': 'lbfgs'
})

# Test baseline
result, metrics = gatys_style_transfer(
    'data/content_examples/taj_mahal.jpg',
    'data/style_examples/vg_starry_night.jpg',
    output_path=None,  # Auto-save
    config=config
)

# Test PCA mix
result, metrics = pca_gatys_style_transfer(
    'data/content_examples/taj_mahal.jpg',
    'data/style_examples/vg_starry_night.jpg',
    'data/style_examples/candy.jpg',
    alpha=0.5,
    mixing_method='joint',
    output_path=None,  # Auto-save
    config=config
)
```

## Verification Checklist

After running tests, verify:

### ✅ Files Created

1. **Image files** in `data/outputs/`:
   ```bash
   ls -lt data/outputs/*.jpg | head -5
   ```

2. **JSON metadata** files:
   ```bash
   ls -lt data/outputs/*.json | head -5
   ```

3. **CSV file** updated:
   ```bash
   tail -5 results/metrics_summary.csv
   ```

### ✅ Filename Format

Check that filenames follow the pattern:
```
YYYYMMDD_HHMMSS__mode_METHOD__contentNAME__styleANAME__styleBNAME__alpha_X.XX.jpg
```

Example:
```
20241215_143022__mode_baseline__METHOD_gatys__contenttaj_mahal__styleAvg_starry_night.jpg
```

### ✅ JSON Metadata

Check a JSON file:
```bash
cat data/outputs/YYYYMMDD_HHMMSS__*.json | python -m json.tool | head -30
```

Should contain:
- `mode`: "baseline" or "pca_mix"
- `method`: "gatys", "joint", "simple", etc.
- `hyperparameters`: All config values
- `final_losses`: Total, content, style, TV losses
- `runtime_seconds`: Execution time
- Metrics (LPIPS, SSIM, PSNR, etc.)

### ✅ CSV Content

Check CSV:
```bash
python -c "import pandas as pd; df = pd.read_csv('results/metrics_summary.csv'); print(df.tail(3).to_string())"
```

Should have columns like:
- `mode`, `method`, `alpha`
- `content_image`, `style1_image`, `style2_image`
- `hyperparameters_*` (flattened)
- `final_losses_*`
- `runtime_seconds`
- `lpips_content`, `ssim_content`, `psnr_content`
- etc.

## Expected Runtime

With 10 iterations and height=256:
- **Baseline Gatys**: ~5-15 seconds (CPU) or ~2-5 seconds (GPU)
- **PCA Mix**: ~8-20 seconds (CPU) or ~3-8 seconds (GPU)

## Troubleshooting

### No files created?

1. Check that directories exist:
   ```bash
   mkdir -p data/outputs
   mkdir -p results
   ```

2. Check permissions:
   ```bash
   ls -ld data/outputs results
   ```

### CSV not updating?

1. Check if file exists:
   ```bash
   ls -l results/metrics_summary.csv
   ```

2. Check for errors in output (should see "Warning: Could not append to CSV" if there's an issue)

### JSON missing metrics?

This is OK - metrics computation may fail if LPIPS/scikit-image not installed. The JSON will still have:
- Hyperparameters
- Final losses
- Runtime

## Next Steps

Once quick tests pass:

1. **Increase iterations** for better quality:
   ```bash
   --iters 100  # Still fast, better results
   ```

2. **Test Streamlit UI**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   - Set iterations to 10-20 in UI
   - Run a transfer
   - Verify auto-save and metadata display

3. **Test batch experiments**:
   ```python
   from src.experiments import run_alpha_grid
   # Use low iterations in config
   ```

## Performance Notes

- **First run**: Slower (computes PCA codes)
- **Subsequent runs**: Faster (uses cached PCA codes)
- **Cache location**: `data/pca_cache/`

To clear cache:
```bash
rm -rf data/pca_cache/*
```

