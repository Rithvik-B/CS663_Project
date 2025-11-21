# PCA-Mixed Neural Style Transfer

Implementation of PCA-based style mixing for neural style transfer with automatic saving of all outputs.

## Overview

This project implements neural style transfer with the ability to mix two artistic styles using Principal Component Analysis (PCA) of feature covariances. Every run automatically saves images, per-iteration metrics, and metadata to organized folders.

## Quick Start

### Command Line Interface

```bash
# Run a single style transfer
python -m src.pca_gatys \
    --content data/content_examples/tubingen.png \
    --style1 data/style_examples/vg_starry_night.jpg \
    --style2 data/style_examples/candy.jpg \
    --method pca_joint \
    --alpha 0.5 \
    --iters 100 \
    --snapshot_interval 10 \
    --seed 42
```

Outputs are automatically saved to `data/outputs/runs/`.

### Streamlit Web Interface

```bash
streamlit run app/streamlit_app.py
```

The interface provides three tabs:
- **Run**: Execute a single style transfer with real-time progress
- **Grid**: Run multiple alpha values or batch experiments
- **Comparisons**: View and compare saved runs side-by-side

## How It Works

### PCA Style Mixing

The implementation extends the original Gatys et al. algorithm to support mixing two styles:

1. **Extract Features**: VGG-19 features from content and style images
2. **Compute Covariances**: Style features are converted to covariance matrices
3. **PCA Decomposition**: Covariances are decomposed into eigenvectors and eigenvalues
4. **Mix Styles**: Two mixing strategies:
   - **Simple PCA**: Uses first style's eigenvectors, mixes eigenvalues
   - **Joint PCA**: Computes joint eigenvectors, then mixes eigenvalues
5. **Optimize**: Gradient descent to match mixed style representation

### Automatic Saving

Every run creates a folder with:
- **images/**: Snapshot images at specified intervals + final image
- **metrics/**: Per-iteration metrics CSV and final metrics JSON
- **meta.json**: Run configuration and metadata

Folder naming: `YYYYMMDD_HHMMSS__content__style1__style2__method_a{ALPHA}_s{SEED}/`

## Methods

- **pca_joint**: Joint PCA mixing (recommended)
- **pca_simple**: Simple PCA mixing
- **gram-linear**: Linear interpolation of Gram matrices
- **covariance-linear**: Linear interpolation of covariance matrices
- **gatys**: Single style (original algorithm)

## Key Features

- **Exact Iteration Control**: No overshoot beyond requested iterations
- **Automatic Saving**: All outputs saved without manual intervention
- **Organized Structure**: Deterministic folder naming for easy tracking
- **Comparisons**: Built-in tools to compare multiple runs
- **Reproducible**: Seed-based initialization for consistent results

## Installation

```bash
pip install -r requirements.txt
```

**Note**: For GPU acceleration, install CUDA-compatible PyTorch.

## Project Structure

```
final_project/
├── app/
│   └── streamlit_app.py      # Web interface
├── src/
│   ├── runner.py             # Main execution API
│   ├── pca_gatys.py          # PCA-Gatys implementation
│   ├── gatys.py              # Baseline Gatys NST
│   ├── pca_code.py           # PCA decomposition
│   ├── mixing.py             # Style mixing strategies
│   ├── vgg_features.py       # VGG feature extraction
│   ├── metrics.py            # Evaluation metrics
│   └── io_utils.py           # Image I/O utilities
├── tests/                     # Unit tests
├── data/
│   ├── content_examples/     # Content images
│   ├── style_examples/       # Style images
│   └── outputs/runs/         # Generated outputs
└── requirements.txt          # Dependencies
```

## Tests

```bash
pytest tests/
```

Tests verify:
- Exact iteration counts (no overshoot)
- All required files are saved correctly
- Integration with Streamlit UI

## Configuration

Key parameters:
- `--iters`: Number of optimization iterations (exact count)
- `--snapshot_interval`: Save image snapshot every N iterations
- `--method`: Mixing method (see Methods section)
- `--alpha`: Mixing coefficient (0.0 = style2, 1.0 = style1)
- `--seed`: Random seed for reproducibility
- `--height`: Target image height in pixels

## Output Format

Each run produces:
- **images/iter_XXX.png**: Snapshot images
- **metrics/metrics_summary.csv**: One row per iteration with losses and metrics
- **metrics/final_metrics.json**: Final quality metrics (LPIPS, SSIM, etc.)
- **meta.json**: Complete run metadata including timestamps and configuration

## Citation

If you use this code, please cite:

```bibtex
@article{gatys2015neural,
  title={A neural algorithm of artistic style},
  author={Gatys, Leon A and Ecker, Alexander S and Bethge, Matthias},
  journal={arXiv preprint arXiv:1508.06576},
  year={2015}
}
```

## License

See LICENSE file for details.
