# PCA-Mixed Neural Style Transfer

A modular implementation of PCA-based style mixing for neural style transfer, extending the original Gatys et al. algorithm to support mixing two artistic styles using Principal Component Analysis of feature covariances.

## Overview

This project implements:

- **Baseline Gatys NST**: Original neural style transfer algorithm
- **PCA Style Mixing**: Two mixing strategies (simple & joint) using PCA decomposition
- **Baseline Mixing Methods**: Gram-linear and covariance-linear interpolation for comparison
- **Comprehensive Metrics**: LPIPS, SSIM, PSNR, Gram/covariance distances, and runtime
- **Interactive UI**: Streamlit web interface for interactive style transfer
- **Batch Experiments**: Scripts for generating grids and computing metrics

## Installation

1. Clone or navigate to the repository root
2. Install dependencies:

```bash
cd final_project
pip install -r requirements.txt
```

**Note**: For GPU acceleration, ensure you have CUDA-compatible PyTorch installed. The code will automatically use CPU if CUDA is unavailable.

## Quick Start

### Single Style Transfer (CLI)

```bash
# Run a single PCA mixed transfer
python -m src.pca_gatys \
    --content data/content_examples/taj_mahal.jpg \
    --style1 data/style_examples/vg_starry_night.jpg \
    --style2 data/style_examples/candy.jpg \
    --alpha 0.5 \
    --iters 400 \
    --out results/taj_mix_0.5.jpg
```

### Using Scripts

```bash
# Run single transfer
bash scripts/run_single.sh \
    --content ../data/content-images/taj_mahal.jpg \
    --style1 ../data/style-images/vg_starry_night.jpg \
    --style2 ../data/style-images/candy.jpg \
    --alpha 0.5 \
    --out results/output.jpg

# Run grid across alpha values
bash scripts/run_grid.sh \
    --content ../data/content-images/taj_mahal.jpg \
    --style1 ../data/style-images/vg_starry_night.jpg \
    --style2 ../data/style-images/candy.jpg \
    --alphas "0.0,0.25,0.5,0.75,1.0" \
    --output results/grids

# Evaluate metrics across output directory
bash scripts/evaluate.sh \
    --gen_dir data/outputs \
    --content_dir data/content_examples \
    --out_csv results/metrics_summary.csv
```

### Streamlit UI

Launch the interactive web interface:

```bash
streamlit run app/streamlit_app.py
```

The UI provides:
- Image upload/selection from examples
- Method selection (PCA joint/simple, Gram-linear, Cov-linear, Gatys)
- Real-time parameter adjustment
- Side-by-side comparisons
- Metrics display
- Batch experiment generation

## Project Structure

```
final_project/
├── data/                    # Data directories
│   ├── content_examples/    # Content images
│   ├── style_examples/      # Style images
│   └── outputs/             # Generated outputs
├── src/                     # Source code
│   ├── config.py           # Configuration and hyperparameters
│   ├── io_utils.py         # Image I/O and visualization
│   ├── vgg_features.py     # VGG feature extractor
│   ├── pca_code.py         # PCA code extraction
│   ├── mixing.py           # Mixing strategies
│   ├── gatys.py            # Baseline Gatys NST
│   ├── pca_gatys.py        # PCA-Gatys style transfer
│   ├── metrics.py           # Evaluation metrics
│   └── experiments.py      # Batch experiment orchestrator
├── app/
│   └── streamlit_app.py    # Streamlit UI
├── tests/                   # Unit tests
├── scripts/                 # CLI scripts
├── results/                 # Sample results and metrics
└── requirements.txt         # Dependencies
```

## Methods

### PCA Mixing Strategies

1. **Simple PCA Mix**: Uses P₁ basis, mixes eigenvalues
   - `D_mix = α·D₁ + (1-α)·diag(P₁ᵀ C₂ P₁)`

2. **Joint PCA Mix**: Computes eigenvectors of (C₁+C₂)/2, then mixes
   - `P_mix = eigenvectors((C₁ + C₂) / 2)`
   - `D_mix = α·diag(P_mixᵀ C₁ P_mix) + (1-α)·diag(P_mixᵀ C₂ P_mix)`

### Baseline Methods

- **Gram-linear**: Linear interpolation of Gram matrices
- **Covariance-linear**: Linear interpolation of covariance matrices
- **Gatys (single style)**: Original algorithm with one style

## Configuration

Default hyperparameters are in `src/config.py`. Key parameters:

- `content_weight`: Weight for content loss (default: 1e5)
- `style_weight`: Weight for style loss (default: 3e4)
- `tv_weight`: Weight for total variation loss (default: 1e0)
- `iterations`: Number of optimization iterations (default: 1000 for LBFGS, 3000 for Adam)
- `height`: Target image height (default: 400)
- `optimizer`: 'lbfgs' or 'adam' (default: 'lbfgs')

## Evaluation Metrics

The project computes:

- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)
- **SSIM**: Structural Similarity Index (higher is better, [0, 1])
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better, dB)
- **Gram Distance**: Frobenius norm between Gram matrices (per layer)
- **Covariance Distance**: Frobenius norm between covariance matrices (per layer)
- **Runtime**: Elapsed time per transfer

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_pca_code.py -v
```

## Expected Runtime

- **Single transfer (LBFGS, 1000 iter)**: ~30-60 seconds on GPU, ~2-5 minutes on CPU
- **Single transfer (Adam, 3000 iter)**: ~1-2 minutes on GPU, ~5-10 minutes on CPU
- **Grid (5 alphas × 4 methods)**: ~10-20 minutes on GPU

**Recommendation**: Use GPU (CUDA) for faster results. The code automatically falls back to CPU if GPU is unavailable.

## Data Setup

The project expects images in:
- `data/content_examples/`: Content images
- `data/style_examples/`: Style images

You can create symbolic links to the parent repository's data:

```bash
# Linux/Mac
ln -s ../../data/content-images data/content_examples
ln -s ../../data/style-images data/style_examples

# Windows (PowerShell)
New-Item -ItemType SymbolicLink -Path data/content_examples -Target ..\..\data\content-images
New-Item -ItemType SymbolicLink -Path data/style_examples -Target ..\..\data\style-images
```

## Citation

If you use this code, please cite the original Gatys et al. paper:

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

## Examples

Example outputs and metrics are available in `results/`:
- Grid visualizations showing alpha interpolation
- Metrics CSV files with quantitative comparisons
- Sample stylized images

## Troubleshooting

**Import errors**: Ensure you're running from the `final_project/` directory or have added it to PYTHONPATH.

**CUDA out of memory**: Reduce image height or batch size in config.

**LPIPS/metrics not available**: Install missing dependencies: `pip install lpips scikit-image`

**Slow performance**: Use GPU if available, or reduce iterations/image size for testing.

