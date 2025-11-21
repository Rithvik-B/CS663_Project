# PCA-Mixed Neural Style Transfer - Codebase Analysis

## Overview

This is a comprehensive implementation of PCA-Mixed Neural Style Transfer, extending the original Gatys et al. (2015) algorithm to support mixing two artistic styles using Principal Component Analysis of feature covariances.

## Architecture Overview

The codebase is well-structured and modular, following best practices:

```
CS663_Project/
├── src/                    # Core implementation modules
│   ├── config.py          # Configuration and hyperparameters
│   ├── vgg_features.py     # VGG-19 feature extractor wrapper
│   ├── pca_code.py         # PCA code extraction (covariance + eigendecomposition)
│   ├── mixing.py           # Style mixing strategies
│   ├── gatys.py            # Baseline Gatys NST implementation
│   ├── pca_gatys.py        # PCA-Gatys style transfer (main algorithm)
│   ├── metrics.py           # Evaluation metrics (LPIPS, SSIM, PSNR, distances)
│   ├── experiments.py       # Batch experiment orchestrator
│   ├── io_utils.py         # Image I/O and visualization
│   └── utils.py            # Utility functions
├── app/
│   └── streamlit_app.py    # Interactive web UI
├── tests/                  # Unit tests
├── scripts/                # CLI scripts for batch processing
├── data/                   # Data directories
└── results/                # Output results and metrics
```

## Core Components

### 1. VGG Feature Extraction (`src/vgg_features.py`)

**Purpose**: Wrapper for VGG-19 feature extraction that provides consistent interface.

**Key Features**:
- Supports both existing repo models and torchvision fallback
- Extracts features at specified layers:
  - **Content layer**: `conv4_2`
  - **Style layers**: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`
- Handles gradient computation for optimization

**Implementation Details**:
- Uses VGG-19 pretrained on ImageNet
- Normalizes with ImageNet mean/std: `[123.675, 116.28, 103.53]`
- Returns features as dictionaries keyed by layer names

### 2. PCA Code Extraction (`src/pca_code.py`)

**Purpose**: Compute PCA decomposition of feature covariances for each style layer.

**Key Components**:

#### `PCACode` Class
Container for:
- `P`: Eigenvectors matrix (C × C)
- `D`: Eigenvalues (diagonal, C × 1)
- `C`: Covariance matrix (C × C)
- `mean`: Feature mean (C × 1)
- `layer_name`: Layer identifier

#### `compute_covariance()`
- Flattens spatial dimensions: (B, C, H, W) → (C, M) where M = H×W
- Centers features per channel
- Computes: `C = (1/M) * F_centered @ F_centered^T`

#### `compute_pca_code()`
- Performs eigendecomposition: `C = P @ diag(D) @ P^T`
- Uses `torch.linalg.eigh()` for symmetric matrices
- Returns eigenvalues in descending order

**Mathematical Foundation**:
- For feature map F ∈ R^(C×H×W), flattened to F_flat ∈ R^(C×M)
- Covariance: C = (1/M) * (F_centered @ F_centered^T)
- Eigendecomposition: C = P D P^T where P are eigenvectors, D are eigenvalues

### 3. Style Mixing Strategies (`src/mixing.py`)

**Purpose**: Implement different strategies for mixing two style representations.

#### A. Simple PCA Mix
- Uses P₁ (style1's eigenvectors) as basis
- Projects C₂ onto P₁: `D₂_proj = diag(P₁^T @ C₂ @ P₁)`
- Mixes eigenvalues: `D_mix = α·D₁ + (1-α)·D₂_proj`
- Reconstructs: `C_mix = P₁ @ diag(D_mix) @ P₁^T`

#### B. Joint PCA Mix (Recommended)
- Computes joint basis: `P_mix = eigenvectors((C₁ + C₂) / 2)`
- Projects both onto joint basis:
  - `D₁_proj = diag(P_mix^T @ C₁ @ P_mix)`
  - `D₂_proj = diag(P_mix^T @ C₂ @ P_mix)`
- Mixes: `D_mix = α·D₁_proj + (1-α)·D₂_proj`
- Reconstructs: `C_mix = P_mix @ diag(D_mix) @ P_mix^T`

#### C. Covariance-Linear Mix (Baseline)
- Direct interpolation: `C_mix = α·C₁ + (1-α)·C₂`
- Then recomputes eigendecomposition

#### D. Gram-Linear Mix (Baseline)
- Linear interpolation of Gram matrices: `G_mix = α·G₁ + (1-α)·G₂`
- Uses original Gatys loss with mixed Gram targets

**Key Function**: `mix_style_codes()`
- Handles per-layer mixing
- Supports per-layer alpha values (optional)
- Returns dictionary of mixed PCA codes

### 4. Baseline Gatys NST (`src/gatys.py`)

**Purpose**: Original Gatys et al. (2015) implementation.

**Loss Function**:
```
L_total = λ_content·L_content + λ_style·L_style + λ_tv·L_tv
```

Where:
- **Content Loss**: MSE on `conv4_2` activations
- **Style Loss**: MSE on Gram matrices across style layers
- **TV Loss**: Total variation regularization

**Optimization**:
- Supports both LBFGS and Adam optimizers
- LBFGS: 1000 iterations (default)
- Adam: 3000 iterations (default)
- Learning rate: 1e1 for Adam

**Initialization Options**:
- `content`: Start from content image (default)
- `random`: Gaussian noise
- `style`: Start from style image

### 5. PCA-Gatys Style Transfer (`src/pca_gatys.py`)

**Purpose**: Main algorithm combining PCA mixing with Gatys optimization.

**Workflow**:
1. Extract PCA codes from both style images
2. Mix codes using selected strategy (simple/joint/cov-linear/gram-linear)
3. Extract target content features
4. Optimize image to match:
   - Content features (MSE loss)
   - Mixed covariance matrices (MSE loss)
   - Total variation (regularization)

**Key Function**: `build_pca_loss()`
- Computes covariance loss: `||C_current - C_target||²`
- Computes mean loss: `||μ_current - μ_target||²`
- Averages across style layers

**Special Handling**:
- `gram-linear` method uses Gram matrix targets (falls back to Gatys loss)
- Other methods use covariance targets

### 6. Evaluation Metrics (`src/metrics.py`)

**Purpose**: Comprehensive metric computation for evaluation.

**Metrics Implemented**:

1. **LPIPS** (Learned Perceptual Image Patch Similarity)
   - Uses AlexNet-based LPIPS model
   - Lower is better (measures perceptual distance)
   - Requires `lpips` package

2. **SSIM** (Structural Similarity Index)
   - Multi-channel SSIM via scikit-image
   - Higher is better (range [0, 1])
   - Measures structural similarity

3. **PSNR** (Peak Signal-to-Noise Ratio)
   - Higher is better (dB)
   - Measures pixel-level similarity

4. **Gram Distance**
   - Frobenius norm per layer: `||G_gen - G_style||_F`
   - Measures style similarity

5. **Covariance Distance**
   - Frobenius norm per layer: `||C_gen - C_style||_F`
   - Measures style similarity in PCA space

6. **Runtime**
   - Elapsed time per transfer

**MetricsComputer Class**:
- Caches VGG feature extractor
- Computes all metrics in one call
- Handles missing dependencies gracefully

### 7. Batch Experiments (`src/experiments.py`)

**Purpose**: Orchestrate systematic experiments and ablations.

**Key Functions**:

#### `run_alpha_grid()`
- Runs style transfer across multiple alpha values
- Supports multiple methods
- Creates visualization grids
- Computes metrics for all combinations
- Saves CSV with metrics

#### `run_batch_experiment()`
- Runs experiments across multiple:
  - Content images
  - Style pairs
  - Alpha values
  - Methods
- Organizes outputs by combination
- Aggregates metrics into single CSV

**Output Structure**:
```
output_dir/
├── metrics_summary.csv
├── <content>_<style1>_<style2>_<method>_grid.jpg
└── individual images...
```

### 8. Interactive UI (`app/streamlit_app.py`)

**Purpose**: User-friendly web interface for interactive style transfer.

**Features**:
- **Tab 1: Style Transfer**
  - Image upload/selection from examples
  - Method selection (PCA joint/simple, Gram-linear, Cov-linear, Gatys)
  - Real-time parameter adjustment (alpha, weights, iterations)
  - Side-by-side comparisons
  - Metrics display
  - Download results

- **Tab 2: Batch Experiments**
  - Configure batch runs
  - Select content/style pairs
  - Set alpha ranges
  - Generate grids

- **Tab 3: Precomputed Results**
  - View saved results
  - Compare metrics

**Technical Details**:
- Uses `@st.cache_resource` for model caching
- Session state for result history
- Progress bars for long operations

### 9. Configuration (`src/config.py`)

**Default Hyperparameters**:
```python
{
    'content_weight': 1e5,
    'style_weight': 3e4,
    'tv_weight': 1e0,
    'iterations': 1000,  # LBFGS
    'adam_iterations': 3000,  # Adam
    'height': 400,
    'optimizer': 'lbfgs',
    'alpha': 0.5,
    'mixing_method': 'joint',
    'init_method': 'content'
}
```

**Style Layers**: `['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']`
**Content Layer**: `'conv4_2'`

## Implementation Quality

### Strengths

1. **Modular Design**: Clear separation of concerns
2. **Type Hints**: Good use of type annotations
3. **Error Handling**: Graceful fallbacks (CPU if no CUDA, torchvision if no parent models)
4. **Documentation**: Comprehensive docstrings
5. **Testing**: Unit tests for core functionality
6. **Flexibility**: Supports multiple mixing methods and optimizers
7. **Evaluation**: Comprehensive metrics suite

### Code Quality

- **Consistent**: Follows Python conventions
- **Readable**: Clear variable names and structure
- **Maintainable**: Well-organized modules
- **Extensible**: Easy to add new mixing methods

## Testing

**Test Coverage**:
- `test_pca_code.py`: Tests covariance computation, PCA code extraction, save/load
- `test_mixing.py`: Tests all mixing strategies
- `test_metrics.py`: (Referenced but not shown)

**Test Quality**:
- Tests mathematical correctness (eigendecomposition, symmetry)
- Tests boundary cases (alpha=0, alpha=1)
- Tests shape consistency

## Usage Patterns

### CLI Usage
```bash
# Single transfer
python -m src.pca_gatys \
    --content data/content_examples/taj_mahal.jpg \
    --style1 data/style_examples/vg_starry_night.jpg \
    --style2 data/style_examples/candy.jpg \
    --alpha 0.5 \
    --method joint \
    --out results/output.jpg
```

### Python API
```python
from src.pca_gatys import pca_gatys_style_transfer
from src.config import DEFAULT_CONFIG

result, metrics = pca_gatys_style_transfer(
    content_path, style1_path, style2_path,
    alpha=0.5,
    mixing_method='joint',
    config=DEFAULT_CONFIG
)
```

### Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

## Dependencies

**Core**:
- PyTorch (>=2.0.0)
- torchvision (>=0.15.0)
- NumPy, OpenCV, Pillow

**Evaluation**:
- lpips (for LPIPS metric)
- scikit-image (for SSIM/PSNR)

**UI**:
- streamlit (>=1.28.0)

**Utilities**:
- matplotlib (for visualization)
- pandas (for metrics CSV)
- tqdm (for progress bars)
- pytest (for testing)

## Project Status

### ✅ Completed

1. **Core Algorithm**
   - ✅ Baseline Gatys NST
   - ✅ PCA code extraction
   - ✅ Simple PCA mixing
   - ✅ Joint PCA mixing
   - ✅ Covariance-linear mixing
   - ✅ Gram-linear mixing

2. **Infrastructure**
   - ✅ VGG feature extractor
   - ✅ Image I/O utilities
   - ✅ Configuration system
   - ✅ Utility functions

3. **Evaluation**
   - ✅ LPIPS metric
   - ✅ SSIM/PSNR metrics
   - ✅ Gram/covariance distances
   - ✅ Runtime measurement
   - ✅ Batch evaluation

4. **User Interface**
   - ✅ Streamlit web UI
   - ✅ Interactive parameter adjustment
   - ✅ Batch experiment interface

5. **Experiments**
   - ✅ Alpha grid generation
   - ✅ Batch experiment orchestrator
   - ✅ Metrics CSV export

6. **Testing**
   - ✅ Unit tests for PCA code
   - ✅ Unit tests for mixing
   - ✅ Test infrastructure

### 🔄 Potentially Missing / To Verify

1. **Experiments** (from requirements):
   - Need to verify systematic experiments have been run:
     - Visual interpolation grids (5 content × 6 style pairs × 5 alphas × 4 methods)
     - Ablation studies (layer sets, normalization, initialization, style weight sweep)
     - Quantitative comparisons (metrics tables)
     - User study (optional)

2. **Documentation**:
   - Project report (PDF) - not in codebase
   - Demo video - not in codebase
   - Presentation slides - not in codebase

3. **Scripts**:
   - Shell scripts exist but may need verification for Windows compatibility

## Key Algorithms

### PCA Mixing (Joint - Recommended)

```
For each style layer:
  1. Extract features F₁, F₂ from style images
  2. Compute covariances C₁, C₂
  3. Compute joint basis: P_mix = eigenvectors((C₁ + C₂) / 2)
  4. Project both: D₁_proj = diag(P_mix^T @ C₁ @ P_mix)
                  D₂_proj = diag(P_mix^T @ C₂ @ P_mix)
  5. Mix: D_mix = α·D₁_proj + (1-α)·D₂_proj
  6. Reconstruct: C_mix = P_mix @ diag(D_mix) @ P_mix^T
  7. Use C_mix as style target in optimization
```

### Optimization Loop

```
Initialize: x = content_image (or noise)
For iteration in range(iterations):
  1. Extract features from x
  2. Compute content loss: ||F_content(x) - F_content(content)||²
  3. Compute style loss: Σ_layers ||C(x, layer) - C_mix(layer)||²
  4. Compute TV loss: TV(x)
  5. Total loss = λ_c·L_content + λ_s·L_style + λ_tv·L_tv
  6. Backprop and update x
```

## Mathematical Correctness

The implementation correctly:
- ✅ Computes covariance matrices with proper normalization
- ✅ Performs eigendecomposition using symmetric eigensolver
- ✅ Reconstructs covariance from PCA: C = P @ diag(D) @ P^T
- ✅ Handles mean centering for covariance computation
- ✅ Maintains symmetry and positive semi-definiteness

## Performance Considerations

- **GPU Acceleration**: Automatically uses CUDA if available
- **Memory**: Feature extraction cached, models loaded once
- **Optimization**: LBFGS typically faster than Adam for this task
- **Image Size**: Default 400px height balances quality/speed

## Recommendations for Next Steps

1. **Run Systematic Experiments**:
   - Execute `run_batch_experiment()` with diverse content/style pairs
   - Generate grids for all alpha values
   - Compute metrics tables

2. **Ablation Studies**:
   - Test different layer combinations
   - Compare initialization methods
   - Sweep style weights

3. **Documentation**:
   - Write project report (PDF)
   - Create demo video
   - Prepare presentation slides

4. **Polish**:
   - Verify all scripts work on target platform
   - Add more example outputs
   - Clean up any temporary files

## Conclusion

This is a **well-implemented, comprehensive codebase** that successfully implements:
- ✅ Original Gatys NST as baseline
- ✅ PCA-based style mixing (simple and joint)
- ✅ Baseline mixing methods for comparison
- ✅ Comprehensive evaluation metrics
- ✅ Interactive UI
- ✅ Batch experiment infrastructure

The code is **production-ready** and follows best practices. The main remaining work is running the systematic experiments and producing the final deliverables (report, video, slides).

