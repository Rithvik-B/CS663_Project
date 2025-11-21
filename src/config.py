"""
Configuration and hyperparameters for PCA-Mixed NST experiments.
"""

from typing import Dict, List, Optional
import os

# Default VGG layers for style and content
DEFAULT_STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
DEFAULT_CONTENT_LAYER = 'conv4_2'

# Default optimization parameters
DEFAULT_CONFIG: Dict = {
    # Model
    'model': 'vgg19',
    'style_layers': DEFAULT_STYLE_LAYERS,
    'content_layer': DEFAULT_CONTENT_LAYER,
    
    # Optimization
    'optimizer': 'lbfgs',  # 'lbfgs' or 'adam'
    'iterations': 1000,  # for LBFGS
    'adam_iterations': 3000,  # for Adam
    'lr': 1e1,  # learning rate for Adam
    
    # Loss weights
    'content_weight': 1e5,
    'style_weight': 3e4,
    'tv_weight': 1e0,
    
    # Initialization
    'init_method': 'content',  # 'content', 'random', or 'style'
    'height': 400,  # target image height
    
    # PCA mixing
    'alpha': 0.5,  # mixing coefficient (0.0 = style2, 1.0 = style1)
    'mixing_method': 'joint',  # 'simple', 'joint', 'covariance-linear', 'gram-linear'
    'per_layer_alpha': None,  # Optional dict mapping layer -> alpha override
    
    # Paths (relative to final_project/)
    'data_dir': 'data',
    'content_dir': 'data/content_examples',
    'style_dir': 'data/style_examples',
    'output_dir': 'data/outputs',
    'results_dir': 'results',
    
    # Device
    'device': 'cuda',  # will fallback to 'cpu' if CUDA unavailable
    
    # Saving
    'saving_freq': -1,  # -1 = only final, >0 = save every N iterations
    'save_intermediate': False,
    
    # Performance optimizations
    'use_pca_cache': True,  # Enable PCA code caching
    'pca_cache_dir': 'data/pca_cache',  # Directory for PCA code cache
}

# ImageNet normalization constants (matching existing repo)
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def get_project_root() -> str:
    """Get the absolute path to final_project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(relative_path: str) -> str:
    """Get absolute path to a data file/directory."""
    project_root = get_project_root()
    return os.path.join(project_root, 'data', relative_path)


def get_results_path(relative_path: str) -> str:
    """Get absolute path to a results file/directory."""
    project_root = get_project_root()
    return os.path.join(project_root, 'results', relative_path)


def update_config(config: Dict, **kwargs) -> Dict:
    """Update config dictionary with new values."""
    config = config.copy()
    config.update(kwargs)
    return config

