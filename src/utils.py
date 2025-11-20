"""
Utility functions: seeding, logging, small helpers.
"""

import random
import numpy as np
import torch
import logging
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device, with fallback to CPU if CUDA unavailable."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    return torch.device(device)


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if not."""
    import os
    os.makedirs(path, exist_ok=True)

