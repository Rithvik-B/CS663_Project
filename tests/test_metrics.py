"""
Tests for evaluation metrics.
"""

import torch
import pytest
import numpy as np
from src.metrics import MetricsComputer


def test_gram_distance():
    """Test Gram distance computation."""
    device = torch.device('cpu')
    computer = MetricsComputer(device=device)
    
    # Create synthetic images
    img1 = torch.randn(1, 3, 64, 64) * 127.5 + 127.5  # [0, 255] range
    img2 = torch.randn(1, 3, 64, 64) * 127.5 + 127.5
    
    # Compute Gram distances
    distances = computer.compute_gram_distance(img1, img2)
    
    # Check that distances are non-negative
    assert all(d >= 0 for d in distances.values())
    
    # Distance to self should be small
    self_distances = computer.compute_gram_distance(img1, img1)
    assert all(d < 1e-3 for d in self_distances.values())


def test_covariance_distance():
    """Test covariance distance computation."""
    device = torch.device('cpu')
    computer = MetricsComputer(device=device)
    
    # Create synthetic images
    img1 = torch.randn(1, 3, 64, 64) * 127.5 + 127.5
    img2 = torch.randn(1, 3, 64, 64) * 127.5 + 127.5
    
    # Compute covariance distances
    distances = computer.compute_covariance_distance(img1, img2)
    
    # Check that distances are non-negative
    assert all(d >= 0 for d in distances.values())
    
    # Distance to self should be small
    self_distances = computer.compute_covariance_distance(img1, img1)
    assert all(d < 1e-3 for d in self_distances.values())


def test_compute_all_metrics():
    """Test computing all metrics."""
    device = torch.device('cpu')
    computer = MetricsComputer(device=device)
    
    # Create synthetic images
    generated = torch.randn(1, 3, 64, 64) * 127.5 + 127.5
    content = torch.randn(1, 3, 64, 64) * 127.5 + 127.5
    style1 = torch.randn(1, 3, 64, 64) * 127.5 + 127.5
    
    # Compute all metrics
    metrics = computer.compute_all_metrics(generated, content, style1, runtime=1.5)
    
    # Check that metrics are computed
    assert 'lpips_content' in metrics
    assert 'ssim_content' in metrics
    assert 'psnr_content' in metrics
    assert 'runtime_seconds' in metrics
    assert metrics['runtime_seconds'] == 1.5
    
    # Check style distances exist
    assert 'gram_dist_style1_avg' in metrics
    assert 'cov_dist_style1_avg' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

