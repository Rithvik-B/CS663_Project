"""
Tests for mixing strategies.
"""

import torch
import pytest
import numpy as np
from src.mixing import (
    simple_pca_mix, joint_pca_mix, covariance_linear_mix,
    gram_linear_mix, mix_style_codes, gram_matrix
)
from src.pca_code import PCACode, compute_pca_code


def create_test_pca_code(C: int = 32) -> PCACode:
    """Create a test PCA code."""
    # Create synthetic features
    features = torch.randn(1, C, 16, 16)
    return compute_pca_code(features, layer_name="test")


def test_simple_pca_mix():
    """Test simple PCA mixing."""
    code1 = create_test_pca_code(32)
    code2 = create_test_pca_code(32)
    
    alpha = 0.5
    mixed = simple_pca_mix(code1, code2, alpha, "test")
    
    # Check shapes
    assert mixed.P.shape == code1.P.shape
    assert mixed.D.shape == code1.D.shape
    assert mixed.C.shape == code1.C.shape
    assert mixed.mean.shape == code1.mean.shape
    
    # Check boundary cases
    mixed_alpha0 = simple_pca_mix(code1, code2, 0.0, "test")
    mixed_alpha1 = simple_pca_mix(code1, code2, 1.0, "test")
    
    # At alpha=1, should be close to code1 (allowing for projection differences)
    # At alpha=0, should use code2 projected onto code1's basis


def test_joint_pca_mix():
    """Test joint PCA mixing."""
    code1 = create_test_pca_code(32)
    code2 = create_test_pca_code(32)
    
    alpha = 0.5
    mixed = joint_pca_mix(code1, code2, alpha, "test")
    
    # Check shapes
    assert mixed.P.shape == code1.P.shape
    assert mixed.D.shape == code1.D.shape
    assert mixed.C.shape == code1.C.shape
    
    # Check eigendecomposition still holds
    D_diag = torch.diag(mixed.D)
    C_reconstructed = torch.matmul(torch.matmul(mixed.P, D_diag), mixed.P.t())
    assert torch.allclose(mixed.C, C_reconstructed, atol=1e-4)


def test_covariance_linear_mix():
    """Test covariance linear mixing."""
    code1 = create_test_pca_code(32)
    code2 = create_test_pca_code(32)
    
    alpha = 0.5
    mixed = covariance_linear_mix(code1, code2, alpha, "test")
    
    # Check that covariance is linear mix
    expected_C = alpha * code1.C + (1 - alpha) * code2.C
    assert torch.allclose(mixed.C, expected_C, atol=1e-5)
    
    # Check eigendecomposition
    D_diag = torch.diag(mixed.D)
    C_reconstructed = torch.matmul(torch.matmul(mixed.P, D_diag), mixed.P.t())
    assert torch.allclose(mixed.C, C_reconstructed, atol=1e-4)


def test_gram_linear_mix():
    """Test Gram matrix linear mixing."""
    C = 32
    gram1 = torch.randn(1, C, C)
    gram2 = torch.randn(1, C, C)
    
    alpha = 0.5
    mixed = gram_linear_mix(gram1, gram2, alpha)
    
    # Check linear mix
    expected = alpha * gram1 + (1 - alpha) * gram2
    assert torch.allclose(mixed, expected, atol=1e-5)


def test_mix_style_codes():
    """Test mixing multiple style codes."""
    codes1 = {
        'layer1': create_test_pca_code(32),
        'layer2': create_test_pca_code(64)
    }
    codes2 = {
        'layer1': create_test_pca_code(32),
        'layer2': create_test_pca_code(64)
    }
    
    alpha = 0.5
    mixed = mix_style_codes(codes1, codes2, alpha, method='joint')
    
    # Check all layers mixed
    assert set(mixed.keys()) == {'layer1', 'layer2'}
    assert mixed['layer1'].layer_name == 'layer1'
    assert mixed['layer2'].layer_name == 'layer2'


def test_gram_matrix():
    """Test Gram matrix computation."""
    B, C, H, W = 1, 32, 16, 16
    features = torch.randn(B, C, H, W)
    
    gram = gram_matrix(features)
    
    # Check shape
    assert gram.shape == (B, C, C)
    
    # Check symmetry
    assert torch.allclose(gram[0], gram[0].t(), atol=1e-5)
    
    # Check positive semi-definite
    eigenvalues = torch.linalg.eigvalsh(gram[0])
    assert torch.all(eigenvalues >= -1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

