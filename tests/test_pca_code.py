"""
Tests for PCA code extraction and operations.
"""

import torch
import pytest
import numpy as np
from src.pca_code import compute_covariance, compute_pca_code, PCACode, reconstruct_covariance_from_pca


def test_compute_covariance():
    """Test covariance computation."""
    # Create synthetic feature maps
    B, C, H, W = 1, 64, 32, 32
    features = torch.randn(B, C, H, W)
    
    # Compute covariance
    cov, mean = compute_covariance(features, center=True)
    
    # Check shapes
    assert cov.shape == (C, C)
    assert mean.shape == (C,)
    
    # Check symmetry
    assert torch.allclose(cov, cov.t(), atol=1e-5)
    
    # Check positive semi-definite (eigenvalues >= 0)
    eigenvalues = torch.linalg.eigvalsh(cov)
    assert torch.all(eigenvalues >= -1e-5)  # Allow small numerical errors


def test_pca_code():
    """Test PCA code computation."""
    # Create synthetic features
    B, C, H, W = 1, 32, 16, 16
    features = torch.randn(B, C, H, W)
    
    # Compute PCA code
    code = compute_pca_code(features, layer_name="test_layer")
    
    # Check attributes
    assert code.P.shape == (C, C)
    assert code.D.shape == (C,)
    assert code.C.shape == (C, C)
    assert code.mean.shape == (C,)
    assert code.layer_name == "test_layer"
    
    # Check eigendecomposition: C = P @ diag(D) @ P^T
    D_diag = torch.diag(code.D)
    C_reconstructed = torch.matmul(torch.matmul(code.P, D_diag), code.P.t())
    assert torch.allclose(code.C, C_reconstructed, atol=1e-4)


def test_reconstruct_covariance():
    """Test covariance reconstruction from PCA code."""
    # Create synthetic features
    B, C, H, W = 1, 32, 16, 16
    features = torch.randn(B, C, H, W)
    
    # Compute PCA code
    code = compute_pca_code(features, layer_name="test")
    
    # Reconstruct
    C_reconstructed = reconstruct_covariance_from_pca(code)
    
    # Should match original
    assert torch.allclose(code.C, C_reconstructed, atol=1e-4)


def test_pca_code_save_load():
    """Test saving and loading PCA codes."""
    import tempfile
    import os
    
    # Create synthetic code
    C = 32
    code = PCACode(
        P=torch.randn(C, C),
        D=torch.rand(C),
        C=torch.randn(C, C),
        mean=torch.randn(C),
        layer_name="test_layer"
    )
    
    # Save and load
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    try:
        code.save(temp_path)
        loaded_code = PCACode.load(temp_path)
        
        # Check all attributes match
        assert torch.allclose(code.P, loaded_code.P)
        assert torch.allclose(code.D, loaded_code.D)
        assert torch.allclose(code.C, loaded_code.C)
        assert torch.allclose(code.mean, loaded_code.mean)
        assert code.layer_name == loaded_code.layer_name
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

