"""
Mixing strategies for combining two style representations.
"""

import torch
from typing import Dict, Optional, Tuple
from .pca_code import PCACode, compute_covariance, compute_pca_code
from .vgg_features import VGGFeatureExtractor


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix from feature maps (optimized vectorized version).
    
    Args:
        features: Feature tensor of shape (1, C, H, W)
    
    Returns:
        Gram matrix of shape (1, C, C)
    """
    B, C, H, W = features.shape
    M = H * W
    # Vectorized: flatten and compute in single operation
    features_flat = features.view(B, C, M)  # (1, C, M)
    # Use bmm for batched matrix multiplication (faster than manual loops)
    gram = torch.bmm(features_flat, features_flat.transpose(1, 2))  # (1, C, C)
    # Normalize (precompute constant)
    gram = gram / (C * M)  # Normalize by C*H*W
    return gram


def simple_pca_mix(
    code1: PCACode,
    code2: PCACode,
    alpha: float,
    layer_name: str
) -> PCACode:
    """
    Simple PCA mixing: use P1 basis, mix eigenvalues (optimized).
    
    Dmix = α * D1 + (1-α) * diag(P1^T @ C2 @ P1)
    
    Args:
        code1: First style PCA code
        code2: Second style PCA code
        alpha: Mixing coefficient (0.0 = style2, 1.0 = style1)
        layer_name: Layer name for output code
    
    Returns:
        Mixed PCA code
    """
    # Use P1 as the basis
    P_mix = code1.P
    
    # D1 eigenvalues
    D1 = code1.D
    
    # Project C2 onto P1 basis: diag(P1^T @ C2 @ P1)
    # Optimized: use single chain of matmuls
    C2_projected = torch.matmul(torch.matmul(P_mix.t(), code2.C), P_mix)
    D2_projected = torch.diag(C2_projected)  # Extract diagonal
    
    # Mix eigenvalues (vectorized)
    D_mix = alpha * D1 + (1 - alpha) * D2_projected
    
    # Reconstruct covariance: C_mix = P_mix @ diag(D_mix) @ P_mix^T
    # Optimized: avoid creating full diagonal matrix if possible
    D_mix_diag = torch.diag(D_mix)
    C_mix = torch.matmul(torch.matmul(P_mix, D_mix_diag), P_mix.t())
    
    # Mix means (vectorized)
    mean_mix = alpha * code1.mean + (1 - alpha) * code2.mean
    
    return PCACode(P=P_mix, D=D_mix, C=C_mix, mean=mean_mix, layer_name=layer_name)


def joint_pca_mix(
    code1: PCACode,
    code2: PCACode,
    alpha: float,
    layer_name: str
) -> PCACode:
    """
    Joint PCA mixing: compute eigenvectors of (C1+C2)/2, then mix eigenvalues (optimized).
    
    Pmix = eigenvectors of (C1 + C2) / 2
    Dmix = α * diag(Pmix^T @ C1 @ Pmix) + (1-α) * diag(Pmix^T @ C2 @ Pmix)
    
    Args:
        code1: First style PCA code
        code2: Second style PCA code
        alpha: Mixing coefficient
        layer_name: Layer name for output code
    
    Returns:
        Mixed PCA code
    """
    # Compute average covariance (vectorized)
    C_avg = (code1.C + code2.C) / 2.0
    
    # Eigendecomposition of average (use eigh for symmetric matrices - faster and more stable)
    eigenvalues, eigenvectors = torch.linalg.eigh(C_avg)
    # eigh returns ascending, flip to descending
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    P_mix = eigenvectors
    
    # Project both covariances onto joint basis (optimized: compute both in sequence)
    C1_projected = torch.matmul(torch.matmul(P_mix.t(), code1.C), P_mix)
    C2_projected = torch.matmul(torch.matmul(P_mix.t(), code2.C), P_mix)
    
    D1_projected = torch.diag(C1_projected)
    D2_projected = torch.diag(C2_projected)
    
    # Mix eigenvalues (vectorized)
    D_mix = alpha * D1_projected + (1 - alpha) * D2_projected
    
    # Reconstruct covariance
    D_mix_diag = torch.diag(D_mix)
    C_mix = torch.matmul(torch.matmul(P_mix, D_mix_diag), P_mix.t())
    
    # Mix means (vectorized)
    mean_mix = alpha * code1.mean + (1 - alpha) * code2.mean
    
    return PCACode(P=P_mix, D=D_mix, C=C_mix, mean=mean_mix, layer_name=layer_name)


def covariance_linear_mix(
    code1: PCACode,
    code2: PCACode,
    alpha: float,
    layer_name: str
) -> PCACode:
    """
    Linear mixing of covariance matrices: Cmix = α * C1 + (1-α) * C2
    
    Then recompute eigendecomposition.
    
    Args:
        code1: First style PCA code
        code2: Second style PCA code
        alpha: Mixing coefficient
        layer_name: Layer name for output code
    
    Returns:
        Mixed PCA code
    """
    # Linear mix of covariances
    C_mix = alpha * code1.C + (1 - alpha) * code2.C
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(C_mix)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    P_mix = eigenvectors
    D_mix = eigenvalues
    
    # Mix means
    mean_mix = alpha * code1.mean + (1 - alpha) * code2.mean
    
    return PCACode(P=P_mix, D=D_mix, C=C_mix, mean=mean_mix, layer_name=layer_name)


def gram_linear_mix(
    gram1: torch.Tensor,
    gram2: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Linear mixing of Gram matrices: Gmix = α * G1 + (1-α) * G2
    
    Args:
        gram1: First style Gram matrix, shape (1, C, C)
        gram2: Second style Gram matrix, shape (1, C, C)
        alpha: Mixing coefficient
    
    Returns:
        Mixed Gram matrix, shape (1, C, C)
    """
    gram_mix = alpha * gram1 + (1 - alpha) * gram2
    return gram_mix


def mix_style_codes(
    codes1: Dict[str, PCACode],
    codes2: Dict[str, PCACode],
    alpha: float,
    method: str = 'joint',
    per_layer_alpha: Optional[Dict[str, float]] = None
) -> Dict[str, PCACode]:
    """
    Mix two sets of style PCA codes across all layers.
    
    Args:
        codes1: Dictionary of PCA codes for style 1
        codes2: Dictionary of PCA codes for style 2
        alpha: Global mixing coefficient
        method: Mixing method ('simple', 'joint', 'covariance-linear')
        per_layer_alpha: Optional dict mapping layer -> alpha override
    
    Returns:
        Dictionary of mixed PCA codes
    """
    mixed_codes = {}
    
    # Get common layers
    common_layers = set(codes1.keys()) & set(codes2.keys())
    
    for layer_name in common_layers:
        # Use per-layer alpha if provided, otherwise global alpha
        layer_alpha = per_layer_alpha.get(layer_name, alpha) if per_layer_alpha else alpha
        
        code1 = codes1[layer_name]
        code2 = codes2[layer_name]
        
        if method == 'simple':
            mixed_code = simple_pca_mix(code1, code2, layer_alpha, layer_name)
        elif method == 'joint':
            mixed_code = joint_pca_mix(code1, code2, layer_alpha, layer_name)
        elif method == 'covariance-linear':
            mixed_code = covariance_linear_mix(code1, code2, layer_alpha, layer_name)
        else:
            raise ValueError(f"Unknown mixing method: {method}")
        
        mixed_codes[layer_name] = mixed_code
    
    return mixed_codes


def mix_gram_matrices(
    grams1: Dict[str, torch.Tensor],
    grams2: Dict[str, torch.Tensor],
    alpha: float,
    per_layer_alpha: Optional[Dict[str, float]] = None
) -> Dict[str, torch.Tensor]:
    """
    Mix two sets of Gram matrices across all layers.
    
    Args:
        grams1: Dictionary of Gram matrices for style 1
        grams2: Dictionary of Gram matrices for style 2
        alpha: Global mixing coefficient
        per_layer_alpha: Optional dict mapping layer -> alpha override
    
    Returns:
        Dictionary of mixed Gram matrices
    """
    mixed_grams = {}
    
    common_layers = set(grams1.keys()) & set(grams2.keys())
    
    for layer_name in common_layers:
        layer_alpha = per_layer_alpha.get(layer_name, alpha) if per_layer_alpha else alpha
        mixed_gram = gram_linear_mix(grams1[layer_name], grams2[layer_name], layer_alpha)
        mixed_grams[layer_name] = mixed_gram
    
    return mixed_grams

