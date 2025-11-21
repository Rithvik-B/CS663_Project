"""
PCA code extraction: compute covariance matrices and eigendecomposition.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import os
import pickle
import hashlib
from pathlib import Path

from .vgg_features import VGGFeatureExtractor


class PCACode:
    """
    Container for PCA code: eigenvectors (P), eigenvalues (D), covariance (C), and mean.
    """
    
    def __init__(self, P: torch.Tensor, D: torch.Tensor, C: torch.Tensor, mean: torch.Tensor, layer_name: str):
        """
        Initialize PCA code.
        
        Args:
            P: Eigenvectors matrix, shape (C, C)
            D: Eigenvalues (diagonal), shape (C,)
            C: Covariance matrix, shape (C, C)
            mean: Feature mean, shape (C,)
            layer_name: Name of the layer this code corresponds to
        """
        self.P = P  # Eigenvectors
        self.D = D  # Eigenvalues (diagonal)
        self.C = C  # Covariance matrix
        self.mean = mean  # Feature mean
        self.layer_name = layer_name
    
    def to_device(self, device: torch.device) -> 'PCACode':
        """Move all tensors to device."""
        return PCACode(
            P=self.P.to(device),
            D=self.D.to(device),
            C=self.C.to(device),
            mean=self.mean.to(device),
            layer_name=self.layer_name
        )
    
    def save(self, filepath: str) -> None:
        """Save PCA code to file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        data = {
            'P': self.P.cpu(),
            'D': self.D.cpu(),
            'C': self.C.cpu(),
            'mean': self.mean.cpu(),
            'layer_name': self.layer_name
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None) -> 'PCACode':
        """Load PCA code from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        code = cls(
            P=data['P'],
            D=data['D'],
            C=data['C'],
            mean=data['mean'],
            layer_name=data['layer_name']
        )
        
        if device is not None:
            code = code.to_device(device)
        
        return code


def compute_covariance(features: torch.Tensor, center: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute covariance matrix from feature maps.
    
    Args:
        features: Feature tensor of shape (B, C, H, W) or (1, C, H, W)
        center: If True, center features before computing covariance
    
    Returns:
        Tuple of (covariance_matrix, mean)
        - covariance_matrix: shape (C, C)
        - mean: shape (C,)
    """
    # Flatten spatial dimensions: (B, C, H, W) -> (B, C, M) where M = H*W
    B, C, H, W = features.shape
    M = H * W
    features_flat = features.view(B, C, M)  # (B, C, M)
    
    # Average over batch dimension if needed
    if B > 1:
        features_flat = features_flat.mean(dim=0, keepdim=True)  # (1, C, M)
    
    # Remove batch dimension: (C, M)
    F = features_flat.squeeze(0)  # (C, M)
    
    # Compute mean
    mean = F.mean(dim=1)  # (C,)
    
    if center:
        # Center features
        F_centered = F - mean.unsqueeze(1)  # (C, M)
    else:
        F_centered = F
    
    # Compute covariance: C = (1/M) * F_centered @ F_centered^T
    # F_centered: (C, M), F_centered^T: (M, C)
    covariance = (1.0 / M) * torch.matmul(F_centered, F_centered.t())  # (C, C)
    
    return covariance, mean


def compute_pca_code(features: torch.Tensor, layer_name: str, center: bool = True) -> PCACode:
    """
    Compute PCA code from feature maps: eigendecomposition of covariance.
    
    Args:
        features: Feature tensor of shape (1, C, H, W)
        layer_name: Name of the layer
        center: If True, center features before computing covariance
    
    Returns:
        PCACode object containing P, D, C, mean
    """
    # Compute covariance
    C, mean = compute_covariance(features, center=center)
    
    # Eigendecomposition: C = P @ D @ P^T
    # Use eigh for symmetric matrices (more stable)
    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    
    # eigh returns eigenvalues in ascending order, we want descending
    eigenvalues = eigenvalues.flip(0)  # (C,)
    eigenvectors = eigenvectors.flip(1)  # (C, C) - flip columns
    
    # Ensure eigenvectors are column vectors (each column is an eigenvector)
    P = eigenvectors  # (C, C), columns are eigenvectors
    D = eigenvalues   # (C,), diagonal values
    
    return PCACode(P=P, D=D, C=C, mean=mean, layer_name=layer_name)


def _get_image_hash(img_tensor: torch.Tensor) -> str:
    """Generate hash from image tensor for caching."""
    # Use a small sample of pixels for fast hashing
    img_np = img_tensor.detach().cpu().numpy()
    # Sample every 10th pixel
    sample = img_np[0, :, ::10, ::10].tobytes()
    return hashlib.md5(sample).hexdigest()


def extract_pca_codes(
    feature_extractor: VGGFeatureExtractor,
    style_img: torch.Tensor,
    layer_names: Optional[list] = None,
    cache_dir: Optional[str] = None,
    style_img_path: Optional[str] = None
) -> Dict[str, PCACode]:
    """
    Extract PCA codes for all style layers from a style image.
    Supports caching to avoid recomputation.
    
    Args:
        feature_extractor: VGGFeatureExtractor instance
        style_img: Style image tensor (1, 3, H, W)
        layer_names: List of layer names to extract (default: all style layers)
        cache_dir: Optional directory to cache/load codes
        style_img_path: Optional path to style image (for cache key)
    
    Returns:
        Dictionary mapping layer names to PCACode objects
    """
    if layer_names is None:
        layer_names = feature_extractor.get_style_layer_names()
    
    codes = {}
    
    # Try to load from cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache key
        if style_img_path:
            cache_key = hashlib.md5(style_img_path.encode()).hexdigest()
        else:
            cache_key = _get_image_hash(style_img)
        
        cache_file = os.path.join(cache_dir, f"pca_code_{cache_key}.pkl")
        
        # Try loading cached codes
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Check if all layers are present
                    if all(layer in cached_data for layer in layer_names):
                        device = style_img.device
                        codes = {layer: cached_data[layer].to_device(device) 
                                for layer in layer_names}
                        return codes
            except Exception as e:
                # If cache load fails, recompute
                pass
    
    # Extract features
    style_features = feature_extractor.get_style_features(style_img)
    
    # Compute PCA codes for each layer
    for layer_name in layer_names:
        if layer_name not in style_features:
            continue
        
        features = style_features[layer_name]
        code = compute_pca_code(features, layer_name=layer_name)
        codes[layer_name] = code
    
    # Save to cache if cache_dir provided
    if cache_dir and codes:
        try:
            cache_file = os.path.join(cache_dir, f"pca_code_{cache_key}.pkl")
            # Move to CPU for caching
            codes_cpu = {layer: code.to_device(torch.device('cpu')) 
                        for layer, code in codes.items()}
            with open(cache_file, 'wb') as f:
                pickle.dump(codes_cpu, f)
        except Exception as e:
            # If cache save fails, continue without caching
            pass
    
    return codes


def reconstruct_covariance_from_pca(code: PCACode) -> torch.Tensor:
    """
    Reconstruct covariance matrix from PCA code: C = P @ diag(D) @ P^T
    
    Args:
        code: PCACode object
    
    Returns:
        Reconstructed covariance matrix
    """
    D_diag = torch.diag(code.D)  # (C, C)
    C_reconstructed = torch.matmul(torch.matmul(code.P, D_diag), code.P.t())
    return C_reconstructed

