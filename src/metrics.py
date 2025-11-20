"""
Evaluation metrics: LPIPS, SSIM, PSNR, Gram/covariance distances, runtime.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import os
from pathlib import Path
import pandas as pd

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Install with: pip install scikit-image")

from .vgg_features import VGGFeatureExtractor
from .pca_code import compute_covariance, extract_pca_codes
from .mixing import gram_matrix
from .io_utils import load_image, tensor_to_image


class MetricsComputer:
    """Compute various metrics for style transfer results."""
    
    def __init__(self, device: Optional[torch.device] = None, model_name: str = 'vgg19'):
        """
        Initialize metrics computer.
        
        Args:
            device: torch device
            model_name: VGG model name
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.feature_extractor = VGGFeatureExtractor(model_name=model_name, device=device)
        
        # Initialize LPIPS if available
        self.lpips_model = None
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
    
    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Compute LPIPS distance between two images.
        
        Args:
            img1: First image tensor (1, 3, H, W) in [0, 255] range
            img2: Second image tensor (1, 3, H, W) in [0, 255] range
        
        Returns:
            LPIPS distance (lower is better)
        """
        if not LPIPS_AVAILABLE or self.lpips_model is None:
            return float('nan')
        
        # LPIPS expects images in [-1, 1] range
        img1_norm = (img1 / 127.5) - 1.0
        img2_norm = (img2 / 127.5) - 1.0
        
        with torch.no_grad():
            dist = self.lpips_model(img1_norm, img2_norm)
        
        return dist.item()
    
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Compute SSIM between two images.
        
        Args:
            img1: First image tensor (1, 3, H, W)
            img2: Second image tensor (1, 3, H, W)
        
        Returns:
            SSIM score (higher is better, in [0, 1])
        """
        if not SKIMAGE_AVAILABLE:
            return float('nan')
        
        # Convert to numpy
        img1_np = tensor_to_image(img1, denormalize=False)
        img2_np = tensor_to_image(img2, denormalize=False)
        
        # Convert to grayscale for SSIM (or use multichannel=True)
        if len(img1_np.shape) == 3 and img1_np.shape[2] == 3:
            # Multi-channel SSIM
            ssim_val = ssim(img1_np, img2_np, data_range=1.0, multichannel=True, channel_axis=2)
        else:
            ssim_val = ssim(img1_np, img2_np, data_range=1.0)
        
        return float(ssim_val)
    
    def compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Compute PSNR between two images.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
        
        Returns:
            PSNR in dB (higher is better)
        """
        if not SKIMAGE_AVAILABLE:
            return float('nan')
        
        img1_np = tensor_to_image(img1, denormalize=False)
        img2_np = tensor_to_image(img2, denormalize=False)
        
        psnr_val = psnr(img1_np, img2_np, data_range=1.0)
        return float(psnr_val)
    
    def compute_gram_distance(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute Gram matrix distance per layer.
        
        Args:
            img1: First image
            img2: Second image
            layer_names: List of layer names (default: all style layers)
        
        Returns:
            Dictionary mapping layer names to Frobenius distances
        """
        if layer_names is None:
            layer_names = self.feature_extractor.get_style_layer_names()
        
        features1 = self.feature_extractor.get_style_features(img1)
        features2 = self.feature_extractor.get_style_features(img2)
        
        distances = {}
        for layer_name in layer_names:
            if layer_name not in features1 or layer_name not in features2:
                continue
            
            gram1 = gram_matrix(features1[layer_name])
            gram2 = gram_matrix(features2[layer_name])
            
            # Frobenius norm
            diff = gram1[0] - gram2[0]  # Remove batch dimension
            dist = torch.norm(diff, p='fro').item()
            distances[layer_name] = dist
        
        return distances
    
    def compute_covariance_distance(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute covariance matrix distance per layer.
        
        Args:
            img1: First image
            img2: Second image
            layer_names: List of layer names
        
        Returns:
            Dictionary mapping layer names to Frobenius distances
        """
        if layer_names is None:
            layer_names = self.feature_extractor.get_style_layer_names()
        
        features1 = self.feature_extractor.get_style_features(img1)
        features2 = self.feature_extractor.get_style_features(img2)
        
        distances = {}
        for layer_name in layer_names:
            if layer_name not in features1 or layer_name not in features2:
                continue
            
            cov1, _ = compute_covariance(features1[layer_name])
            cov2, _ = compute_covariance(features2[layer_name])
            
            # Frobenius norm
            diff = cov1 - cov2
            dist = torch.norm(diff, p='fro').item()
            distances[layer_name] = dist
        
        return distances
    
    def compute_all_metrics(
        self,
        generated_img: torch.Tensor,
        content_img: torch.Tensor,
        style1_img: Optional[torch.Tensor] = None,
        style2_img: Optional[torch.Tensor] = None,
        runtime: Optional[float] = None
    ) -> Dict:
        """
        Compute all available metrics.
        
        Args:
            generated_img: Generated result image
            content_img: Original content image
            style1_img: Optional first style image
            style2_img: Optional second style image
            runtime: Optional runtime in seconds
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Content similarity metrics
        metrics['lpips_content'] = self.compute_lpips(generated_img, content_img)
        metrics['ssim_content'] = self.compute_ssim(generated_img, content_img)
        metrics['psnr_content'] = self.compute_psnr(generated_img, content_img)
        
        # Style distances
        if style1_img is not None:
            gram_dists1 = self.compute_gram_distance(generated_img, style1_img)
            cov_dists1 = self.compute_covariance_distance(generated_img, style1_img)
            
            for layer, dist in gram_dists1.items():
                metrics[f'gram_dist_style1_{layer}'] = dist
            for layer, dist in cov_dists1.items():
                metrics[f'cov_dist_style1_{layer}'] = dist
            
            # Aggregate distances
            metrics['gram_dist_style1_avg'] = np.mean(list(gram_dists1.values()))
            metrics['cov_dist_style1_avg'] = np.mean(list(cov_dists1.values()))
        
        if style2_img is not None:
            gram_dists2 = self.compute_gram_distance(generated_img, style2_img)
            cov_dists2 = self.compute_covariance_distance(generated_img, style2_img)
            
            for layer, dist in gram_dists2.items():
                metrics[f'gram_dist_style2_{layer}'] = dist
            for layer, dist in cov_dists2.items():
                metrics[f'cov_dist_style2_{layer}'] = dist
            
            metrics['gram_dist_style2_avg'] = np.mean(list(gram_dists2.values()))
            metrics['cov_dist_style2_avg'] = np.mean(list(cov_dists2.values()))
        
        if runtime is not None:
            metrics['runtime_seconds'] = runtime
        
        return metrics


def evaluate_directory(
    gen_dir: str,
    content_dir: str,
    styles_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    """
    Evaluate all generated images in a directory.
    
    Args:
        gen_dir: Directory containing generated images
        content_dir: Directory containing content images
        styles_dir: Optional directory containing style images
        output_csv: Optional path to save CSV
        device: torch device
    
    Returns:
        DataFrame with metrics for each image
    """
    from .io_utils import prepare_img
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    computer = MetricsComputer(device=device)
    
    gen_path = Path(gen_dir)
    content_path = Path(content_dir)
    
    results = []
    
    # Find all generated images
    gen_images = list(gen_path.glob('*.jpg')) + list(gen_path.glob('*.png'))
    
    for gen_img_path in gen_images:
        # Try to find corresponding content image
        # Assume naming convention: content_style_alpha.jpg
        content_name = gen_img_path.stem.split('_')[0]  # Simple heuristic
        
        content_img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = content_path / f"{content_name}{ext}"
            if candidate.exists():
                content_img_path = candidate
                break
        
        if content_img_path is None:
            print(f"Warning: Could not find content image for {gen_img_path.name}")
            continue
        
        try:
            gen_img = prepare_img(str(gen_img_path), None, device)
            content_img = prepare_img(str(content_img_path), None, device)
            
            metrics = computer.compute_all_metrics(gen_img, content_img)
            metrics['generated_image'] = gen_img_path.name
            metrics['content_image'] = content_img_path.name
            
            results.append(metrics)
        except Exception as e:
            print(f"Error processing {gen_img_path.name}: {e}")
    
    df = pd.DataFrame(results)
    
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        df.to_csv(output_csv, index=False)
    
    return df

