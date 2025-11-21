"""
VGG feature extractor wrapper that safely uses existing models from parent repo.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import sys
import os

# Add parent directory to path to import existing models
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from models.definitions.vgg_nets import Vgg19, Vgg16
except ImportError:
    # Fallback: use torchvision directly if models not available
    from torchvision import models
    Vgg19 = None
    Vgg16 = None


class VGGFeatureExtractor:
    """
    Wrapper for VGG feature extraction that provides consistent interface
    for style and content layers.
    """
    
    def __init__(self, model_name: str = 'vgg19', device: torch.device = None):
        """
        Initialize VGG feature extractor.
        
        Args:
            model_name: 'vgg19' or 'vgg16'
            device: torch device (defaults to cuda if available)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.model_name = model_name
        
        # Use existing models if available, otherwise torchvision
        if model_name == 'vgg19':
            if Vgg19 is not None:
                self.model = Vgg19(requires_grad=False, show_progress=True).to(device).eval()
                self.layer_names = self.model.layer_names
                self.content_idx = self.model.content_feature_maps_index
                self.style_indices = self.model.style_feature_maps_indices
            else:
                # Fallback to torchvision
                vgg = models.vgg19(pretrained=True).features.to(device).eval()
                self.model = self._wrap_torchvision_vgg19(vgg)
                self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']
                self.content_idx = 4  # conv4_2
                self.style_indices = [0, 1, 2, 3, 5]  # all except conv4_2
        elif model_name == 'vgg16':
            if Vgg16 is not None:
                self.model = Vgg16(requires_grad=False, show_progress=True).to(device).eval()
                self.layer_names = self.model.layer_names
                self.content_idx = self.model.content_feature_maps_index
                self.style_indices = self.model.style_feature_maps_indices
            else:
                vgg = models.vgg16(pretrained=True).features.to(device).eval()
                self.model = self._wrap_torchvision_vgg16(vgg)
                self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
                self.content_idx = 1  # relu2_2
                self.style_indices = [0, 1, 2, 3]
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Create layer name to index mapping
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(self.layer_names)}
    
    def _wrap_torchvision_vgg19(self, features):
        """Wrap torchvision VGG19 to match existing interface."""
        class WrappedVGG19(nn.Module):
            def __init__(self, features):
                super().__init__()
                self.features = features
            
            def forward(self, x):
                outputs = []
                for i, layer in enumerate(self.features):
                    x = layer(x)
                    # Capture at relu1_1, relu2_1, relu3_1, relu4_1, conv4_2, relu5_1
                    if i in [1, 6, 11, 20, 21, 28]:
                        outputs.append(x)
                from collections import namedtuple
                VggOutputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1'])
                return VggOutputs(*outputs)
        
        return WrappedVGG19(features)
    
    def _wrap_torchvision_vgg16(self, features):
        """Wrap torchvision VGG16 to match existing interface."""
        class WrappedVGG16(nn.Module):
            def __init__(self, features):
                super().__init__()
                self.features = features
            
            def forward(self, x):
                outputs = []
                for i, layer in enumerate(self.features):
                    x = layer(x)
                    if i in [3, 8, 15, 22]:
                        outputs.append(x)
                from collections import namedtuple
                VggOutputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
                return VggOutputs(*outputs)
        
        return WrappedVGG16(features)
    
    def extract_features(self, img: torch.Tensor, requires_grad: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract features from image at all layers (optimized single forward pass).
        
        Args:
            img: Input tensor of shape (1, 3, H, W)
            requires_grad: If True, allow gradients to flow (for optimizing_img)
        
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        # Single forward pass through VGG (optimized - no separate calls per layer)
        if requires_grad:
            outputs = self.model(img)
        else:
            with torch.no_grad():
                outputs = self.model(img)
        
        # Build feature dictionary (vectorized)
        features = {}
        for idx, name in enumerate(self.layer_names):
            features[name] = outputs[idx]
        return features
    
    def get_style_features(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract style features (all style layers).
        
        Args:
            img: Input tensor
        
        Returns:
            Dictionary mapping style layer names to feature tensors
        """
        all_features = self.extract_features(img)
        style_features = {name: all_features[name] for idx, name in enumerate(self.layer_names) 
                         if idx in self.style_indices}
        return style_features
    
    def get_content_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract content features (content layer only).
        
        Args:
            img: Input tensor
        
        Returns:
            Content feature tensor
        """
        all_features = self.extract_features(img)
        content_layer_name = self.layer_names[self.content_idx]
        return all_features[content_layer_name]
    
    def get_layer_index(self, layer_name: str) -> int:
        """Get index of a layer by name."""
        return self.layer_name_to_idx.get(layer_name, -1)
    
    def get_style_layer_names(self) -> List[str]:
        """Get list of style layer names."""
        return [self.layer_names[idx] for idx in self.style_indices]

