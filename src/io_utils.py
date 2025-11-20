"""
Image I/O utilities: loading, saving, transforms, visualization grids.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List, Optional, Tuple, Union
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .config import IMAGENET_MEAN_255, IMAGENET_STD_NEUTRAL


def load_image(img_path: str, target_shape: Optional[Union[int, Tuple[int, int]]] = None) -> np.ndarray:
    """
    Load image from path, optionally resize.
    
    Args:
        img_path: Path to image file
        target_shape: If int, resize to this height (maintain aspect). 
                     If tuple (H, W), resize to exact dimensions.
                     If None, keep original size.
    
    Returns:
        Image as numpy array in RGB format, shape (H, W, 3), values in [0, 1]
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'Image not found: {img_path}')
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'Could not read image: {img_path}')
    
    img = img[:, :, ::-1]  # BGR to RGB
    
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif isinstance(target_shape, (tuple, list)) and len(target_shape) == 2:
            img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    
    img = img.astype(np.float32) / 255.0  # [0, 1] range
    return img


def prepare_img(img_path: str, target_shape: Optional[Union[int, Tuple[int, int]]], device: torch.device) -> torch.Tensor:
    """
    Load and prepare image for VGG network input.
    
    Args:
        img_path: Path to image
        target_shape: Target shape (int for height, or (H, W) tuple)
        device: torch device
    
    Returns:
        Tensor of shape (1, 3, H, W), normalized with ImageNet stats
    """
    img = load_image(img_path, target_shape=target_shape)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),  # [0, 1] -> [0, 255]
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])
    
    img_tensor = transform(img).to(device).unsqueeze(0)
    return img_tensor


def tensor_to_image(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    """
    Convert tensor to numpy image array.
    
    Args:
        tensor: Tensor of shape (1, 3, H, W) or (3, H, W)
        denormalize: If True, reverse ImageNet normalization
    
    Returns:
        Image array of shape (H, W, 3), values in [0, 255], uint8
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    img = tensor.detach().cpu().numpy()
    img = np.moveaxis(img, 0, 2)  # (C, H, W) -> (H, W, C)
    
    if denormalize:
        img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def save_image(img: Union[np.ndarray, torch.Tensor], img_path: str, denormalize: bool = True) -> None:
    """
    Save image to file.
    
    Args:
        img: Image as numpy array (H, W, 3) or torch tensor
        img_path: Output path
        denormalize: If True and tensor provided, reverse normalization
    """
    if isinstance(img, torch.Tensor):
        img = tensor_to_image(img, denormalize=denormalize)
    
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(img_path) if os.path.dirname(img_path) else '.', exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = img[:, :, ::-1]
    cv2.imwrite(img_path, img_bgr)


def create_comparison_grid(
    images: List[Union[np.ndarray, torch.Tensor]],
    labels: Optional[List[str]] = None,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    denormalize: bool = True
) -> None:
    """
    Create a side-by-side comparison grid of images.
    
    Args:
        images: List of images (numpy arrays or tensors)
        labels: Optional list of labels for each image
        titles: Optional list of titles (overrides labels)
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
        denormalize: If True and tensors provided, reverse normalization
    """
    n_images = len(images)
    if n_images == 0:
        return
    
    # Convert tensors to numpy
    img_arrays = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = tensor_to_image(img, denormalize=denormalize)
        img_arrays.append(img)
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    
    display_titles = titles if titles is not None else labels if labels is not None else [f'Image {i+1}' for i in range(n_images)]
    
    for ax, img, title in zip(axes, img_arrays, display_titles):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_alpha_grid(
    content_img: Union[np.ndarray, torch.Tensor],
    style1_img: Union[np.ndarray, torch.Tensor],
    style2_img: Union[np.ndarray, torch.Tensor],
    results: List[Tuple[float, Union[np.ndarray, torch.Tensor]]],  # List of (alpha, result_img)
    method_name: str,
    save_path: Optional[str] = None,
    denormalize: bool = True
) -> None:
    """
    Create a grid showing content, styles, and results across alpha values.
    
    Args:
        content_img: Content image
        style1_img: First style image
        style2_img: Second style image
        results: List of (alpha, result_image) tuples
        method_name: Name of the method (for title)
        save_path: If provided, save figure
        denormalize: If True and tensors provided, reverse normalization
    """
    n_alphas = len(results)
    n_cols = n_alphas + 3  # content + style1 + style2 + results
    n_rows = 1
    
    fig = plt.figure(figsize=(n_cols * 2, 3))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.1, hspace=0.1)
    
    # Convert to numpy if needed
    def to_np(img):
        if isinstance(img, torch.Tensor):
            return tensor_to_image(img, denormalize=denormalize)
        return img
    
    content_np = to_np(content_img)
    style1_np = to_np(style1_img)
    style2_np = to_np(style2_img)
    
    # Content
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(content_np)
    ax.axis('off')
    ax.set_title('Content', fontsize=9)
    
    # Style 1
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(style1_np)
    ax.axis('off')
    ax.set_title('Style 1', fontsize=9)
    
    # Style 2
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(style2_np)
    ax.axis('off')
    ax.set_title('Style 2', fontsize=9)
    
    # Results
    for idx, (alpha, result_img) in enumerate(results):
        ax = fig.add_subplot(gs[0, 3 + idx])
        result_np = to_np(result_img)
        ax.imshow(result_np)
        ax.axis('off')
        ax.set_title(f'α={alpha:.2f}', fontsize=9)
    
    fig.suptitle(f'{method_name} - Alpha Interpolation', fontsize=12, y=0.95)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

