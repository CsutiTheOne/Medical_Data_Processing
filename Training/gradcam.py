"""
Grad-CAM visualization for model interpretability
Uses pytorch-grad-cam library for robust implementations
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import List, Callable, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    PYTORCH_GRAD_CAM_AVAILABLE = True
except ImportError:
    PYTORCH_GRAD_CAM_AVAILABLE = False
    print("pytorch-grad-cam not installed. Using fallback implementation.")


class SimpleGradCAM:
    """
    Fallback Grad-CAM implementation if pytorch-grad-cam is not available
    """
    def __init__(self, model, target_layers: List):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.activations = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())
        
        for layer in self.target_layers:
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM for input"""
        self.model.eval()
        self.gradients = []
        self.activations = []
        
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward(retain_graph=True)
        
        if not self.gradients or not self.activations:
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Get the last activation and gradient
        activations = self.activations[-1][0].cpu().numpy()  # (C, H, W)
        gradients = self.gradients[-1][0].cpu().numpy()      # (C, H, W)
        
        # Compute weights
        weights = gradients.mean(axis=(1, 2))  # (C,)
        
        # Weighted combination
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
        cam = np.maximum(cam, 0)  # ReLU
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))
        
        return cam


def get_gradcam_for_swin(model: torch.nn.Module, input_tensor: torch.Tensor, 
                         target_class: Optional[int] = None, 
                         use_hi_res: bool = False) -> np.ndarray:
    """
    Generate Grad-CAM for Swin Transformer model
    
    Args:
        model: Swin Transformer model
        input_tensor: Input image (B, C, H, W)
        target_class: Target class for CAM (if None, uses argmax)
        use_hi_res: Use HiResCAM for better spatial resolution
    
    Returns:
        Grad-CAM heatmap (H, W) in range [0, 1]
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    model.eval()
    
    # Get target layer - the last residual block in features
    target_layer = model.features[-1][-1]
    
    if PYTORCH_GRAD_CAM_AVAILABLE:
        try:
            cam_class = HiResCAM if use_hi_res else GradCAM
            cam = cam_class(model=model, target_layers=[target_layer])
            
            # Get target class
            if target_class is None:
                with torch.no_grad():
                    output = model(input_tensor)
                    target_class = output.argmax(dim=1).item()
            
            targets = [ClassifierOutputTarget(target_class)]
            cam_map = cam(input_tensor=input_tensor, targets=targets)
            
            return cam_map[0]
        except Exception as e:
            print(f"pytorch-grad-cam error: {e}. Falling back to simple implementation.")
    
    # Fallback implementation
    simple_cam = SimpleGradCAM(model, [target_layer])
    return simple_cam.generate_cam(input_tensor, target_class)


def visualize_gradcam_on_image(
    image: torch.Tensor,
    cam: np.ndarray,
    title: str = "Grad-CAM Visualization",
    alpha: float = 0.5,
    cmap: str = 'jet'
) -> plt.Figure:
    """
    Visualize Grad-CAM heatmap overlaid on the original image
    
    Args:
        image: Input image tensor (C, H, W), normalized in [0, 1] or [-1, 1]
        cam: Grad-CAM heatmap (H, W) in range [0, 1]
        title: Plot title
        alpha: Overlay transparency (0-1)
        cmap: Colormap name for heatmap
    
    Returns:
        Matplotlib figure
    """
    # Convert image to numpy and handle dimensions
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().detach().numpy()
    else:
        image_np = image.copy()
    
    # Handle channel-first format
    if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Normalize to [0, 1]
    if image_np.max() > 1:
        image_np = np.clip(image_np, -1, 1)
        image_np = (image_np + 1) / 2
    else:
        image_np = np.clip(image_np, 0, 1)
    
    # Handle grayscale
    if image_np.ndim == 2 or image_np.shape[2] == 1:
        image_np = np.repeat(image_np[..., np.newaxis] if image_np.ndim == 2 else image_np, 3, axis=2)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Original
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im1 = axes[1].imshow(cam, cmap=cmap)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    heatmap_3channel = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_3channel = cv2.cvtColor(heatmap_3channel, cv2.COLOR_BGR2RGB) / 255.0
    
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_3channel, alpha, 0)
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (α={alpha})', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def visualize_batch_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    device: torch.device = torch.device('cpu'),
    num_samples: int = 4
) -> plt.Figure:
    """
    Visualize Grad-CAM for multiple images in a grid
    
    Args:
        model: Neural network model
        images: Batch of images (B, C, H, W)
        labels: Ground truth labels (B,)
        class_names: List of class names
        device: Device to run on
        num_samples: Number of samples to visualize
    
    Returns:
        Matplotlib figure with grid of visualizations
    """
    model.eval()
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        img = images[i:i+1].to(device)
        label = labels[i].item()
        
        # Get prediction
        with torch.no_grad():
            pred = model(img).argmax(dim=1).item()
        
        # Generate Grad-CAM
        cam = get_gradcam_for_swin(model, img, target_class=label)
        
        # Get image for visualization
        img_np = img[0].cpu().numpy()
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = np.clip(img_np, 0, 1)
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        
        # Plot
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Image {i}', fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cam, cmap='jet')
        axes[i, 1].set_title('Grad-CAM', fontweight='bold')
        axes[i, 1].axis('off')
        
        heatmap_3ch = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_3ch = cv2.cvtColor(heatmap_3ch, cv2.COLOR_BGR2RGB) / 255.0
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_3ch, 0.4, 0)
        axes[i, 2].imshow(overlay)
        
        class_name = class_names[label] if class_names else f"Class {label}"
        pred_name = class_names[pred] if class_names else f"Class {pred}"
        status = "✓" if label == pred else "✗"
        axes[i, 2].set_title(f'{status} True: {class_name}\nPred: {pred_name}', fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig
