"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) Implementation.
Computes gradient-based explainability heatmaps for the custom deepfake CNN,
visualizing specific facial regions and visual artefacts influencing predictions.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM Explainer for CustomDeepfakeCNN.
    Hooks into the final convolutional layer of block5 to record feature maps
    and backward gradients with respect to the output classification score.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default to the final Conv2d layer in block4
            self.target_layer = self.model.block4.conv2
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate raw Grad-CAM heatmap for a single image tensor of shape (1, 3, H, W).
        Returns a 2D numpy array normalized in [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class is None:
            score = logits[0, 0]
        else:
            score = logits[0, 0] if target_class == 1 else -logits[0, 0]

        score.backward(retain_graph=True)

        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')

        # Global average pool the gradients across spatial dimensions
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Discard negative influences

        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay the Grad-CAM heatmap onto the original RGB image.
        Args:
            original_image: numpy array (H, W, 3) in uint8 [0, 255] or PIL Image
            heatmap: 2D numpy array (H', W') in float [0, 1]
            alpha: blending ratio
        Returns:
            heatmap_colored: colored heatmap RGB (H, W, 3)
            overlay: blended image RGB (H, W, 3)
        """
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply OpenCV colormap (outputs BGR)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Ensure original is RGB
        if original_image.ndim == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        # Alpha blending
        overlay = cv2.addWeighted(heatmap_rgb, alpha, original_image, 1 - alpha, 0)

        return heatmap_rgb, overlay


if __name__ == "__main__":
    from src.models.custom_cnn import CustomDeepfakeCNN
    model = CustomDeepfakeCNN()
    gradcam = GradCAM(model)
    dummy_input = torch.randn(1, 3, 128, 128, requires_grad=True)
    cam = gradcam.generate_heatmap(dummy_input)
    dummy_orig = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    hm_color, overlay = gradcam.overlay_heatmap(dummy_orig, cam)
    print("GradCAM initialized and tested successfully!")
    print(f"Heatmap shape: {cam.shape}, range: [{cam.min():.2f}, {cam.max():.2f}]")
    print(f"Overlay shape: {overlay.shape}, dtype: {overlay.dtype}")
