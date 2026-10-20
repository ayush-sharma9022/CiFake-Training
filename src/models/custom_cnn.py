"""
Custom Deepfake Detection Convolutional Neural Network (CustomDeepfakeCNN).
Designed and trained from scratch with random weight initialization (Kaiming Normal).
NO pretrained architectures, NO transfer learning, NO external model weights.
Features residual gradient propagation for stable, high-accuracy artefact learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """
    Custom Dual-Convolution Residual Feature Block.
    Combines two 3x3 convolutions with Batch Normalization, LeakyReLU,
    and a projection shortcut to preserve high-frequency forensic artefacts.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act2(x + res)
        return x


class CustomDeepfakeCNN(nn.Module):
    """
    Custom Deep Learning Architecture for Image Deepfake Detection.
    Architecture:
    1. Input layer (3, 128, 128)
    2. Stem Conv Layer (3 -> 32 channels)
    3. Block 1: Spatial Artefact & High-Frequency Texture Capture (32 -> 64, stride=2)
    4. Block 2: Boundary & Edge Inconsistency Extraction (64 -> 128, stride=2)
    5. Block 3: Blending Inconsistencies & Facial Feature Symmetry (128 -> 256, stride=2)
    6. Block 4: Chromatic Aberration & Illumination Analysis (256 -> 512, stride=2)
    7. Global Average Pooling (512, 1, 1)
    8. Fully Connected Regularized Classifier Head (512 -> 128 -> 1)
    """
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem Conv: Extract initial low-level pixel cues
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # 4 Residual Artefact Extraction Stages
        self.block1 = ResidualConvBlock(32, 64, stride=2)    # -> 64x64
        self.block2 = ResidualConvBlock(64, 128, stride=2)   # -> 32x32
        self.block3 = ResidualConvBlock(128, 256, stride=2)  # -> 16x16
        self.block4 = ResidualConvBlock(256, 512, stride=2)  # -> 8x8

        # Global Average Pooling for spatial collapse
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

        # Random Weight Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming / He normal initialization for convolutional layers.
        Xavier normal for hidden dense layers.
        Small Xavier uniform gain for output projection to center initial logits near zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Scale final classification layer so initial predictions are balanced (prob ~ 0.5)
        nn.init.xavier_uniform_(self.classifier[-1].weight, gain=0.05)
        if self.classifier[-1].bias is not None:
            nn.init.constant_(self.classifier[-1].bias, 0.0)

    def extract_features(self, x):
        """Extract intermediate 512-dimensional embedding vector."""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        """Forward pass returning raw logits (for BCEWithLogitsLoss)."""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # Target layer for Grad-CAM
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x):
        """Inference helper returning probability in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


if __name__ == "__main__":
    model = CustomDeepfakeCNN()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    features = model.extract_features(x)
    print("CustomDeepfakeCNN Initialized Successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {out.shape}")
    print(f"Features shape: {features.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
