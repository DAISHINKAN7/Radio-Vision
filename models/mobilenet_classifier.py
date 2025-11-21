"""
MOBILENET V2 CLASSIFIER - OPTIMIZED FOR SMALL DATASETS
Lightweight model with 3.5M parameters (vs 22M EfficientNet)
Designed for real-world radio astronomy data (576 samples)

Author: Radio Vision Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class MobileNetV2Classifier(nn.Module):
    """
    Lightweight classifier based on MobileNetV2
    - 3.5M parameters (16% of EfficientNet-B0)
    - Optimized for small datasets
    - High dropout for regularization
    """

    def __init__(self, num_classes=4, pretrained=True, dropout=0.7, in_channels=3):
        super().__init__()

        # Load MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(pretrained=pretrained)

        # Only modify first conv if using single-channel input
        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )

        # Get feature dimension
        feature_dim = 1280

        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier head weights"""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze backbone for transfer learning"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def get_num_params(self):
        """Get number of parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: https://arxiv.org/abs/1708.02002

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing for regularization
    Prevents overconfident predictions
    """

    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C) logits
            target: (N,) class indices
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def create_class_weights(dataset_path):
    """
    Calculate class weights for imbalanced datasets

    Args:
        dataset_path: Path to dataset with metadata.json

    Returns:
        torch.Tensor: Class weights
    """
    import json
    from pathlib import Path
    from collections import Counter

    metadata_path = Path(dataset_path) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Count samples per class
    class_counts = Counter([item['object_type'] for item in metadata])

    # Map to standard order
    class_order = ['spiral_galaxy', 'emission_nebula', 'quasar', 'pulsar']
    counts = [class_counts.get(cls, 1) for cls in class_order]

    # Calculate weights (inverse frequency)
    total = sum(counts)
    weights = [total / (len(class_order) * count) for count in counts]

    print(f"\n📊 Class Distribution:")
    for cls, count, weight in zip(class_order, counts, weights):
        print(f"   {cls:20s}: {count:4d} samples (weight: {weight:.3f})")

    return torch.FloatTensor(weights)


def preprocess_signal_v2(signal, target_size=(224, 224)):
    """
    Enhanced signal preprocessing for MobileNet

    Args:
        signal: numpy array (H, W)
        target_size: tuple (H, W)

    Returns:
        torch.Tensor: Preprocessed signal (1, 1, H, W)
    """
    from scipy.ndimage import zoom

    # Handle NaN values
    signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=0.0)

    # Normalize to [0, 1]
    if signal.max() > signal.min():
        signal = (signal - signal.min()) / (signal.max() - signal.min())
    else:
        signal = np.zeros_like(signal)

    # Resize to target size
    h, w = signal.shape
    target_h, target_w = target_size

    zoom_h = target_h / h
    zoom_w = target_w / w

    signal_resized = zoom(signal, (zoom_h, zoom_w), order=1)
    signal_resized = signal_resized[:target_h, :target_w]

    # Convert to tensor
    signal_tensor = torch.from_numpy(signal_resized).float()
    signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    return signal_tensor


if __name__ == "__main__":
    # Test the model
    print("="*60)
    print("TESTING MOBILENET V2 CLASSIFIER")
    print("="*60)

    model = MobileNetV2Classifier(num_classes=4, pretrained=True)
    params = model.get_num_params()

    print(f"\n✅ Model created successfully!")
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model(dummy_input)

    print(f"\n✅ Forward pass successful!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")

    # Test Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    targets = torch.tensor([0, 1])
    loss = focal_loss(output, targets)

    print(f"\n✅ Focal Loss computed!")
    print(f"   Loss value: {loss.item():.4f}")

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED!")
    print("="*60)
