"""
Multi-modal Fusion Model
Combines radio, optical, and signal features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class FeatureExtractor(nn.Module):
    """Extract features from a backbone and return fixed-size embedding"""
    def __init__(self, backbone, embed_dim=512):
        super().__init__()
        self.backbone = backbone

        # Get backbone output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            if len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            in_features = features.shape[1]

        self.projection = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.projection(features)


class MultiModalFusionClassifier(nn.Module):
    """
    Multi-modal fusion classifier
    Combines features from radio images, optical images, and optionally signal data
    """
    def __init__(self, num_classes=4, embed_dim=512, fusion_hidden=512, dropout=0.5):
        super().__init__()

        # Radio branch - MobileNetV2
        radio_backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        radio_backbone = nn.Sequential(*list(radio_backbone.children())[:-1])
        self.radio_extractor = FeatureExtractor(radio_backbone, embed_dim)

        # Optical branch - ConvNeXt or ResNet
        if TIMM_AVAILABLE:
            optical_backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0)
        else:
            optical_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            optical_backbone = nn.Sequential(*list(optical_backbone.children())[:-1])
        self.optical_extractor = FeatureExtractor(optical_backbone, embed_dim)

        # Fusion layers
        total_features = 2 * embed_dim  # radio + optical
        self.fusion = nn.Sequential(
            nn.Linear(total_features, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, num_classes)
        )

    def forward(self, radio_img, optical_img):
        """
        Args:
            radio_img: Radio image tensor [B, 3, H, W]
            optical_img: Optical image tensor [B, 3, H, W]
        """
        radio_features = self.radio_extractor(radio_img)
        optical_features = self.optical_extractor(optical_img)

        # Concatenate all features
        combined = torch.cat([radio_features, optical_features], dim=1)

        # Classification
        return self.fusion(combined)


class EarlyFusionClassifier(nn.Module):
    """Simple early fusion - concatenate images in channel dimension"""
    def __init__(self, num_classes=4):
        super().__init__()

        # Use a strong backbone for 6-channel input
        backbone = models.resnet50(weights=None)
        # Modify first conv to accept 6 channels (3 radio + 3 optical)
        backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, radio_img, optical_img):
        # Concatenate in channel dimension
        combined = torch.cat([radio_img, optical_img], dim=1)
        features = self.backbone(combined)
        return self.classifier(features)


class AttentionFusionClassifier(nn.Module):
    """Cross-attention based fusion"""
    def __init__(self, num_classes=4, embed_dim=512, num_heads=8):
        super().__init__()

        # Feature extractors
        radio_backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        radio_backbone = nn.Sequential(*list(radio_backbone.children())[:-1])
        self.radio_extractor = FeatureExtractor(radio_backbone, embed_dim)

        if TIMM_AVAILABLE:
            optical_backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0)
        else:
            optical_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            optical_backbone = nn.Sequential(*list(optical_backbone.children())[:-1])
        self.optical_extractor = FeatureExtractor(optical_backbone, embed_dim)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, radio_img, optical_img):
        radio_features = self.radio_extractor(radio_img).unsqueeze(0)  # [1, B, D]
        optical_features = self.optical_extractor(optical_img).unsqueeze(0)  # [1, B, D]

        # Cross-attention: radio attends to optical
        attended_radio, _ = self.cross_attention(radio_features, optical_features, optical_features)
        attended_radio = self.norm1(attended_radio + radio_features)

        # Cross-attention: optical attends to radio
        attended_optical, _ = self.cross_attention(optical_features, radio_features, radio_features)
        attended_optical = self.norm2(attended_optical + optical_features)

        # Combine and classify
        combined = torch.cat([attended_radio.squeeze(0), attended_optical.squeeze(0)], dim=1)
        return self.classifier(combined)
