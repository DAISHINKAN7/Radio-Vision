"""
ConvNeXt-V2 Classifier for optical images
Using pretrained weights from timm library
"""

import torch
import torch.nn as nn
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Install with: pip install timm")


class ConvNeXtV2Classifier(nn.Module):
    """ConvNeXt-V2 classifier with pretrained weights"""
    def __init__(self, num_classes=4, model_size='base', pretrained=True, dropout=0.5):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm library required. Install with: pip install timm")

        # Available sizes: tiny, base, large
        model_name = f'convnextv2_{model_size}.fcmae_ft_in22k_in1k'

        if pretrained:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        else:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class ConvNeXtV2EnsembleClassifier(nn.Module):
    """Ensemble of multiple ConvNeXt-V2 models"""
    def __init__(self, num_classes=4, sizes=['tiny', 'base'], pretrained=True):
        super().__init__()
        self.models = nn.ModuleList([
            ConvNeXtV2Classifier(num_classes, size, pretrained)
            for size in sizes
        ])

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)
