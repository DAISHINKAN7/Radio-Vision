"""
ADVANCED CLASSIFIER V2 - IMPROVED ARCHITECTURE
Better performance for radio signal classification

Features:
- EfficientNet backbone
- Attention mechanisms
- Multi-scale feature extraction
- Better regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class SelfAttention(nn.Module):
    """Self-attention module for feature enhancement"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Query, Key, Value
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        # Attention map
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Different kernel sizes for multi-scale
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, padding=0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 5, padding=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, 1, padding=0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class AdvancedClassifierV2(nn.Module):
    """
    Advanced Signal Classifier V2
    
    Improvements:
    - EfficientNet-B0 backbone
    - Self-attention mechanism
    - Multi-scale feature extraction
    - Stronger regularization
    - Better architecture for small datasets
    """
    
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        super().__init__()
        
        self.num_classes = num_classes
        self.class_names = ['spiral_galaxy', 'emission_nebula', 'quasar', 'pulsar']
        
        # EfficientNet-B0 backbone (lighter than ResNet, better for small data)
        efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Remove the classifier
        self.features = nn.Sequential(*list(efficientnet.children())[:-2])
        
        # Get feature dimension
        self.feature_dim = 1280  # EfficientNet-B0 output channels
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleBlock(self.feature_dim, self.feature_dim)
        
        # Self-attention
        self.attention = SelfAttention(self.feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Additional feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),  # *2 because we concat avg and max pool
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in [self.feature_processor, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            return_features: If True, return features before classification
        
        Returns:
            logits or (logits, features)
        """
        
        # Feature extraction
        features = self.features(x)
        
        # Multi-scale features
        features = self.multi_scale(features)
        
        # Self-attention
        features = self.attention(features)
        
        # Global pooling (both avg and max)
        avg_pool = self.global_pool(features).flatten(1)
        max_pool = F.adaptive_max_pool2d(features, 1).flatten(1)
        
        # Concatenate
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Feature processing
        processed_features = self.feature_processor(pooled)
        
        # Classification
        logits = self.classifier(processed_features)
        
        if return_features:
            return logits, processed_features
        
        return logits
    
    def predict(self, x):
        """
        Predict class for input
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            predictions, probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs
    
    def get_class_name(self, class_idx):
        """Get class name from index"""
        return self.class_names[class_idx]


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple classifiers for better accuracy
    """
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.num_classes = num_classes
        self.class_names = ['spiral_galaxy', 'emission_nebula', 'quasar', 'pulsar']
        
        # Multiple models with different architectures
        self.model1 = AdvancedClassifierV2(num_classes=num_classes)
        
        # ResNet variant
        resnet = models.resnet34(pretrained=True)
        self.model2_features = nn.Sequential(*list(resnet.children())[:-1])
        self.model2_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Ensemble prediction"""
        
        # Model 1 (Advanced V2)
        logits1 = self.model1(x)
        
        # Model 2 (ResNet)
        features2 = self.model2_features(x)
        logits2 = self.model2_classifier(features2)
        
        # Average logits
        logits = (logits1 + logits2) / 2
        
        return logits
    
    def predict(self, x):
        """Predict with ensemble"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs
    
    def get_class_name(self, class_idx):
        """Get class name from index"""
        return self.class_names[class_idx]


def preprocess_signal_v2(signal, target_size=(224, 224)):
    """
    Preprocess signal for AdvancedClassifierV2
    
    Args:
        signal: numpy array (128, 1024) or (H, W)
        target_size: Target image size (H, W)
    
    Returns:
        torch.Tensor (1, 3, H, W)
    """
    
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    
    # Normalize
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    
    # Convert to 3-channel image
    from PIL import Image
    signal_uint8 = (signal * 255).astype(np.uint8)
    img = Image.fromarray(signal_uint8)
    img = img.resize(target_size, Image.LANCZOS)
    img_rgb = img.convert('RGB')
    
    # To tensor
    img_array = np.array(img_rgb).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_array - mean) / std
    
    # To torch (HWC -> CHW)
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor


# Test function
if __name__ == '__main__':
    print("Testing AdvancedClassifierV2...")
    
    model = AdvancedClassifierV2(num_classes=4)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test prediction
    preds, probs = model.predict(x)
    print(f"Predictions: {preds}")
    print(f"Probabilities shape: {probs.shape}")
    
    print("\nTesting EnsembleClassifier...")
    ensemble = EnsembleClassifier(num_classes=4)
    print(f"Ensemble created with {sum(p.numel() for p in ensemble.parameters()):,} parameters")
    
    logits = ensemble(x)
    print(f"Ensemble output shape: {logits.shape}")
    
    print("\n✅ All tests passed!")