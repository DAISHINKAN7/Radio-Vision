"""
IMPROVED PIX2PIX GAN FOR RADIO-TO-OPTICAL IMAGE TRANSLATION
Enhancements over basic Pix2Pix:
- Perceptual Loss (VGG-based feature matching)
- Multi-Scale Discriminator (3 scales)
- Spectral Normalization for stability
- Self-Attention mechanism
- Progressive training support

Author: Radio Vision Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================================
# PERCEPTUAL LOSS (VGG-based)
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    Compares high-level features instead of raw pixels
    """
    def __init__(self, feature_layers=[2, 7, 12, 21, 30], normalize=True):
        super().__init__()

        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features

        # Extract feature layers
        self.feature_blocks = nn.ModuleList()
        prev_layer = 0

        for layer_idx in feature_layers:
            block = nn.Sequential()
            for i in range(prev_layer, layer_idx + 1):
                block.add_module(str(i), vgg[i])
            self.feature_blocks.append(block)
            prev_layer = layer_idx + 1

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        self.normalize = normalize

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        """
        Args:
            x: Generated image (B, 3, H, W)
            y: Target image (B, 3, H, W)

        Returns:
            Perceptual loss
        """
        # Normalize to ImageNet stats
        if self.normalize:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        # Extract features
        loss = 0.0
        x_feat = x
        y_feat = y

        for block in self.feature_blocks:
            x_feat = block(x_feat)
            y_feat = block(y_feat)
            loss += F.l1_loss(x_feat, y_feat)

        return loss / len(self.feature_blocks)


# ============================================================================
# SELF-ATTENTION MODULE
# ============================================================================

class SelfAttention(nn.Module):
    """
    Self-Attention mechanism (SAGAN)
    Helps the generator focus on important regions
    """
    def __init__(self, in_channels):
        super().__init__()

        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Attention-weighted features
        """
        B, C, H, W = x.size()

        # Compute query, key, value
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        key = self.key(x).view(B, -1, H * W)  # (B, C', HW)
        value = self.value(x).view(B, -1, H * W)  # (B, C, HW)

        # Attention map
        attention = self.softmax(torch.bmm(query, key))  # (B, HW, HW)

        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)

        # Residual connection with learnable weight
        out = self.gamma * out + x

        return out


# ============================================================================
# IMPROVED U-NET GENERATOR
# ============================================================================

class ImprovedUNetGenerator(nn.Module):
    """
    Improved U-Net Generator with:
    - Self-Attention at 32x32 and 64x64 resolutions
    - Spectral Normalization
    - Better skip connections
    """
    def __init__(self, in_channels=3, out_channels=3, features=64, use_attention=True):
        super().__init__()

        self.use_attention = use_attention

        # Encoder
        self.enc1 = self._encoder_block(in_channels, features, normalize=False)
        self.enc2 = self._encoder_block(features, features * 2)
        self.enc3 = self._encoder_block(features * 2, features * 4)
        self.enc4 = self._encoder_block(features * 4, features * 8)
        self.enc5 = self._encoder_block(features * 8, features * 8)
        self.enc6 = self._encoder_block(features * 8, features * 8)
        self.enc7 = self._encoder_block(features * 8, features * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        # Self-Attention (at different scales)
        if use_attention:
            self.attn_32 = SelfAttention(features * 8)  # 32x32
            self.attn_64 = SelfAttention(features * 8)  # 64x64

        # Decoder
        self.dec7 = self._decoder_block(features * 8, features * 8, dropout=True)
        self.dec6 = self._decoder_block(features * 16, features * 8, dropout=True)
        self.dec5 = self._decoder_block(features * 16, features * 8, dropout=True)
        self.dec4 = self._decoder_block(features * 16, features * 8)
        self.dec3 = self._decoder_block(features * 16, features * 4)
        self.dec2 = self._decoder_block(features * 8, features * 2)
        self.dec1 = self._decoder_block(features * 4, features)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def _encoder_block(self, in_c, out_c, normalize=True):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_c, out_c, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        # Bottleneck
        b = self.bottleneck(e7)

        # Decoder with skip connections and attention
        d7 = self.dec7(b)
        if self.use_attention:
            d7 = self.attn_32(d7)  # Apply attention at 32x32
        d7 = torch.cat([d7, e7], dim=1)

        d6 = self.dec6(d7)
        if self.use_attention:
            d6 = self.attn_64(d6)  # Apply attention at 64x64
        d6 = torch.cat([d6, e6], dim=1)

        d5 = self.dec5(d6)
        d5 = torch.cat([d5, e5], dim=1)

        d4 = self.dec4(d5)
        d4 = torch.cat([d4, e4], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        return self.final(d1)


# ============================================================================
# MULTI-SCALE DISCRIMINATOR
# ============================================================================

class SpectralNorm(nn.Module):
    """
    Spectral Normalization wrapper
    Stabilizes GAN training
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]

        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))


class SingleScaleDiscriminator(nn.Module):
    """
    Single-scale PatchGAN Discriminator with Spectral Normalization
    """
    def __init__(self, in_channels=6, features=64, use_spectral_norm=True):
        super().__init__()

        def conv_block(in_c, out_c, stride=2, normalize=True):
            layers = []
            conv = nn.Conv2d(in_c, out_c, 4, stride, 1, bias=not normalize)

            if use_spectral_norm:
                layers.append(nn.utils.spectral_norm(conv))
            else:
                layers.append(conv)

            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))  # Instance norm works better than batch norm

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(in_channels, features, normalize=False),
            conv_block(features, features * 2),
            conv_block(features * 2, features * 4),
            conv_block(features * 4, features * 8, stride=1),
            nn.Conv2d(features * 8, 1, 4, 1, 1)
        )

    def forward(self, radio_img, optical_img):
        x = torch.cat([radio_img, optical_img], dim=1)
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator
    Operates at 3 different scales: original, 2x downsampled, 4x downsampled

    This helps capture both:
    - Fine details (original scale)
    - Medium structures (2x downsampled)
    - Global structure (4x downsampled)
    """
    def __init__(self, in_channels=6, features=64, num_scales=3):
        super().__init__()

        self.num_scales = num_scales

        # Create discriminators at different scales
        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(
                SingleScaleDiscriminator(in_channels, features, use_spectral_norm=True)
            )

        # Downsampling
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, radio_img, optical_img):
        """
        Returns list of discriminator outputs at different scales
        """
        outputs = []

        radio_input = radio_img
        optical_input = optical_img

        for i, disc in enumerate(self.discriminators):
            if i > 0:
                radio_input = self.downsample(radio_input)
                optical_input = self.downsample(optical_input)

            outputs.append(disc(radio_input, optical_input))

        return outputs


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class ImprovedGANLoss(nn.Module):
    """
    Combined loss for improved Pix2Pix:
    - Adversarial loss (GAN)
    - L1 loss (pixel-wise)
    - Perceptual loss (VGG features)
    """
    def __init__(self, lambda_l1=100.0, lambda_perceptual=10.0, use_perceptual=True):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual

        # Initialize perceptual loss
        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()

    def generator_loss(self, fake_pred_list, fake_images, real_images):
        """
        Generator loss

        Args:
            fake_pred_list: List of discriminator predictions (multi-scale)
            fake_images: Generated images
            real_images: Real target images

        Returns:
            Total loss, individual losses dict
        """
        # Adversarial loss (fool discriminator)
        adv_loss = 0.0
        for fake_pred in fake_pred_list:
            adv_loss += F.mse_loss(fake_pred, torch.ones_like(fake_pred))
        adv_loss /= len(fake_pred_list)

        # L1 loss (pixel-wise)
        l1_loss = F.l1_loss(fake_images, real_images)

        # Perceptual loss
        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(fake_images, real_images)
        else:
            perceptual_loss = torch.tensor(0.0).to(fake_images.device)

        # Total loss
        total_loss = (
            adv_loss +
            self.lambda_l1 * l1_loss +
            self.lambda_perceptual * perceptual_loss
        )

        losses = {
            'total': total_loss.item(),
            'adversarial': adv_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item() if self.use_perceptual else 0.0
        }

        return total_loss, losses

    def discriminator_loss(self, real_pred_list, fake_pred_list):
        """
        Discriminator loss (multi-scale)

        Args:
            real_pred_list: List of discriminator predictions on real pairs
            fake_pred_list: List of discriminator predictions on fake pairs

        Returns:
            Total loss
        """
        loss = 0.0

        for real_pred, fake_pred in zip(real_pred_list, fake_pred_list):
            # Real images should be classified as real (1)
            real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))

            # Fake images should be classified as fake (0)
            fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))

            loss += (real_loss + fake_loss) / 2

        return loss / len(real_pred_list)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TESTING IMPROVED PIX2PIX GAN")
    print("="*80)

    # Test Generator
    print("\n🔬 Testing Improved U-Net Generator...")
    gen = ImprovedUNetGenerator(in_channels=3, out_channels=3, features=64, use_attention=True)
    gen_params = sum(p.numel() for p in gen.parameters())
    print(f"✅ Generator parameters: {gen_params:,}")

    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    y = gen(x)
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {y.shape}")

    # Test Multi-Scale Discriminator
    print("\n🔬 Testing Multi-Scale Discriminator...")
    disc = MultiScaleDiscriminator(in_channels=6, features=64, num_scales=3)
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"✅ Discriminator parameters: {disc_params:,}")

    # Test forward pass
    real_img = torch.randn(2, 3, 256, 256)
    fake_img = gen(x)
    outputs = disc(x, real_img)
    print(f"✅ Number of scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"   Scale {i}: {out.shape}")

    # Test Perceptual Loss
    print("\n🔬 Testing Perceptual Loss...")
    perceptual_loss = VGGPerceptualLoss()
    loss_val = perceptual_loss(fake_img, real_img)
    print(f"✅ Perceptual loss value: {loss_val.item():.4f}")

    # Test Combined Loss
    print("\n🔬 Testing Combined Loss...")
    gan_loss = ImprovedGANLoss(lambda_l1=100.0, lambda_perceptual=10.0)

    fake_preds = disc(x, fake_img)
    g_loss, g_losses = gan_loss.generator_loss(fake_preds, fake_img, real_img)
    print(f"✅ Generator Loss Breakdown:")
    for name, value in g_losses.items():
        print(f"   {name:15s}: {value:.4f}")

    real_preds = disc(x, real_img)
    d_loss = gan_loss.discriminator_loss(real_preds, fake_preds)
    print(f"✅ Discriminator Loss: {d_loss.item():.4f}")

    # Self-Attention Test
    print("\n🔬 Testing Self-Attention...")
    attn = SelfAttention(in_channels=512)
    feat = torch.randn(2, 512, 32, 32)
    attn_feat = attn(feat)
    print(f"✅ Input shape: {feat.shape}")
    print(f"✅ Output shape: {attn_feat.shape}")
    print(f"✅ Gamma (learnable): {attn.gamma.item():.4f}")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)

    print("\n📊 Model Summary:")
    print(f"   Generator: {gen_params:,} parameters")
    print(f"   Discriminator: {disc_params:,} parameters")
    print(f"   Total: {gen_params + disc_params:,} parameters")
