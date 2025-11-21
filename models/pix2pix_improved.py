"""
IMPROVED PIX2PIX GAN FOR RADIO-TO-OPTICAL IMAGE TRANSLATION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[2, 7, 12, 21, 30], normalize=True):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.feature_blocks = nn.ModuleList()
        prev_layer = 0
        for layer_idx in feature_layers:
            block = nn.Sequential()
            for i in range(prev_layer, layer_idx + 1):
                block.add_module(str(i), vgg[i])
            self.feature_blocks.append(block)
            prev_layer = layer_idx + 1

        for param in self.parameters():
            param.requires_grad = False

        self.normalize = normalize
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # Clone inputs to avoid inplace issues
        x = x.clone()
        y = y.clone()

        if self.normalize:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        loss = 0.0
        x_feat = x
        y_feat = y

        for block in self.feature_blocks:
            x_feat = block(x_feat)
            y_feat = block(y_feat)
            loss = loss + F.l1_loss(x_feat, y_feat)

        return loss / len(self.feature_blocks)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        x_input = x.clone()

        query = self.query(x_input).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x_input).view(B, -1, H * W)
        value = self.value(x_input).view(B, -1, H * W)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + x_input


class ImprovedUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        self.enc1 = self._encoder_block(in_channels, features, normalize=False)
        self.enc2 = self._encoder_block(features, features * 2)
        self.enc3 = self._encoder_block(features * 2, features * 4)
        self.enc4 = self._encoder_block(features * 4, features * 8)
        self.enc5 = self._encoder_block(features * 8, features * 8)
        self.enc6 = self._encoder_block(features * 8, features * 8)
        self.enc7 = self._encoder_block(features * 8, features * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU()
        )

        if use_attention:
            self.attn_32 = SelfAttention(features * 8)
            self.attn_64 = SelfAttention(features * 8)

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
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_c, out_c, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        b = self.bottleneck(e7)

        d7 = self.dec7(b)
        if self.use_attention:
            d7 = self.attn_32(d7)
        d7 = torch.cat([d7, e7.clone()], dim=1)

        d6 = self.dec6(d7)
        if self.use_attention:
            d6 = self.attn_64(d6)
        d6 = torch.cat([d6, e6.clone()], dim=1)

        d5 = self.dec5(d6)
        d5 = torch.cat([d5, e5.clone()], dim=1)

        d4 = self.dec4(d5)
        d4 = torch.cat([d4, e4.clone()], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3.clone()], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2.clone()], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1.clone()], dim=1)

        return self.final(d1)


class SingleScaleDiscriminator(nn.Module):
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
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
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
    def __init__(self, in_channels=6, features=64, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(
                SingleScaleDiscriminator(in_channels, features, use_spectral_norm=True)
            )
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, radio_img, optical_img):
        outputs = []
        radio_input = radio_img
        optical_input = optical_img

        for i, disc in enumerate(self.discriminators):
            if i > 0:
                radio_input = self.downsample(radio_input)
                optical_input = self.downsample(optical_input)
            outputs.append(disc(radio_input, optical_input))
        return outputs


class ImprovedGANLoss(nn.Module):
    def __init__(self, lambda_l1=100.0, lambda_perceptual=10.0, use_perceptual=True):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()

    def generator_loss(self, fake_pred_list, fake_images, real_images):
        adv_loss = 0.0
        for fake_pred in fake_pred_list:
            adv_loss = adv_loss + F.mse_loss(fake_pred, torch.ones_like(fake_pred))
        adv_loss = adv_loss / len(fake_pred_list)

        l1_loss = F.l1_loss(fake_images, real_images)

        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(fake_images.clone(), real_images.clone())
        else:
            perceptual_loss = torch.tensor(0.0).to(fake_images.device)

        total_loss = adv_loss + self.lambda_l1 * l1_loss + self.lambda_perceptual * perceptual_loss

        losses = {
            'total': total_loss.item(),
            'adversarial': adv_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item() if self.use_perceptual else 0.0
        }
        return total_loss, losses

    def discriminator_loss(self, real_pred_list, fake_pred_list):
        loss = 0.0
        for real_pred, fake_pred in zip(real_pred_list, fake_pred_list):
            real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
            fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
            loss = loss + (real_loss + fake_loss) / 2
        return loss / len(real_pred_list)
