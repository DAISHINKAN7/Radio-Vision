"""
SPADE GAN for Radio-to-Optical Image Translation
Uses Spatially-Adaptive Denormalization for high-quality conditional image synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization"""
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
        hidden_nc = 128
        self.shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)
        self.beta = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)

    def forward(self, x, segmap):
        normalized = self.norm(x)
        segmap = F.interpolate(segmap, size=x.shape[2:], mode='bilinear', align_corners=False)
        actv = self.shared(segmap)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        return normalized * (1 + gamma) + beta


class SPADEResBlock(nn.Module):
    """Residual block with SPADE normalization"""
    def __init__(self, in_nc, out_nc, label_nc):
        super().__init__()
        middle_nc = min(in_nc, out_nc)

        self.spade1 = SPADE(in_nc, label_nc)
        self.conv1 = nn.Conv2d(in_nc, middle_nc, 3, padding=1)
        self.spade2 = SPADE(middle_nc, label_nc)
        self.conv2 = nn.Conv2d(middle_nc, out_nc, 3, padding=1)

        self.learned_skip = in_nc != out_nc
        if self.learned_skip:
            self.spade_skip = SPADE(in_nc, label_nc)
            self.conv_skip = nn.Conv2d(in_nc, out_nc, 1, bias=False)

    def forward(self, x, segmap):
        skip = x
        if self.learned_skip:
            skip = self.conv_skip(self.spade_skip(x, segmap))

        x = self.conv1(F.leaky_relu(self.spade1(x, segmap), 0.2))
        x = self.conv2(F.leaky_relu(self.spade2(x, segmap), 0.2))
        return x + skip


class SPADEGenerator(nn.Module):
    """SPADE Generator - outputs 256x256 images"""
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super().__init__()
        self.ngf = ngf

        # Initial projection from noise
        self.fc = nn.Linear(256, 16 * ngf * 4 * 4)

        # SPADE ResBlocks with upsampling: 4->8->16->32->64->128->256
        self.head = SPADEResBlock(16 * ngf, 16 * ngf, input_nc)
        self.up1 = SPADEResBlock(16 * ngf, 16 * ngf, input_nc)  # 8x8
        self.up2 = SPADEResBlock(16 * ngf, 8 * ngf, input_nc)   # 16x16
        self.up3 = SPADEResBlock(8 * ngf, 4 * ngf, input_nc)    # 32x32
        self.up4 = SPADEResBlock(4 * ngf, 2 * ngf, input_nc)    # 64x64
        self.up5 = SPADEResBlock(2 * ngf, ngf, input_nc)        # 128x128
        self.up6 = SPADEResBlock(ngf, ngf // 2, input_nc)       # 256x256

        self.conv_out = nn.Conv2d(ngf // 2, output_nc, 3, padding=1)

    def forward(self, segmap):
        batch_size = segmap.size(0)
        z = torch.randn(batch_size, 256, device=segmap.device)

        x = self.fc(z).view(batch_size, 16 * self.ngf, 4, 4)

        x = self.head(x, segmap)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(x, segmap)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(x, segmap)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up3(x, segmap)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up4(x, segmap)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up5(x, segmap)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up6(x, segmap)

        x = self.conv_out(F.leaky_relu(x, 0.2))
        return torch.tanh(x)


class ConditionalDiscriminator(nn.Module):
    """Multi-scale PatchGAN Discriminator"""
    def __init__(self, input_nc=6, ndf=64, n_layers=4):
        super().__init__()

        self.scale1 = self._make_discriminator(input_nc, ndf, n_layers)
        self.scale2 = self._make_discriminator(input_nc, ndf, n_layers)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def _make_discriminator(self, input_nc, ndf, n_layers):
        layers = [
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, stride=2, padding=1),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers.append(nn.Conv2d(nf, 1, 4, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, radio, optical):
        x = torch.cat([radio, optical], dim=1)

        out1 = self.scale1(x)
        x_down = self.downsample(x)
        out2 = self.scale2(x_down)

        return [out1, out2]


class VGGFeatureLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(2)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(2, 7)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(7, 12)])
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(12, 21)])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x1, y1 = self.slice1(x), self.slice1(y)
        x2, y2 = self.slice2(x1), self.slice2(y1)
        x3, y3 = self.slice3(x2), self.slice3(y2)
        x4, y4 = self.slice4(x3), self.slice4(y3)

        loss = F.l1_loss(x1, y1) + F.l1_loss(x2, y2) + F.l1_loss(x3, y3) + F.l1_loss(x4, y4)
        return loss


class SPADEGANLoss(nn.Module):
    """Combined losses for SPADE GAN training"""
    def __init__(self, device='cuda'):
        super().__init__()
        self.vgg_loss = VGGFeatureLoss().to(device)
        self.l1_weight = 10.0
        self.vgg_weight = 10.0
        self.fm_weight = 10.0

    def generator_loss(self, fake_preds, real_feats=None, fake_feats=None,
                       fake_img=None, real_img=None):
        # Hinge loss for generator
        g_loss = 0
        for pred in fake_preds:
            g_loss += -pred.mean()
        g_loss /= len(fake_preds)

        total_loss = g_loss

        # L1 reconstruction loss
        if fake_img is not None and real_img is not None:
            l1_loss = F.l1_loss(fake_img, real_img)
            total_loss += self.l1_weight * l1_loss

            # VGG perceptual loss
            vgg_loss = self.vgg_loss(fake_img, real_img)
            total_loss += self.vgg_weight * vgg_loss

        return total_loss

    def discriminator_loss(self, real_preds, fake_preds):
        # Hinge loss with label smoothing
        d_loss = 0
        for real_pred, fake_pred in zip(real_preds, fake_preds):
            d_loss += F.relu(0.9 - real_pred).mean()  # Label smoothing
            d_loss += F.relu(0.9 + fake_pred).mean()
        return d_loss / len(real_preds)
