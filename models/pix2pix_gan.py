"""
Pix2Pix GAN for Radio to Optical Image Translation
Production-grade implementation
"""

import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """U-Net Generator with skip connections"""
    
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        
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
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        
        b = self.bottleneck(e7)
        
        d7 = torch.cat([self.dec7(b), e7], dim=1)
        d6 = torch.cat([self.dec6(d7), e6], dim=1)
        d5 = torch.cat([self.dec5(d6), e5], dim=1)
        d4 = torch.cat([self.dec4(d5), e4], dim=1)
        d3 = torch.cat([self.dec3(d4), e3], dim=1)
        d2 = torch.cat([self.dec2(d3), e2], dim=1)
        d1 = torch.cat([self.dec1(d2), e1], dim=1)
        
        return self.final(d1)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator"""
    
    def __init__(self, in_channels=6, features=64):
        super(PatchGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features * 4, features * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features * 8, 1, 4, 1, 1)
        )
    
    def forward(self, radio_img, optical_img):
        x = torch.cat([radio_img, optical_img], dim=1)
        return self.model(x)


if __name__ == "__main__":
    print("Testing Pix2Pix...")
    gen = UNetGenerator()
    disc = PatchGANDiscriminator()
    print(f"✅ Generator: {sum(p.numel() for p in gen.parameters()):,} params")
    print(f"✅ Discriminator: {sum(p.numel() for p in disc.parameters()):,} params")
    
    x = torch.randn(2, 3, 256, 256)
    y = gen(x)
    print(f"✅ Output shape: {y.shape}")
    
    d_out = disc(x, y)
    print(f"✅ Discriminator output: {d_out.shape}")
    print("🎉 All tests passed!")