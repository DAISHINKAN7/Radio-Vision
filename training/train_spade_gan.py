"""
SPADE GAN Training Script with comprehensive metrics tracking
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spade_gan import SPADEGenerator, ConditionalDiscriminator, SPADEGANLoss


class PairedDataset(Dataset):
    """Dataset for paired radio-optical images"""
    def __init__(self, radio_dir, optical_dir, transform=None, img_size=256):
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Get sorted file lists
        radio_files = sorted([f for f in os.listdir(radio_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        optical_files = sorted([f for f in os.listdir(optical_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Match by index
        num_pairs = min(len(radio_files), len(optical_files))
        self.pairs = [
            (os.path.join(radio_dir, radio_files[i]),
             os.path.join(optical_dir, optical_files[i]))
            for i in range(num_pairs)
        ]
        print(f"Found {len(self.pairs)} image pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        radio_path, optical_path = self.pairs[idx]
        radio = Image.open(radio_path).convert('RGB')
        optical = Image.open(optical_path).convert('RGB')
        return self.transform(radio), self.transform(optical)


class GANMetricsTracker:
    """Track all training metrics"""
    def __init__(self):
        self.metrics = {
            'g_losses': [], 'd_losses': [],
            'psnr': [], 'ssim': [],
            'g_losses_val': [], 'd_losses_val': [],
            'psnr_val': [], 'ssim_val': []
        }

    def update(self, g_loss, d_loss, psnr_val=None, ssim_val=None, val=False):
        prefix = '_val' if val else ''
        self.metrics[f'g_losses{prefix}'].append(g_loss)
        self.metrics[f'd_losses{prefix}'].append(d_loss)
        if psnr_val is not None:
            self.metrics[f'psnr{prefix}'].append(psnr_val)
        if ssim_val is not None:
            self.metrics[f'ssim{prefix}'].append(ssim_val)

    def save_plots(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        # Loss curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(self.metrics['g_losses'], label='Train G')
        axes[0, 0].plot(self.metrics['d_losses'], label='Train D')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()

        if self.metrics['psnr']:
            axes[0, 1].plot(self.metrics['psnr'], label='Train')
            if self.metrics['psnr_val']:
                axes[0, 1].plot(self.metrics['psnr_val'], label='Val')
            axes[0, 1].set_title('PSNR')
            axes[0, 1].legend()

        if self.metrics['ssim']:
            axes[1, 0].plot(self.metrics['ssim'], label='Train')
            if self.metrics['ssim_val']:
                axes[1, 0].plot(self.metrics['ssim_val'], label='Val')
            axes[1, 0].set_title('SSIM')
            axes[1, 0].legend()

        # Loss ratio
        if self.metrics['g_losses'] and self.metrics['d_losses']:
            ratio = [g / (d + 1e-8) for g, d in zip(self.metrics['g_losses'], self.metrics['d_losses'])]
            axes[1, 1].plot(ratio)
            axes[1, 1].set_title('G/D Loss Ratio')
            axes[1, 1].axhline(y=1, color='r', linestyle='--')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.close()

        # Save metrics as JSON
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)


def compute_image_metrics(fake, real):
    """Compute PSNR and SSIM"""
    fake_np = ((fake.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    real_np = ((real.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

    psnr_vals, ssim_vals = [], []
    for i in range(fake_np.shape[0]):
        fake_img = np.transpose(fake_np[i], (1, 2, 0))
        real_img = np.transpose(real_np[i], (1, 2, 0))
        psnr_vals.append(psnr(real_img, fake_img))
        ssim_vals.append(ssim(real_img, fake_img, channel_axis=2, data_range=255))

    return np.mean(psnr_vals), np.mean(ssim_vals)


def save_samples(generator, dataloader, save_dir, epoch, device):
    """Save sample generations"""
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()

    with torch.no_grad():
        radio, optical = next(iter(dataloader))
        radio, optical = radio[:4].to(device), optical[:4].to(device)
        fake = generator(radio)

        # Denormalize
        radio = (radio + 1) / 2
        optical = (optical + 1) / 2
        fake = (fake + 1) / 2

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for i in range(4):
            axes[0, i].imshow(radio[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[0, i].set_title('Radio')
            axes[0, i].axis('off')

            axes[1, i].imshow(fake[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[1, i].set_title('Generated')
            axes[1, i].axis('off')

            axes[2, i].imshow(optical[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[2, i].set_title('Real Optical')
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'), dpi=150)
        plt.close()

    generator.train()


def train_spade_gan(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = PairedDataset(
        config['train_radio_dir'],
        config['train_optical_dir'],
        img_size=config.get('img_size', 256)
    )

    val_dataset = PairedDataset(
        config['val_radio_dir'],
        config['val_optical_dir'],
        img_size=config.get('img_size', 256)
    ) if config.get('val_radio_dir') else None

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 4),
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False) if val_dataset else None

    # Create models
    generator = SPADEGenerator(input_nc=3, output_nc=3, ngf=config.get('ngf', 64)).to(device)
    discriminator = ConditionalDiscriminator(input_nc=6, ndf=config.get('ndf', 64)).to(device)
    criterion = SPADEGANLoss(device=device)

    # TTUR: Different learning rates
    g_lr = config.get('g_lr', 0.0001)
    d_lr = config.get('d_lr', 0.0004)
    opt_G = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.0, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.0, 0.999))

    tracker = GANMetricsTracker()
    save_dir = config.get('save_dir', 'outputs/spade_gan')
    os.makedirs(save_dir, exist_ok=True)

    best_psnr = 0
    epochs = config.get('epochs', 100)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        g_losses, d_losses = [], []
        psnr_vals, ssim_vals = [], []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for radio, optical in pbar:
            radio, optical = radio.to(device), optical.to(device)

            # Train Discriminator
            opt_D.zero_grad()
            with torch.no_grad():
                fake = generator(radio)
            real_preds = discriminator(radio, optical)
            fake_preds = discriminator(radio, fake)
            d_loss = criterion.discriminator_loss(real_preds, fake_preds)
            d_loss.backward()
            opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            fake = generator(radio)
            fake_preds = discriminator(radio, fake)
            g_loss = criterion.generator_loss(fake_preds, fake_img=fake, real_img=optical)
            g_loss.backward()
            opt_G.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            # Compute metrics occasionally
            if len(g_losses) % 10 == 0:
                p, s = compute_image_metrics(fake.detach(), optical)
                psnr_vals.append(p)
                ssim_vals.append(s)

            pbar.set_postfix({'G': f'{g_loss.item():.4f}', 'D': f'{d_loss.item():.4f}'})

        # Log epoch metrics
        avg_g = np.mean(g_losses)
        avg_d = np.mean(d_losses)
        avg_psnr = np.mean(psnr_vals) if psnr_vals else 0
        avg_ssim = np.mean(ssim_vals) if ssim_vals else 0
        tracker.update(avg_g, avg_d, avg_psnr, avg_ssim)

        print(f"Epoch {epoch+1}: G_loss={avg_g:.4f}, D_loss={avg_d:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

        # Save samples
        if (epoch + 1) % config.get('sample_interval', 5) == 0:
            save_samples(generator, train_loader, os.path.join(save_dir, 'samples'), epoch+1, device)

        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch,
                'psnr': best_psnr
            }, os.path.join(save_dir, 'best_model.pth'))

        # Save checkpoint
        if (epoch + 1) % config.get('checkpoint_interval', 10) == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'epoch': epoch
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Save final metrics
    tracker.save_plots(save_dir)

    summary = {
        'final_g_loss': float(np.mean(g_losses)),
        'final_d_loss': float(np.mean(d_losses)),
        'best_psnr': float(best_psnr),
        'epochs_trained': epochs
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete! Best PSNR: {best_psnr:.2f}")
    return generator, discriminator


if __name__ == '__main__':
    config = {
        'train_radio_dir': 'dataset/train/radio',
        'train_optical_dir': 'dataset/train/optical',
        'val_radio_dir': 'dataset/val/radio',
        'val_optical_dir': 'dataset/val/optical',
        'batch_size': 4,
        'epochs': 100,
        'g_lr': 0.0001,
        'd_lr': 0.0004,
        'ngf': 64,
        'ndf': 64,
        'img_size': 256,
        'save_dir': 'outputs/spade_gan',
        'sample_interval': 5,
        'checkpoint_interval': 10
    }

    train_spade_gan(config)
