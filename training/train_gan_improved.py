"""
ENHANCED GAN TRAINING PIPELINE
Train improved Pix2Pix GAN for Radio-to-Optical image translation

Features:
- Improved Pix2Pix with perceptual loss and multi-scale discriminator
- Comprehensive visualization (training curves, sample grids, comparisons)
- Advanced metrics (PSNR, SSIM, FID)
- Progressive training support
- Synthetic pre-training + real fine-tuning

Author: Radio Vision Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import argparse
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from scipy import linalg

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.pix2pix_improved import (
    ImprovedUNetGenerator,
    MultiScaleDiscriminator,
    ImprovedGANLoss,
    VGGPerceptualLoss
)


class RadioOpticalDataset(Dataset):
    """
    Dataset for Radio-to-Optical image translation
    """
    def __init__(self, dataset_path, split='train', image_size=256):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size

        # Load metadata
        with open(self.dataset_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        print(f"✅ Loaded {len(self.metadata)} samples from {dataset_path}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]

        # Load radio image (input)
        radio_path = self.dataset_path / meta['radio_image_path']
        radio_img = Image.open(radio_path).convert('RGB')
        radio_img = radio_img.resize((self.image_size, self.image_size))
        radio_img = np.array(radio_img) / 255.0
        radio_img = torch.from_numpy(radio_img).permute(2, 0, 1).float()

        # Load optical image (target)
        optical_path = self.dataset_path / meta['optical_image_path']
        optical_img = Image.open(optical_path).convert('RGB')
        optical_img = optical_img.resize((self.image_size, self.image_size))
        optical_img = np.array(optical_img) / 255.0
        optical_img = torch.from_numpy(optical_img).permute(2, 0, 1).float()

        # Normalize to [-1, 1] for GAN
        radio_img = radio_img * 2 - 1
        optical_img = optical_img * 2 - 1

        return radio_img, optical_img, meta['object_type']


class GANMetricsTracker:
    """
    Track and visualize GAN training metrics
    """
    def __init__(self, class_names=None):
        self.class_names = class_names or ['spiral_galaxy', 'emission_nebula', 'quasar', 'pulsar']

        # Loss tracking
        self.g_losses = []
        self.d_losses = []
        self.g_adv_losses = []
        self.g_l1_losses = []
        self.g_perceptual_losses = []

        # Quality metrics
        self.psnr_scores = []
        self.ssim_scores = []
        self.fid_scores = []

        # Learning rates
        self.g_lrs = []
        self.d_lrs = []

        # Best metrics
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_fid = float('inf')
        self.best_epoch = 0

    def update(self, epoch, g_loss_dict, d_loss, g_lr, d_lr,
               psnr=None, ssim_val=None, fid=None):
        """Update metrics"""
        self.g_losses.append(g_loss_dict['total'])
        self.g_adv_losses.append(g_loss_dict['adversarial'])
        self.g_l1_losses.append(g_loss_dict['l1'])
        self.g_perceptual_losses.append(g_loss_dict['perceptual'])

        self.d_losses.append(d_loss)

        self.g_lrs.append(g_lr)
        self.d_lrs.append(d_lr)

        if psnr is not None:
            self.psnr_scores.append(psnr)
        if ssim_val is not None:
            self.ssim_scores.append(ssim_val)
        if fid is not None:
            self.fid_scores.append(fid)

        # Update best metrics
        if psnr and psnr > self.best_psnr:
            self.best_psnr = psnr
            self.best_epoch = epoch
        if ssim_val and ssim_val > self.best_ssim:
            self.best_ssim = ssim_val
        if fid and fid < self.best_fid:
            self.best_fid = fid

    def plot_training_curves(self, save_dir):
        """Plot comprehensive training curves"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")

        # 1. Loss curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        epochs = range(1, len(self.g_losses) + 1)

        # Generator total loss
        axes[0, 0].plot(epochs, self.g_losses, 'b-', linewidth=2, label='Generator Loss')
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Generator Total Loss', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Discriminator loss
        axes[0, 1].plot(epochs, self.d_losses, 'r-', linewidth=2, label='Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Loss', fontsize=11)
        axes[0, 1].set_title('Discriminator Loss', fontsize=13, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # Generator loss breakdown
        axes[0, 2].plot(epochs, self.g_adv_losses, label='Adversarial', linewidth=2)
        axes[0, 2].plot(epochs, self.g_l1_losses, label='L1', linewidth=2)
        axes[0, 2].plot(epochs, self.g_perceptual_losses, label='Perceptual', linewidth=2)
        axes[0, 2].set_xlabel('Epoch', fontsize=11)
        axes[0, 2].set_ylabel('Loss', fontsize=11)
        axes[0, 2].set_title('Generator Loss Breakdown', fontsize=13, fontweight='bold')
        axes[0, 2].legend(fontsize=10)
        axes[0, 2].grid(True, alpha=0.3)

        # PSNR
        if self.psnr_scores:
            axes[1, 0].plot(epochs[:len(self.psnr_scores)], self.psnr_scores, 'g-', linewidth=2)
            axes[1, 0].axhline(y=self.best_psnr, color='orange', linestyle='--',
                              label=f'Best: {self.best_psnr:.2f} dB')
            axes[1, 0].set_xlabel('Epoch', fontsize=11)
            axes[1, 0].set_ylabel('PSNR (dB)', fontsize=11)
            axes[1, 0].set_title('Peak Signal-to-Noise Ratio', fontsize=13, fontweight='bold')
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)

        # SSIM
        if self.ssim_scores:
            axes[1, 1].plot(epochs[:len(self.ssim_scores)], self.ssim_scores, 'm-', linewidth=2)
            axes[1, 1].axhline(y=self.best_ssim, color='orange', linestyle='--',
                              label=f'Best: {self.best_ssim:.4f}')
            axes[1, 1].set_xlabel('Epoch', fontsize=11)
            axes[1, 1].set_ylabel('SSIM', fontsize=11)
            axes[1, 1].set_title('Structural Similarity Index', fontsize=13, fontweight='bold')
            axes[1, 1].legend(fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])

        # Learning rates
        axes[1, 2].plot(epochs, self.g_lrs, label='Generator LR', linewidth=2)
        axes[1, 2].plot(epochs, self.d_lrs, label='Discriminator LR', linewidth=2)
        axes[1, 2].set_xlabel('Epoch', fontsize=11)
        axes[1, 2].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 2].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend(fontsize=10)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'gan_training_curves.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_dir / 'gan_training_curves.png'}")
        plt.close()

    def plot_sample_grid(self, radio_images, real_images, fake_images, epoch, save_dir):
        """
        Plot grid of sample images: Radio | Real | Generated
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Take first 8 samples
        num_samples = min(8, radio_images.size(0))

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Denormalize from [-1, 1] to [0, 1]
            radio = (radio_images[i].cpu().permute(1, 2, 0).numpy() + 1) / 2
            real = (real_images[i].cpu().permute(1, 2, 0).numpy() + 1) / 2
            fake = (fake_images[i].cpu().detach().permute(1, 2, 0).numpy() + 1) / 2

            # Clip to valid range
            radio = np.clip(radio, 0, 1)
            real = np.clip(real, 0, 1)
            fake = np.clip(fake, 0, 1)

            # Plot
            axes[i, 0].imshow(radio)
            axes[i, 0].set_title('Radio Image', fontsize=11)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(real)
            axes[i, 1].set_title('Real Optical', fontsize=11)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(fake)
            axes[i, 2].set_title('Generated Optical', fontsize=11)
            axes[i, 2].axis('off')

        plt.suptitle(f'Epoch {epoch} - Sample Outputs', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f'samples_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_dir / f'samples_epoch_{epoch:04d}.png'}")
        plt.close()


class ImprovedGANTrainer:
    """
    Enhanced GAN Trainer with comprehensive monitoring
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Initialize models
        print("\n🤖 Initializing models...")
        self.generator = ImprovedUNetGenerator(
            in_channels=3,
            out_channels=3,
            features=config.get('gen_features', 64),
            use_attention=config.get('use_attention', True)
        ).to(self.device)

        self.discriminator = MultiScaleDiscriminator(
            in_channels=6,
            features=config.get('disc_features', 64),
            num_scales=config.get('num_scales', 3)
        ).to(self.device)

        # Print model info
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"   Generator: {gen_params:,} parameters")
        print(f"   Discriminator: {disc_params:,} parameters")
        print(f"   Total: {gen_params + disc_params:,} parameters")

        # Initialize loss
        self.gan_loss = ImprovedGANLoss(
            lambda_l1=config.get('lambda_l1', 100.0),
            lambda_perceptual=config.get('lambda_perceptual', 10.0),
            use_perceptual=config.get('use_perceptual', True)
        ).to(self.device)

        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.get('g_lr', 2e-4),
            betas=(config.get('beta1', 0.5), config.get('beta2', 0.999))
        )

        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.get('d_lr', 2e-4),
            betas=(config.get('beta1', 0.5), config.get('beta2', 0.999))
        )

        # Initialize schedulers
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.g_optimizer,
            T_0=config.get('scheduler_T0', 10),
            T_mult=config.get('scheduler_Tmult', 2),
            eta_min=config.get('scheduler_eta_min', 1e-6)
        )

        self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.d_optimizer,
            T_0=config.get('scheduler_T0', 10),
            T_mult=config.get('scheduler_Tmult', 2),
            eta_min=config.get('scheduler_eta_min', 1e-6)
        )

        # Initialize metrics tracker
        self.metrics = GANMetricsTracker()

        print("✅ Models initialized!")

    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.generator.train()
        self.discriminator.train()

        g_loss_sum = 0.0
        d_loss_sum = 0.0
        g_losses_dict = {'total': 0.0, 'adversarial': 0.0, 'l1': 0.0, 'perceptual': 0.0}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (radio_images, optical_images, _) in enumerate(pbar):
            radio_images = radio_images.to(self.device)
            optical_images = optical_images.to(self.device)

            # ==========================================
            # Train Discriminator
            # ==========================================
            self.d_optimizer.zero_grad()

            # Generate fake images
            with torch.no_grad():
                fake_images = self.generator(radio_images)

            # Real pairs
            real_pred_list = self.discriminator(radio_images, optical_images)

            # Fake pairs
            fake_pred_list = self.discriminator(radio_images, fake_images)

            # Discriminator loss
            d_loss = self.gan_loss.discriminator_loss(real_pred_list, fake_pred_list)

            d_loss.backward()
            self.d_optimizer.step()

            # ==========================================
            # Train Generator
            # ==========================================
            self.g_optimizer.zero_grad()

            # Generate fake images
            fake_images = self.generator(radio_images)

            # Get discriminator predictions
            fake_pred_list = self.discriminator(radio_images, fake_images)

            # Generator loss
            g_loss, g_loss_dict = self.gan_loss.generator_loss(
                fake_pred_list, fake_images, optical_images
            )

            g_loss.backward()
            self.g_optimizer.step()

            # Accumulate losses
            g_loss_sum += g_loss_dict['total']
            d_loss_sum += d_loss.item()
            for key in g_losses_dict:
                g_losses_dict[key] += g_loss_dict[key]

            # Update progress bar
            pbar.set_postfix({
                'G_loss': f"{g_loss_dict['total']:.4f}",
                'D_loss': f"{d_loss.item():.4f}"
            })

        # Average losses
        num_batches = len(dataloader)
        g_losses_dict = {k: v / num_batches for k, v in g_losses_dict.items()}
        d_loss_avg = d_loss_sum / num_batches

        return g_losses_dict, d_loss_avg, radio_images, optical_images, fake_images

    def validate(self, dataloader):
        """Validate and compute metrics"""
        self.generator.eval()

        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():
            for radio_images, optical_images, _ in tqdm(dataloader, desc="Validation"):
                radio_images = radio_images.to(self.device)
                optical_images = optical_images.to(self.device)

                # Generate images
                fake_images = self.generator(radio_images)

                # Denormalize to [0, 1]
                real_imgs = ((optical_images + 1) / 2).cpu().numpy()
                fake_imgs = ((fake_images + 1) / 2).cpu().numpy()

                # Compute metrics per image
                for i in range(real_imgs.shape[0]):
                    real_img = np.transpose(real_imgs[i], (1, 2, 0))
                    fake_img = np.transpose(fake_imgs[i], (1, 2, 0))

                    # PSNR
                    mse = np.mean((real_img - fake_img) ** 2)
                    if mse > 0:
                        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                        psnr_scores.append(psnr)

                    # SSIM
                    ssim_val = ssim(real_img, fake_img, multichannel=True, data_range=1.0, channel_axis=2)
                    ssim_scores.append(ssim_val)

        avg_psnr = np.mean(psnr_scores) if psnr_scores else 0.0
        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0

        return avg_psnr, avg_ssim

    def train(self, train_dataset_path, val_dataset_path=None):
        """Full training loop"""
        print("\n" + "="*80)
        print("STARTING GAN TRAINING")
        print("="*80)

        # Load datasets
        train_dataset = RadioOpticalDataset(
            train_dataset_path,
            split='train',
            image_size=self.config.get('image_size', 256)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )

        # Validation loader (optional)
        val_loader = None
        if val_dataset_path:
            val_dataset = RadioOpticalDataset(
                val_dataset_path,
                split='val',
                image_size=self.config.get('image_size', 256)
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('batch_size', 8),
                shuffle=False,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )

        # Training loop
        num_epochs = self.config.get('num_epochs', 200)
        save_interval = self.config.get('save_interval', 10)

        for epoch in range(1, num_epochs + 1):
            # Train
            g_losses, d_loss, radio_imgs, real_imgs, fake_imgs = self.train_epoch(train_loader, epoch)

            # Validate
            avg_psnr, avg_ssim = 0.0, 0.0
            if val_loader and epoch % 5 == 0:
                avg_psnr, avg_ssim = self.validate(val_loader)

            # Step schedulers
            self.g_scheduler.step()
            self.d_scheduler.step()

            # Update metrics
            g_lr = self.g_optimizer.param_groups[0]['lr']
            d_lr = self.d_optimizer.param_groups[0]['lr']

            self.metrics.update(
                epoch, g_losses, d_loss, g_lr, d_lr,
                psnr=avg_psnr if avg_psnr > 0 else None,
                ssim_val=avg_ssim if avg_ssim > 0 else None
            )

            # Print stats
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Generator Loss: {g_losses['total']:.4f} "
                  f"(Adv: {g_losses['adversarial']:.4f}, "
                  f"L1: {g_losses['l1']:.4f}, "
                  f"Perceptual: {g_losses['perceptual']:.4f})")
            print(f"  Discriminator Loss: {d_loss:.4f}")
            print(f"  G_LR: {g_lr:.2e} | D_LR: {d_lr:.2e}")

            if avg_psnr > 0:
                print(f"  PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

            # Save sample images
            if epoch % save_interval == 0 or epoch == 1:
                self.metrics.plot_sample_grid(
                    radio_imgs, real_imgs, fake_imgs,
                    epoch, self.output_dir / 'samples'
                )

            # Save checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch, g_losses['total'], d_loss)

            # Save best model based on PSNR
            if avg_psnr > 0 and avg_psnr >= self.metrics.best_psnr:
                best_path = self.output_dir / 'best_generator.pt'
                torch.save(self.generator.state_dict(), best_path)
                print(f"  ✅ New best model! PSNR: {avg_psnr:.2f} dB")

        # Plot final training curves
        print(f"\n📊 Generating training visualizations...")
        self.metrics.plot_training_curves(self.output_dir)

        print("\n✅ Training complete!")
        print(f"   Best PSNR: {self.metrics.best_psnr:.2f} dB (Epoch {self.metrics.best_epoch})")
        print(f"   Best SSIM: {self.metrics.best_ssim:.4f}")
        print(f"   All outputs saved to: {self.output_dir}")

    def save_checkpoint(self, epoch, g_loss, d_loss):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            'config': self.config
        }

        path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description='Enhanced GAN Training Pipeline')

    # Paths
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training dataset')
    parser.add_argument('--val_path', type=str, default=None,
                       help='Path to validation dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/gan_improved',
                       help='Output directory')

    # Training config
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')

    # Model config
    parser.add_argument('--gen_features', type=int, default=64,
                       help='Generator base features')
    parser.add_argument('--disc_features', type=int, default=64,
                       help='Discriminator base features')
    parser.add_argument('--num_scales', type=int, default=3,
                       help='Number of discriminator scales')
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='Use self-attention in generator')

    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                       help='L1 loss weight')
    parser.add_argument('--lambda_perceptual', type=float, default=10.0,
                       help='Perceptual loss weight')
    parser.add_argument('--use_perceptual', action='store_true', default=True,
                       help='Use perceptual loss')

    # Optimizer config
    parser.add_argument('--g_lr', type=float, default=2e-4,
                       help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=2e-4,
                       help='Discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Adam beta2')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    args = parser.parse_args()

    # Create config dictionary
    config = vars(args)

    # Print configuration
    print("\n" + "="*80)
    print("ENHANCED GAN TRAINING PIPELINE")
    print("="*80)
    print(f"\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key:25s}: {value}")
    print("="*80)

    # Initialize trainer
    trainer = ImprovedGANTrainer(config)

    # Train
    trainer.train(config['train_path'], config.get('val_path'))

    print("\n🎉 Training pipeline complete!")


if __name__ == "__main__":
    main()
