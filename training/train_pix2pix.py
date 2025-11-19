"""
Training Script for Pix2Pix GAN
- Trains on 10k radio-optical image pairs
- Adversarial + L1 loss
- Progressive training
- Checkpointing & visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from models.pix2pix_gan import UNetGenerator, PatchGANDiscriminator

try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False


class RadioOpticalPairDataset(Dataset):
    """Dataset for radio-optical image pairs"""
    
    def __init__(self, dataset_path, split='train', train_ratio=0.8, image_size=256):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        
        # Load metadata
        with open(self.dataset_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Split dataset
        num_samples = len(self.metadata)
        train_size = int(num_samples * train_ratio)
        
        if split == 'train':
            self.indices = list(range(0, train_size))
        else:
            self.indices = list(range(train_size, num_samples))
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        print(f"📊 {split.upper()} dataset: {len(self.indices)} pairs")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.metadata[actual_idx]
        
        # Load radio image
        radio_path = self.dataset_path / sample['radio_image_path']
        radio_img = Image.open(radio_path).convert('RGB')
        radio_img = self.transform(radio_img)
        
        # Load optical image
        optical_path = self.dataset_path / sample['optical_image_path']
        optical_img = Image.open(optical_path).convert('RGB')
        optical_img = self.transform(optical_img)
        
        return radio_img, optical_img


def train_epoch(generator, discriminator, train_loader, 
                optimizer_g, optimizer_d, criterion_gan, criterion_l1, 
                device, epoch, lambda_l1=100):
    """Train for one epoch"""
    
    generator.train()
    discriminator.train()
    
    total_loss_g = 0
    total_loss_d = 0
    total_loss_l1 = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for radio_imgs, optical_imgs in pbar:
        batch_size = radio_imgs.size(0)
        radio_imgs = radio_imgs.to(device)
        optical_imgs = optical_imgs.to(device)
        
        # Labels
        real_label = torch.ones(batch_size, 1, 30, 30).to(device)
        fake_label = torch.zeros(batch_size, 1, 30, 30).to(device)
        
        # ==================
        # Train Generator
        # ==================
        optimizer_g.zero_grad()
        
        fake_optical = generator(radio_imgs)
        
        # Adversarial loss
        pred_fake = discriminator(radio_imgs, fake_optical)
        loss_gan = criterion_gan(pred_fake, real_label)
        
        # L1 loss
        loss_l1 = criterion_l1(fake_optical, optical_imgs)
        
        # Total generator loss
        loss_g = loss_gan + lambda_l1 * loss_l1
        
        loss_g.backward()
        optimizer_g.step()
        
        # =====================
        # Train Discriminator
        # =====================
        optimizer_d.zero_grad()
        
        # Real loss
        pred_real = discriminator(radio_imgs, optical_imgs)
        loss_real = criterion_gan(pred_real, real_label)
        
        # Fake loss
        pred_fake = discriminator(radio_imgs, fake_optical.detach())
        loss_fake = criterion_gan(pred_fake, fake_label)
        
        # Total discriminator loss
        loss_d = (loss_real + loss_fake) / 2
        
        loss_d.backward()
        optimizer_d.step()
        
        # Update stats
        total_loss_g += loss_g.item()
        total_loss_d += loss_d.item()
        total_loss_l1 += loss_l1.item()
        
        pbar.set_postfix({
            'G': f'{loss_g.item():.3f}',
            'D': f'{loss_d.item():.3f}',
            'L1': f'{loss_l1.item():.3f}'
        })
    
    avg_loss_g = total_loss_g / len(train_loader)
    avg_loss_d = total_loss_d / len(train_loader)
    avg_loss_l1 = total_loss_l1 / len(train_loader)
    
    return avg_loss_g, avg_loss_d, avg_loss_l1


@torch.no_grad()
def validate(generator, val_loader, criterion_l1, device):
    """Validate generator"""
    
    generator.eval()
    total_l1 = 0
    
    for radio_imgs, optical_imgs in tqdm(val_loader, desc='Validation'):
        radio_imgs = radio_imgs.to(device)
        optical_imgs = optical_imgs.to(device)
        
        fake_optical = generator(radio_imgs)
        loss_l1 = criterion_l1(fake_optical, optical_imgs)
        
        total_l1 += loss_l1.item()
    
    avg_l1 = total_l1 / len(val_loader)
    return avg_l1


@torch.no_grad()
def save_sample_images(generator, val_loader, device, save_path, num_samples=4):
    """Save sample generations"""
    
    generator.eval()
    
    # Get samples
    radio_imgs, optical_imgs = next(iter(val_loader))
    radio_imgs = radio_imgs[:num_samples].to(device)
    optical_imgs = optical_imgs[:num_samples].to(device)
    
    # Generate
    fake_optical = generator(radio_imgs)
    
    # Denormalize
    radio_imgs = (radio_imgs + 1) / 2
    optical_imgs = (optical_imgs + 1) / 2
    fake_optical = (fake_optical + 1) / 2
    
    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    
    for i in range(num_samples):
        # Radio
        axes[i, 0].imshow(radio_imgs[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title('Radio Input')
        axes[i, 0].axis('off')
        
        # Generated
        axes[i, 1].imshow(fake_optical[i].cpu().permute(1, 2, 0))
        axes[i, 1].set_title('Generated Optical')
        axes[i, 1].axis('off')
        
        # Ground truth
        axes[i, 2].imshow(optical_imgs[i].cpu().permute(1, 2, 0))
        axes[i, 2].set_title('Real Optical')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_pix2pix(args):
    """Main training function"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # W&B
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project="radio-vision-pix2pix",
            config=vars(args),
            name=args.run_name
        )
    
    # Datasets
    print("📂 Loading datasets...")
    train_dataset = RadioOpticalPairDataset(
        args.dataset_path,
        split='train',
        train_ratio=args.train_ratio,
        image_size=args.image_size
    )
    
    val_dataset = RadioOpticalPairDataset(
        args.dataset_path,
        split='val',
        train_ratio=args.train_ratio,
        image_size=args.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Mac compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Models
    print("🧠 Creating models...")
    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    print(f"   Generator: {sum(p.numel() for p in generator.parameters()):,} parameters")
    print(f"   Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} parameters")
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    
    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    # Optimizers
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )
    
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )
    
    # Training loop
    best_val_l1 = float('inf')
    
    print(f"\n🎓 Starting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        loss_g, loss_d, loss_l1 = train_epoch(
            generator, discriminator, train_loader,
            optimizer_g, optimizer_d, criterion_gan, criterion_l1,
            device, epoch, args.lambda_l1
        )
        
        # Validate
        val_l1 = validate(generator, val_loader, criterion_l1, device)
        
        # Print results
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - G: {loss_g:.4f}, D: {loss_d:.4f}, L1: {loss_l1:.4f}")
        print(f"Val   - L1: {val_l1:.4f}")
        print("=" * 60)
        
        # W&B logging
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss_g': loss_g,
                'train_loss_d': loss_d,
                'train_loss_l1': loss_l1,
                'val_loss_l1': val_l1
            })
        
        # Save samples
        if epoch % args.save_freq == 0:
            sample_dir = Path(args.save_dir) / 'samples'
            sample_dir.mkdir(exist_ok=True, parents=True)
            save_sample_images(
                generator, val_loader, device,
                sample_dir / f'epoch_{epoch}.png'
            )
        
        # Save best model
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            
            checkpoint = {
                'epoch': epoch,
                'generator': generator.state_dict() if not isinstance(generator, nn.DataParallel) else generator.module.state_dict(),
                'discriminator': discriminator.state_dict() if not isinstance(discriminator, nn.DataParallel) else discriminator.module.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'val_l1': val_l1
            }
            
            save_path = Path(args.save_dir) / 'pix2pix_best.pth'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(checkpoint, save_path)
            print(f"✅ Saved best model: L1={val_l1:.4f}")
    
    print(f"\n🎉 Training complete!")
    print(f"✅ Best validation L1: {best_val_l1:.4f}")
    
    # Save final
    final_path = Path(args.save_dir) / 'pix2pix_final.pth'
    torch.save(checkpoint, final_path)
    print(f"💾 Saved final model to: {final_path}")
    
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description='Train Pix2Pix GAN')
    
    # Dataset
    parser.add_argument('--dataset-path', type=str, default='radio_vision_dataset_10k')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--image-size', type=int, default=256)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.0002)
    parser.add_argument('--lambda-l1', type=float, default=100)
    
    # System
    parser.add_argument('--save-dir', type=str, default='models/checkpoints')
    parser.add_argument('--save-freq', type=int, default=10)
    
    # Logging
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--run-name', type=str, default='pix2pix_v1')
    
    args = parser.parse_args()
    
    train_pix2pix(args)