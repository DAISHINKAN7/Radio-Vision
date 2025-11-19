"""
Advanced Training Pipeline for Radio Vision
- Data augmentation
- Multi-scale training
- Advanced metrics
- Visualization tools
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import h5py
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb  # Weights & Biases for experiment tracking
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RadioSignalAugmentation:
    """Advanced augmentation for radio signals"""
    
    def __init__(self):
        self.augmentations = [
            self.add_noise,
            self.frequency_mask,
            self.time_mask,
            self.amplitude_scale,
            self.frequency_shift,
            self.time_stretch
        ]
    
    def add_noise(self, signal, noise_level=0.1):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, signal.shape)
        return np.clip(signal + noise, 0, 1)
    
    def frequency_mask(self, signal, mask_fraction=0.1):
        """Mask random frequency channels"""
        num_freq = signal.shape[0]
        num_mask = int(num_freq * mask_fraction)
        mask_idx = np.random.choice(num_freq, num_mask, replace=False)
        signal_aug = signal.copy()
        signal_aug[mask_idx] = 0
        return signal_aug
    
    def time_mask(self, signal, mask_fraction=0.1):
        """Mask random time steps"""
        num_time = signal.shape[1]
        num_mask = int(num_time * mask_fraction)
        mask_idx = np.random.choice(num_time, num_mask, replace=False)
        signal_aug = signal.copy()
        signal_aug[:, mask_idx] = 0
        return signal_aug
    
    def amplitude_scale(self, signal, scale_range=(0.8, 1.2)):
        """Scale amplitude"""
        scale = np.random.uniform(*scale_range)
        return np.clip(signal * scale, 0, 1)
    
    def frequency_shift(self, signal, shift_range=(-10, 10)):
        """Shift frequencies"""
        shift = np.random.randint(*shift_range)
        if shift > 0:
            signal_aug = np.pad(signal, ((shift, 0), (0, 0)), mode='edge')[:signal.shape[0]]
        else:
            signal_aug = np.pad(signal, ((0, -shift), (0, 0)), mode='edge')[-shift:]
        return signal_aug
    
    def time_stretch(self, signal, stretch_factor_range=(0.9, 1.1)):
        """Stretch or compress in time"""
        from scipy.ndimage import zoom
        stretch_factor = np.random.uniform(*stretch_factor_range)
        signal_aug = zoom(signal, (1, stretch_factor), order=1)
        
        # Crop or pad to original size
        if signal_aug.shape[1] > signal.shape[1]:
            start = (signal_aug.shape[1] - signal.shape[1]) // 2
            signal_aug = signal_aug[:, start:start+signal.shape[1]]
        elif signal_aug.shape[1] < signal.shape[1]:
            pad_width = signal.shape[1] - signal_aug.shape[1]
            signal_aug = np.pad(signal_aug, ((0, 0), (0, pad_width)), mode='edge')
        
        return signal_aug
    
    def apply_random(self, signal, num_augmentations=2):
        """Apply random augmentations"""
        aug_funcs = np.random.choice(self.augmentations, num_augmentations, replace=False)
        for aug_func in aug_funcs:
            signal = aug_func(signal)
        return signal


class ImageAugmentation:
    """Advanced augmentation for optical images"""
    
    def __init__(self, image_size=256):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.Resize(image_size, image_size),
        ])
    
    def apply(self, image):
        """Apply augmentations"""
        if isinstance(image, np.ndarray):
            augmented = self.transform(image=image)
            return augmented['image']
        return image


class RadioVisionDataset(Dataset):
    """PyTorch Dataset for Radio Vision"""
    
    def __init__(self, dataset_path, split='train', augment=True, image_size=256):
        self.dataset_path = dataset_path
        self.augment = augment
        self.image_size = image_size
        
        # Load metadata
        with open(f"{dataset_path}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Split dataset
        split_idx = int(len(metadata) * 0.8)
        if split == 'train':
            self.metadata = metadata[:split_idx]
        else:
            self.metadata = metadata[split_idx:]
        
        # Load signals
        self.signals = h5py.File(f"{dataset_path}/signals.h5", 'r')['signals']
        
        # Augmentation
        if augment:
            self.signal_aug = RadioSignalAugmentation()
            self.image_aug = ImageAugmentation(image_size)
        
        print(f"📊 Loaded {len(self.metadata)} samples for {split} set")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata[idx]
        
        # Load signal
        signal = self.signals[sample['sample_id']]
        
        # Load optical image
        optical_img = Image.open(f"{self.dataset_path}/{sample['optical_image_path']}")
        optical_img = np.array(optical_img)
        
        # Load radio image
        radio_img = Image.open(f"{self.dataset_path}/{sample['radio_image_path']}")
        radio_img = np.array(radio_img)
        
        # Augmentation
        if self.augment:
            signal = self.signal_aug.apply_random(signal)
            optical_img = self.image_aug.apply(optical_img)
        
        # Convert to tensors
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0)  # Add channel dim
        optical_tensor = torch.FloatTensor(optical_img).permute(2, 0, 1) / 255.0
        radio_tensor = torch.FloatTensor(radio_img).permute(2, 0, 1) / 255.0
        
        # Object type as label
        object_types = ['spiral_galaxy', 'emission_nebula', 'quasar', 'pulsar']
        label = object_types.index(sample['object_type'])
        
        return {
            'signal': signal_tensor,
            'optical_image': optical_tensor,
            'radio_image': radio_tensor,
            'label': label,
            'object_type': sample['object_type']
        }


class AdvancedMetrics:
    """Advanced evaluation metrics"""
    
    def __init__(self):
        self.metrics_history = {
            'mse': [],
            'psnr': [],
            'ssim': [],
            'fid': [],
            'lpips': []
        }
    
    def calculate_fid(self, real_images, fake_images):
        """Frechet Inception Distance"""
        # Simplified FID calculation
        # In production, use proper Inception features
        real_mean = np.mean(real_images, axis=(0, 1, 2))
        fake_mean = np.mean(fake_images, axis=(0, 1, 2))
        
        real_cov = np.cov(real_images.reshape(-1, real_images.shape[-1]).T)
        fake_cov = np.cov(fake_images.reshape(-1, fake_images.shape[-1]).T)
        
        mean_diff = np.sum((real_mean - fake_mean) ** 2)
        cov_mean = np.sqrt(real_cov.diagonal().mean() * fake_cov.diagonal().mean())
        
        fid = mean_diff + cov_mean
        return float(fid)
    
    def calculate_lpips(self, real_images, fake_images):
        """Learned Perceptual Image Patch Similarity"""
        # Simplified LPIPS
        # In production, use lpips library
        diff = np.abs(real_images - fake_images)
        lpips = np.mean(diff)
        return float(lpips)
    
    def update(self, real_images, fake_images):
        """Update all metrics"""
        mse = np.mean((real_images - fake_images) ** 2)
        psnr = 20 * np.log10(255.0 / (np.sqrt(mse) + 1e-10))
        
        self.metrics_history['mse'].append(mse)
        self.metrics_history['psnr'].append(psnr)
        
        # Calculate other metrics periodically
        fid = self.calculate_fid(real_images, fake_images)
        lpips = self.calculate_lpips(real_images, fake_images)
        
        self.metrics_history['fid'].append(fid)
        self.metrics_history['lpips'].append(lpips)
        
        return {
            'mse': mse,
            'psnr': psnr,
            'fid': fid,
            'lpips': lpips
        }
    
    def plot_history(self, save_path='metrics_history.png'):
        """Plot metrics history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['mse', 'psnr', 'fid', 'lpips']
        titles = ['MSE', 'PSNR (dB)', 'FID Score', 'LPIPS']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            ax.plot(self.metrics_history[metric])
            ax.set_title(title)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(title)
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📊 Metrics plot saved: {save_path}")


class AdvancedTrainer:
    """Advanced training with all features"""
    
    def __init__(self, model, dataset_path, config):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # Dataset
        self.train_dataset = RadioVisionDataset(
            dataset_path,
            split='train',
            augment=True,
            image_size=config.get('image_size', 256)
        )
        
        self.val_dataset = RadioVisionDataset(
            dataset_path,
            split='val',
            augment=False,
            image_size=config.get('image_size', 256)
        )
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 100),
            eta_min=1e-6
        )
        
        # Metrics
        self.metrics = AdvancedMetrics()
        
        # WandB logging
        if config.get('use_wandb', False):
            wandb.init(
                project="radio-vision",
                config=config,
                name=config.get('experiment_name', 'radio_vision_training')
            )
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            signal = batch['signal'].to(self.device)
            optical = batch['optical_image'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(signal)
            
            # Loss
            loss = F.mse_loss(output, optical)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to WandB
            if self.config.get('use_wandb', False):
                wandb.log({'train_loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                signal = batch['signal'].to(self.device)
                optical = batch['optical_image'].to(self.device)
                
                output = self.model(signal)
                loss = F.mse_loss(output, optical)
                
                total_loss += loss.item()
                all_outputs.append(output.cpu().numpy())
                all_targets.append(optical.cpu().numpy())
        
        # Calculate metrics
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = self.metrics.update(all_targets, all_outputs)
        
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, metrics
    
    def train(self):
        """Full training loop"""
        num_epochs = self.config.get('num_epochs', 100)
        best_val_loss = float('inf')
        
        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, metrics = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            # Print stats
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  FID: {metrics['fid']:.2f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print("  ✅ Best model saved!")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Plot metrics
        self.metrics.plot_history()
        
        print("\n🎉 Training complete!")
    
    def save_checkpoint(self, path):
        """Save checkpoint"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': self.metrics.metrics_history
        }, path)


if __name__ == "__main__":
    # Training configuration
    config = {
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'image_size': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'use_wandb': False,
        'experiment_name': 'radio_vision_v1'
    }
    
    print("🔬 Advanced Training Pipeline")
    print("=" * 60)
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print("=" * 60)