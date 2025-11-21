"""
TRANSFER LEARNING TRAINING PIPELINE
Pre-train on synthetic data → Fine-tune on real data
Addresses the domain gap and data scarcity issues

Features:
- MobileNetV2 (3.5M params) optimized for small datasets
- Focal Loss for class imbalance
- Extreme augmentation pipeline
- MixUp/CutMix
- Comprehensive visualization (training/validation graphs)
- Per-class metrics tracking
- Gradual unfreezing strategy

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
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import argparse
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.mobilenet_classifier import MobileNetV2Classifier, FocalLoss, LabelSmoothingLoss, create_class_weights
from models.extreme_augmentation import ExtremeSignalAugmentation, MixUpAugmentation, CutMixAugmentation, mixup_criterion


class RadioSignalDataset(Dataset):
    """
    PyTorch Dataset for Radio Signals with Advanced Augmentation
    """
    def __init__(self, dataset_path, split='train', augment=True, target_size=(224, 224)):
        self.dataset_path = Path(dataset_path)
        self.augment = augment
        self.target_size = target_size

        # Load metadata
        with open(self.dataset_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Load signals
        with h5py.File(self.dataset_path / 'signals.h5', 'r') as f:
            self.signals = f['signals'][:]

        # Class mapping - standard class names
        self.class_to_idx = {
            'galaxy': 0,
            'quasar': 1,
            'radio_galaxy': 2,
            'agn': 3
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Map folder names with 's' suffix to standard names
        self.folder_to_class = {
            'galaxys': 'galaxy',
            'quasars': 'quasar',
            'radio_galaxys': 'radio_galaxy',
            'agns': 'agn',
            'galaxy': 'galaxy',
            'quasar': 'quasar',
            'radio_galaxy': 'radio_galaxy',
            'agn': 'agn'
        }

        # Create augmentation pipeline
        if augment:
            self.signal_aug = ExtremeSignalAugmentation(p=0.9)
            self.mixup = MixUpAugmentation(alpha=0.4)
            self.cutmix = CutMixAugmentation(alpha=1.0)

        print(f"✅ Loaded {len(self.metadata)} samples from {dataset_path}")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution"""
        from collections import Counter
        # Map folder names to standard class names
        mapped_types = [self.folder_to_class.get(item['object_type'], item['object_type'])
                       for item in self.metadata]
        class_counts = Counter(mapped_types)
        print(f"\n📊 Class Distribution:")
        for cls in ['galaxy', 'quasar', 'radio_galaxy', 'agn']:
            count = class_counts.get(cls, 0)
            print(f"   {cls:20s}: {count:4d} samples")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]

        # Load signal
        signal = self.signals[meta['sample_id']]

        # Apply augmentation
        if self.augment:
            signal = self.signal_aug(signal)

        # Preprocess signal
        signal_tensor = self._preprocess_signal(signal)

        # Get label - map folder name to standard class name
        object_type = self.folder_to_class.get(meta['object_type'], meta['object_type'])
        label = self.class_to_idx[object_type]

        return signal_tensor, label, object_type

    def _preprocess_signal(self, signal):
        """
        Preprocess signal to target size
        """
        from scipy.ndimage import zoom

        # Handle NaN values
        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=0.0)

        # Normalize to [0, 1]
        if signal.max() > signal.min():
            signal = (signal - signal.min()) / (signal.max() - signal.min())
        else:
            signal = np.zeros_like(signal)

        # Resize to target size
        h, w = signal.shape
        target_h, target_w = self.target_size

        zoom_h = target_h / h
        zoom_w = target_w / w

        signal_resized = zoom(signal, (zoom_h, zoom_w), order=1)
        signal_resized = signal_resized[:target_h, :target_w]

        # Convert to tensor
        signal_tensor = torch.from_numpy(signal_resized).float()
        signal_tensor = signal_tensor.unsqueeze(0)  # Add channel dim

        return signal_tensor


class ImageFolderDataset(Dataset):
    """
    PyTorch Dataset for loading images from folders.
    Expects structure: dataset_path/class_name/*.png
    """
    def __init__(self, dataset_path, split='train', augment=True, target_size=(224, 224)):
        self.dataset_path = Path(dataset_path)
        self.augment = augment
        self.target_size = target_size

        # Class mapping - standard class names
        self.class_to_idx = {
            'galaxy': 0,
            'quasar': 1,
            'radio_galaxy': 2,
            'agn': 3
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Map folder names with 's' suffix to standard names
        self.folder_to_class = {
            'galaxys': 'galaxy',
            'quasars': 'quasar',
            'radio_galaxys': 'radio_galaxy',
            'agns': 'agn',
            'galaxy': 'galaxy',
            'quasar': 'quasar',
            'radio_galaxy': 'radio_galaxy',
            'agn': 'agn'
        }

        # Load samples from folders
        self.samples = []
        for folder in self.dataset_path.iterdir():
            if folder.is_dir() and folder.name in self.folder_to_class:
                class_name = self.folder_to_class[folder.name]
                for img_path in folder.glob('*.png'):
                    self.samples.append((img_path, class_name))
                for img_path in folder.glob('*.jpg'):
                    self.samples.append((img_path, class_name))
                for img_path in folder.glob('*.jpeg'):
                    self.samples.append((img_path, class_name))

        # Transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ])

        print(f"Loaded {len(self.samples)} samples from {dataset_path}")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution"""
        from collections import Counter
        class_counts = Counter([s[1] for s in self.samples])
        print(f"Class distribution:")
        for cls in ['galaxy', 'quasar', 'radio_galaxy', 'agn']:
            count = class_counts.get(cls, 0)
            print(f"  {cls}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        # Convert to grayscale (single channel) for consistency
        img_gray = img_tensor.mean(dim=0, keepdim=True)

        label = self.class_to_idx[class_name]
        return img_gray, label, class_name


class MetricsTracker:
    """
    Track and visualize training metrics
    """
    def __init__(self, num_classes=4, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or ['galaxy', 'quasar', 'radio_galaxy', 'agn']

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []

        # Per-class metrics
        self.val_precisions = {cls: [] for cls in self.class_names}
        self.val_recalls = {cls: [] for cls in self.class_names}
        self.val_f1s = {cls: [] for cls in self.class_names}

        # Best metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr,
               per_class_metrics=None):
        """Update metrics"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)

        # Update per-class metrics
        if per_class_metrics:
            for i, cls in enumerate(self.class_names):
                self.val_precisions[cls].append(per_class_metrics['precision'][i])
                self.val_recalls[cls].append(per_class_metrics['recall'][i])
                self.val_f1s[cls].append(per_class_metrics['f1'][i])

        # Update best
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch

    def plot_training_curves(self, save_dir):
        """Plot comprehensive training curves"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")

        # 1. Loss and Accuracy Curves
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].axhline(y=self.best_val_acc, color='g', linestyle='--',
                          label=f'Best Val Acc: {self.best_val_acc:.2%} (Epoch {self.best_epoch})')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

        # Learning rate
        axes[1, 0].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Gap between train and val accuracy
        gap = [train - val for train, val in zip(self.train_accs, self.val_accs)]
        axes[1, 1].plot(epochs, gap, 'm-', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Train Acc - Val Acc', fontsize=12)
        axes[1, 1].set_title('Overfitting Monitor (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_dir / 'training_curves.png'}")
        plt.close()

        # 2. Per-Class Performance
        if self.val_f1s[self.class_names[0]]:  # Check if we have per-class metrics
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            for i, (metric_name, metric_dict) in enumerate([
                ('Precision', self.val_precisions),
                ('Recall', self.val_recalls),
                ('F1-Score', self.val_f1s)
            ]):
                row = i // 2
                col = i % 2

                for cls in self.class_names:
                    if metric_dict[cls]:
                        axes[row, col].plot(epochs, metric_dict[cls],
                                          label=cls.replace('_', ' ').title(),
                                          linewidth=2, marker='o', markersize=4)

                axes[row, col].set_xlabel('Epoch', fontsize=12)
                axes[row, col].set_ylabel(metric_name, fontsize=12)
                axes[row, col].set_title(f'Validation {metric_name} per Class',
                                        fontsize=14, fontweight='bold')
                axes[row, col].legend(fontsize=10)
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].set_ylim([0, 1])

            # Remove empty subplot
            fig.delaxes(axes[1, 1])

            plt.tight_layout()
            plt.savefig(save_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_dir / 'per_class_metrics.png'}")
            plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, save_dir):
        """Plot confusion matrix"""
        save_dir = Path(save_dir)

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[c.replace('_', '\n').title() for c in self.class_names],
                   yticklabels=[c.replace('_', '\n').title() for c in self.class_names],
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=[c.replace('_', '\n').title() for c in self.class_names],
                   yticklabels=[c.replace('_', '\n').title() for c in self.class_names],
                   ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_dir / 'confusion_matrix.png'}")
        plt.close()


class TransferLearningTrainer:
    """
    Transfer Learning Trainer
    Phase 1: Pre-train on synthetic data
    Phase 2: Fine-tune on real data
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

        # Initialize model
        self.model = MobileNetV2Classifier(
            num_classes=4,
            pretrained=True,
            dropout=config.get('dropout', 0.7)
        ).to(self.device)

        # Print model info
        params = self.model.get_num_params()
        print(f"\n🤖 Model: MobileNetV2Classifier")
        print(f"   Total parameters: {params['total']:,}")
        print(f"   Trainable parameters: {params['trainable']:,}")

        # Calculate class weights if real dataset provided
        self.class_weights = None
        if config.get('real_dataset_path'):
            self.class_weights = create_class_weights(config['real_dataset_path'])
            self.class_weights = self.class_weights.to(self.device)

        # Initialize loss function
        if config.get('use_focal_loss', True):
            self.criterion = FocalLoss(
                alpha=self.class_weights,
                gamma=config.get('focal_gamma', 2.0)
            )
            print(f"📊 Loss: Focal Loss (gamma={config.get('focal_gamma', 2.0)})")
        else:
            self.criterion = LabelSmoothingLoss(
                num_classes=4,
                smoothing=config.get('label_smoothing', 0.1)
            )
            print(f"📊 Loss: Label Smoothing (smoothing={config.get('label_smoothing', 0.1)})")

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('scheduler_T0', 10),
            T_mult=config.get('scheduler_Tmult', 2),
            eta_min=config.get('scheduler_eta_min', 1e-6)
        )

        # Initialize metrics tracker
        self.metrics = MetricsTracker(num_classes=4)

        # MixUp/CutMix
        self.use_mixup = config.get('use_mixup', True)
        self.use_cutmix = config.get('use_cutmix', True)
        self.mixup = MixUpAugmentation(alpha=0.4)
        self.cutmix = CutMixAugmentation(alpha=1.0)

        print(f"🔄 Augmentation: MixUp={self.use_mixup}, CutMix={self.use_cutmix}")

    def train_epoch(self, dataloader, epoch, phase='pretrain'):
        """Train one epoch"""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"[{phase.upper()}] Epoch {epoch}")

        for batch_idx, (signals, labels, _) in enumerate(pbar):
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            # Apply MixUp/CutMix randomly
            use_mix = False
            if self.use_mixup and np.random.rand() < 0.5:
                # Get another random batch for mixing
                idx = np.random.randint(len(dataloader.dataset))
                signal2, label2, _ = dataloader.dataset[idx]
                signal2 = signal2.unsqueeze(0).to(self.device)
                label2 = torch.tensor([label2]).to(self.device)

                # Apply MixUp
                mixed_signals = []
                for i in range(signals.size(0)):
                    mixed_signal, y1, y2, lam = self.mixup(
                        signals[i].cpu().numpy(),
                        signal2[0].cpu().numpy(),
                        labels[i].item(),
                        label2[0].item()
                    )
                    mixed_signals.append(torch.from_numpy(mixed_signal))

                signals = torch.stack(mixed_signals).to(self.device)
                use_mix = True

            elif self.use_cutmix and np.random.rand() < 0.5:
                # Apply CutMix
                idx = np.random.randint(len(dataloader.dataset))
                signal2, label2, _ = dataloader.dataset[idx]
                signal2 = signal2.unsqueeze(0).to(self.device)
                label2 = torch.tensor([label2]).to(self.device)

                mixed_signals = []
                for i in range(signals.size(0)):
                    mixed_signal, y1, y2, lam = self.cutmix(
                        signals[i].cpu().numpy(),
                        signal2[0].cpu().numpy(),
                        labels[i].item(),
                        label2[0].item()
                    )
                    mixed_signals.append(torch.from_numpy(mixed_signal))

                signals = torch.stack(mixed_signals).to(self.device)
                use_mix = True

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(signals)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            running_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for signals, labels, _ in tqdm(dataloader, desc="Validation"):
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                running_loss += loss.item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        per_class_metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return epoch_loss, epoch_acc, per_class_metrics, all_labels, all_preds

    def pretrain(self, synthetic_dataset_path):
        """
        Phase 1: Pre-train on synthetic data
        """
        print("\n" + "="*80)
        print("PHASE 1: PRE-TRAINING ON SYNTHETIC DATA")
        print("="*80)

        # Load synthetic dataset
        train_dataset = RadioSignalDataset(
            synthetic_dataset_path,
            split='train',
            augment=True,
            target_size=(224, 224)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )

        num_epochs = self.config.get('pretrain_epochs', 50)

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch, phase='pretrain')

            # Step scheduler
            self.scheduler.step()

            # Print stats
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Acc: {train_acc:.2%}")
            print(f"  LR: {current_lr:.2e}")

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = self.output_dir / f'pretrain_epoch_{epoch}.pt'
                self.save_checkpoint(checkpoint_path, epoch, train_loss, train_acc)

        # Save final pretrained model
        pretrained_path = self.output_dir / 'pretrained_model.pt'
        self.save_checkpoint(pretrained_path, num_epochs, train_loss, train_acc)
        print(f"\n✅ Pre-training complete! Saved to: {pretrained_path}")

        return pretrained_path

    def finetune(self, real_dataset_path, pretrained_path=None):
        """
        Phase 2: Fine-tune on real data
        """
        print("\n" + "="*80)
        print("PHASE 2: FINE-TUNING ON REAL DATA")
        print("="*80)

        # Load pretrained weights if provided
        if pretrained_path:
            print(f"\n📦 Loading pretrained weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Pretrained weights loaded!")

        # Load real dataset (use ImageFolderDataset for image folders)
        full_dataset = ImageFolderDataset(
            real_dataset_path,
            split='train',
            augment=True,
            target_size=(224, 224)
        )

        # Split into train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"\n📊 Dataset Split:")
        print(f"   Train: {train_size} samples")
        print(f"   Val: {val_size} samples")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),  # Smaller batch for real data
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )

        # Gradual unfreezing strategy
        num_epochs = self.config.get('finetune_epochs', 100)
        unfreeze_epoch = self.config.get('unfreeze_epoch', 20)

        # Start with frozen backbone
        if self.config.get('freeze_backbone', True):
            print(f"\n❄️  Freezing backbone (will unfreeze at epoch {unfreeze_epoch})")
            self.model.freeze_backbone()

        # Training loop
        best_val_acc = 0.0
        patience = self.config.get('patience', 15)
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Unfreeze backbone after certain epochs
            if epoch == unfreeze_epoch and self.config.get('freeze_backbone', True):
                print(f"\n🔥 Unfreezing backbone at epoch {epoch}")
                self.model.unfreeze_backbone()
                # Lower learning rate after unfreezing
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch, phase='finetune')

            # Validate
            val_loss, val_acc, per_class_metrics, val_labels, val_preds = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step()

            # Update metrics tracker
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics.update(
                epoch, train_loss, val_loss, train_acc, val_acc,
                current_lr, per_class_metrics
            )

            # Print stats
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
            print(f"  LR: {current_lr:.2e}")

            # Print per-class metrics
            class_names = ['galaxy', 'quasar', 'radio_galaxy', 'agn']
            print(f"\n  Per-Class Metrics:")
            for i, cls in enumerate(class_names):
                print(f"    {cls:20s}: P={per_class_metrics['precision'][i]:.2%} "
                      f"R={per_class_metrics['recall'][i]:.2%} "
                      f"F1={per_class_metrics['f1'][i]:.2%}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                best_path = self.output_dir / 'best_model.pt'
                self.save_checkpoint(best_path, epoch, val_loss, val_acc)
                print(f"  ✅ New best model! Val Acc: {val_acc:.2%}")

                # Save confusion matrix for best model
                self.metrics.plot_confusion_matrix(val_labels, val_preds, self.output_dir)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {patience} epochs without improvement")
                break

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = self.output_dir / f'finetune_epoch_{epoch}.pt'
                self.save_checkpoint(checkpoint_path, epoch, val_loss, val_acc)

        # Plot final training curves
        print(f"\n📊 Generating training visualizations...")
        self.metrics.plot_training_curves(self.output_dir)

        # Print final classification report
        print(f"\n" + "="*80)
        print("FINAL CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(val_labels, val_preds, target_names=class_names))

        print(f"\n✅ Fine-tuning complete!")
        print(f"   Best Val Acc: {best_val_acc:.2%} (Epoch {self.metrics.best_epoch})")
        print(f"   All outputs saved to: {self.output_dir}")

    def save_checkpoint(self, path, epoch, loss, acc):
        """Save checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'acc': acc,
            'config': self.config
        }, path)


def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Training Pipeline')

    # Paths
    parser.add_argument('--synthetic_path', type=str, required=True,
                       help='Path to synthetic dataset')
    parser.add_argument('--real_path', type=str, required=True,
                       help='Path to real dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/transfer_learning',
                       help='Output directory')

    # Training config
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--pretrain_epochs', type=int, default=50,
                       help='Number of pre-training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=100,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.7,
                       help='Dropout rate')

    # Strategy
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Freeze backbone during initial fine-tuning')
    parser.add_argument('--unfreeze_epoch', type=int, default=20,
                       help='Epoch to unfreeze backbone')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                       help='Use Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal Loss gamma')
    parser.add_argument('--use_mixup', action='store_true', default=True,
                       help='Use MixUp augmentation')
    parser.add_argument('--use_cutmix', action='store_true', default=True,
                       help='Use CutMix augmentation')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')

    # Mode
    parser.add_argument('--skip_pretrain', action='store_true',
                       help='Skip pre-training phase')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pretrained model (if skipping pre-training)')

    args = parser.parse_args()

    # Create config dictionary
    config = vars(args)

    # Print configuration
    print("\n" + "="*80)
    print("TRANSFER LEARNING TRAINING PIPELINE")
    print("="*80)
    print(f"\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key:25s}: {value}")
    print("="*80)

    # Initialize trainer
    trainer = TransferLearningTrainer(config)

    # Phase 1: Pre-train (if not skipped)
    pretrained_path = config.get('pretrained_model')
    if not config.get('skip_pretrain'):
        pretrained_path = trainer.pretrain(config['synthetic_path'])

    # Phase 2: Fine-tune
    trainer.finetune(config['real_path'], pretrained_path)

    print("\n🎉 Training pipeline complete!")


if __name__ == "__main__":
    main()
