"""
Training Script for Signal Classifier
- Trains on 10k dataset
- Data augmentation
- Learning rate scheduling
- Model checkpointing
- W&B logging (optional)
- Multi-GPU support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os

# Add models to path
sys.path.append(str(Path(__file__).parent.parent))
from models.signal_classifier import SignalClassifier, LightweightClassifier, preprocess_signal, augment_signal

try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False
    print("⚠️  W&B not available, logging to console only")


class RadioSignalDataset(Dataset):
    """Dataset for radio signals"""
    
    def __init__(self, dataset_path, split='train', train_ratio=0.8, augment=True):
        self.dataset_path = Path(dataset_path)
        self.augment = augment and (split == 'train')
        
        # Load metadata
        with open(self.dataset_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Store path to signals file (don't open yet - causes pickling issues)
        self.signals_path = str(self.dataset_path / 'signals.h5')
        self.signals_file = None
        self.signals = None
        
        # Split dataset
        num_samples = len(self.metadata)
        train_size = int(num_samples * train_ratio)
        
        if split == 'train':
            self.indices = list(range(0, train_size))
        else:  # val
            self.indices = list(range(train_size, num_samples))
        
        # Class mapping
        self.class_to_idx = {
            'spiral_galaxy': 0,
            'emission_nebula': 1,
            'quasar': 2,
            'pulsar': 3
        }
        
        print(f"📊 {split.upper()} dataset: {len(self.indices)} samples")
    
    def _open_hdf5(self):
        """Open HDF5 file (called in each worker process)"""
        if self.signals_file is None:
            self.signals_file = h5py.File(self.signals_path, 'r')
            self.signals = self.signals_file['signals']
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Open HDF5 file if not already open
        self._open_hdf5()
        
        actual_idx = self.indices[idx]
        
        # Get signal
        signal = self.signals[actual_idx]
        
        # Preprocess
        signal = preprocess_signal(signal)
        
        # Augment if training
        if self.augment and np.random.rand() > 0.5:
            signal = augment_signal(signal, augmentation_type='moderate')
        
        # Get label
        object_type = self.metadata[actual_idx]['object_type']
        label = self.class_to_idx[object_type]
        
        # Convert to tensor
        signal_tensor = torch.from_numpy(signal).unsqueeze(0).float()
        
        return signal_tensor, label


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for signals, labels in pbar:
        signals, labels = signals.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * 4
    class_total = [0] * 4
    
    with torch.no_grad():
        for signals, labels in tqdm(val_loader, desc='Validation'):
            signals, labels = signals.to(device), labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class stats
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    # Per-class accuracy
    class_names = ['spiral_galaxy', 'emission_nebula', 'quasar', 'pulsar']
    class_acc = {}
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc[name] = 100. * class_correct[i] / class_total[i]
        else:
            class_acc[name] = 0.0
    
    return val_loss, val_acc, class_acc


def train_classifier(args):
    """Main training function"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Initialize W&B
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project="radio-vision-classifier",
            config=vars(args),
            name=f"{args.model_type}_{args.run_name}"
        )
    
    # Create datasets
    print("📂 Loading datasets...")
    train_dataset = RadioSignalDataset(
        args.dataset_path,
        split='train',
        train_ratio=args.train_ratio,
        augment=True
    )
    
    val_dataset = RadioSignalDataset(
        args.dataset_path,
        split='val',
        train_ratio=args.train_ratio,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print(f"🧠 Creating {args.model_type} model...")
    if args.model_type == 'advanced':
        model = SignalClassifier(num_classes=4, pretrained=args.pretrained)
    else:
        model = LightweightClassifier(num_classes=4)
    
    model = model.to(device)
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n🎓 Starting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, class_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Print results
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"Per-class accuracy:")
        for name, acc in class_acc.items():
            print(f"  {name:20s}: {acc:.2f}%")
        print("=" * 60)
        
        # Log to W&B
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                **{f'class_acc_{k}': v for k, v in class_acc.items()},
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_acc': class_acc
            }
            
            save_path = Path(args.save_dir) / f'{args.model_type}_best.pth'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(checkpoint, save_path)
            print(f"✅ Saved best model: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            break
    
    print(f"\n🎉 Training complete!")
    print(f"✅ Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final save
    final_path = Path(args.save_dir) / f'{args.model_type}_final.pth'
    torch.save(checkpoint, final_path)
    print(f"💾 Saved final model to: {final_path}")
    
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Fix for macOS multiprocessing
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description='Train Signal Classifier')
    
    # Dataset
    parser.add_argument('--dataset-path', type=str, default='radio_vision_dataset_10k',
                      help='Path to dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Ratio of training data')
    
    # Model
    parser.add_argument('--model-type', type=str, default='advanced',
                      choices=['advanced', 'lightweight'],
                      help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained ResNet (for advanced model)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                      help='Early stopping patience')
    
    # System
    parser.add_argument('--num-workers', type=int, default=0,
                      help='Number of data loader workers (0 for Mac, 4+ for Linux/GPU)')
    parser.add_argument('--save-dir', type=str, default='models/checkpoints',
                      help='Directory to save models')
    
    # Logging
    parser.add_argument('--use-wandb', action='store_true',
                      help='Use Weights & Biases logging')
    parser.add_argument('--run-name', type=str, default='',
                      help='Run name for W&B')
    
    args = parser.parse_args()
    
    # Train
    train_classifier(args)