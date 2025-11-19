"""
TRAIN ADVANCED CLASSIFIER V2
Improved training for better accuracy

Run: python training/train_advanced_v2.py --dataset-path <path>
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.advanced_classifier_v2 import AdvancedClassifierV2, EnsembleClassifier, preprocess_signal_v2


class SignalDataset(Dataset):
    """Dataset for signal classification"""
    
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = Path(dataset_path)
        
        # Load metadata
        with open(self.dataset_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load signals
        with h5py.File(self.dataset_path / 'signals.h5', 'r') as f:
            self.signals = f['signals'][:]
        
        self.transform = transform
        
        # Class mapping
        self.class_to_idx = {
            'spiral_galaxy': 0,
            'emission_nebula': 1,
            'quasar': 2,
            'pulsar': 3
        }
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata[idx]
        signal = self.signals[idx]
        
        # Preprocess signal
        signal_tensor = preprocess_signal_v2(signal, target_size=(224, 224))
        signal_tensor = signal_tensor.squeeze(0)  # Remove batch dim
        
        # Get label
        label = self.class_to_idx[sample['object_type']]
        
        return signal_tensor, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for signals, labels in pbar:
        signals, labels = signals.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * 4
    class_total = [0] * 4
    class_names = ['spiral_galaxy', 'emission_nebula', 'quasar', 'pulsar']
    
    with torch.no_grad():
        for signals, labels in tqdm(dataloader, desc="Validation"):
            signals, labels = signals.to(device), labels.to(device)
            
            logits = model(signals)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class metrics
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    # Per-class accuracy
    class_accuracies = {}
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_accuracies[name] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_accuracies[name] = 0.0
    
    return avg_loss, accuracy, class_accuracies


def train_classifier_v2(args):
    """Main training function"""
    
    print("="*60)
    print("TRAINING ADVANCED CLASSIFIER V2")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load dataset
    print(f"📂 Loading dataset from: {args.dataset_path}")
    full_dataset = SignalDataset(args.dataset_path)
    
    # Train/val split
    train_size = int(args.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"📊 TRAIN dataset: {len(train_dataset)} samples")
    print(f"📊 VAL dataset: {len(val_dataset)} samples")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print(f"🧠 Creating model: {args.model_type}")
    
    if args.model_type == 'advanced_v2':
        model = AdvancedClassifierV2(
            num_classes=4,
            pretrained=args.pretrained,
            dropout=args.dropout
        )
    elif args.model_type == 'ensemble':
        model = EnsembleClassifier(num_classes=4)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer - AdamW with cosine annealing
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Training loop
    print(f"\n🎓 Starting training for {args.epochs} epochs...")
    print("="*60)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, class_accs = validate(
            model, val_loader, criterion, device
        )
        
        # Step scheduler
        scheduler.step()
        
        # Print epoch results
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print("Per-class accuracy:")
        for name, acc in class_accs.items():
            print(f"  {name:20s}: {acc:.2f}%")
        print("="*60)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_path = Path(args.save_dir) / 'checkpoints' / f'{args.model_type}_best.pth'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_accuracies': class_accs
            }, save_path)
            
            print(f"✅ Saved best model: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    final_path = Path(args.save_dir) / 'checkpoints' / f'{args.model_type}_final.pth'
    torch.save(model.state_dict(), final_path)
    
    print(f"\n🎉 Training complete!")
    print(f"✅ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Saved final model to: {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Advanced Classifier V2')
    
    # Data
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to processed dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train/val split ratio')
    
    # Model
    parser.add_argument('--model-type', type=str, default='advanced_v2',
                        choices=['advanced_v2', 'ensemble'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Save
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save models')
    
    args = parser.parse_args()
    
    train_classifier_v2(args)