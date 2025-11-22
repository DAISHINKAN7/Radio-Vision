"""
Improved MobileNetV2 Training with:
- Batch-level MixUp/CutMix
- Mixed Precision Training (AMP)
- Exponential Moving Average (EMA)
- Test-Time Augmentation (TTA)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mobilenet_classifier import MobileNetV2Classifier
from models.improved_augmentation import MixedAugmentation, mixup_criterion
from models.ema import ModelEMA
from models.tta import TestTimeAugmentation


class MetricsTracker:
    def __init__(self, class_names):
        self.class_names = class_names
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'ema_val_acc': [],
            'per_class_acc': {c: [] for c in class_names}
        }
        self.best_val_acc = 0
        self.all_preds = []
        self.all_labels = []

    def update(self, train_loss, train_acc, val_loss, val_acc, ema_val_acc=None, per_class=None):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        if ema_val_acc is not None:
            self.metrics['ema_val_acc'].append(ema_val_acc)
        if per_class:
            for c, acc in per_class.items():
                if c in self.metrics['per_class_acc']:
                    self.metrics['per_class_acc'][c].append(acc)

    def save_plots(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        # Training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.metrics['train_loss'], label='Train')
        axes[0].plot(self.metrics['val_loss'], label='Val')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.metrics['train_acc'], label='Train')
        axes[1].plot(self.metrics['val_acc'], label='Val')
        if self.metrics['ema_val_acc']:
            axes[1].plot(self.metrics['ema_val_acc'], label='EMA Val')
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.close()

        # Per-class accuracy
        if any(self.metrics['per_class_acc'].values()):
            plt.figure(figsize=(12, 6))
            for c, accs in self.metrics['per_class_acc'].items():
                if accs:
                    plt.plot(accs, label=c, linewidth=2)
            plt.title('Per-Class Accuracy Over Training')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'), dpi=150)
            plt.close()

        # Confusion matrix
        if self.all_preds and self.all_labels:
            cm = confusion_matrix(self.all_labels, self.all_preds)

            # Normalized confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0])
            axes[0].set_title('Confusion Matrix (Counts)')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')

            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[1])
            axes[1].set_title('Normalized Confusion Matrix')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('True')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
            plt.close()

            # Classification report
            report = classification_report(self.all_labels, self.all_preds,
                                          target_names=self.class_names, digits=4)
            with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)
            print("\n" + report)

        # Save metrics JSON
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)


def train_epoch(model, loader, criterion, optimizer, scaler, augmenter, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Apply batch-level augmentation
        mixed_images, y_a, y_b, lam = augmenter(images, labels)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        # For accuracy, use the dominant label
        correct += (outputs.argmax(1) == y_a).sum().item() * lam
        correct += (outputs.argmax(1) == y_b).sum().item() * (1 - lam)
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, class_names, use_tta=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}

    tta = TestTimeAugmentation(model, num_augments=8) if use_tta else None

    pbar = tqdm(loader, desc='Validating')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if use_tta:
            outputs = tta.predict(images)
        else:
            outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            c = class_names[label]
            class_total[c] += 1
            if pred == label:
                class_correct[c] += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    per_class = {c: class_correct[c] / max(class_total[c], 1) for c in class_names}
    return total_loss / total, correct / total, per_class, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/mobilenet_improved')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--use_tta', action='store_true', help='Use TTA for final evaluation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"\nClasses: {class_names}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = MobileNetV2Classifier(num_classes=num_classes, pretrained=True, in_channels=3)
    model = model.to(device)

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay, device=device)

    # Augmentation
    augmenter = MixedAugmentation(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        prob=0.5
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # Metrics tracker
    tracker = MetricsTracker(class_names)
    os.makedirs(args.output_dir, exist_ok=True)

    best_acc = 0
    best_ema_acc = 0

    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, augmenter, device)

        # Update EMA
        ema.update(model)

        # Validate with regular model
        val_loss, val_acc, per_class, preds, labels = validate(
            model, val_loader, criterion, device, class_names, use_tta=False
        )

        # Validate with EMA model
        _, ema_val_acc, _, ema_preds, _ = validate(
            ema.module, val_loader, criterion, device, class_names, use_tta=False
        )

        scheduler.step()

        # Update metrics
        tracker.update(train_loss, train_acc, val_loss, val_acc, ema_val_acc, per_class)

        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  EMA Val Acc: {ema_val_acc:.4f}")
        print(f"  Per-class: {per_class}")

        # Save best regular model
        if val_acc > best_acc:
            best_acc = val_acc
            tracker.all_preds = preds
            tracker.all_labels = labels
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_names': class_names
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  → Saved new best model (acc: {val_acc:.4f})")

        # Save best EMA model
        if ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            torch.save({
                'model': ema.state_dict(),
                'epoch': epoch,
                'val_acc': ema_val_acc,
                'class_names': class_names
            }, os.path.join(args.output_dir, 'best_ema_model.pth'))
            print(f"  → Saved new best EMA model (acc: {ema_val_acc:.4f})")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Final evaluation with TTA if requested
    if args.use_tta:
        print("\n" + "="*60)
        print("Running final evaluation with Test-Time Augmentation...")
        print("="*60)

        # Load best EMA model
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_ema_model.pth'))
        model.load_state_dict(checkpoint['model'])

        _, tta_acc, _, tta_preds, tta_labels = validate(
            model, val_loader, criterion, device, class_names, use_tta=True
        )
        print(f"\nTTA Val Accuracy: {tta_acc:.4f}")

        # Update tracker with TTA results
        tracker.all_preds = tta_preds
        tracker.all_labels = tta_labels

    # Save final metrics
    tracker.save_plots(args.output_dir)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Best EMA Val Acc: {best_ema_acc:.4f}")
    if args.use_tta:
        print(f"TTA Val Acc: {tta_acc:.4f}")


if __name__ == '__main__':
    main()
