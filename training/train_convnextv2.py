"""
ConvNeXt-V2 Training Script with EMA and AMP
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
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.convnext_v2_classifier import ConvNeXtV2Classifier
from models.ema import ModelEMA
from models.tta import TestTimeAugmentation
from training.train_mobilenet_improved import MetricsTracker, validate


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/convnextv2')
    parser.add_argument('--model_size', type=str, default='base', choices=['tiny', 'base', 'large'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--use_tta', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Data transforms - stronger augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25)
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
    print(f"\nCreating ConvNeXt-V2-{args.model_size} model...")
    model = ConvNeXtV2Classifier(num_classes=num_classes, model_size=args.model_size, pretrained=True)
    model = model.to(device)

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay, device=device)

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

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        ema.update(model)

        val_loss, val_acc, per_class, preds, labels = validate(
            model, val_loader, criterion, device, class_names, use_tta=False
        )
        _, ema_val_acc, _, ema_preds, _ = validate(
            ema.module, val_loader, criterion, device, class_names, use_tta=False
        )

        scheduler.step()

        tracker.update(train_loss, train_acc, val_loss, val_acc, ema_val_acc, per_class)

        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  EMA Val Acc: {ema_val_acc:.4f}")
        print(f"  Per-class: {per_class}")

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

        if ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            torch.save({
                'model': ema.state_dict(),
                'epoch': epoch,
                'val_acc': ema_val_acc,
                'class_names': class_names
            }, os.path.join(args.output_dir, 'best_ema_model.pth'))
            print(f"  → Saved new best EMA model (acc: {ema_val_acc:.4f})")

    # Final evaluation with TTA
    if args.use_tta:
        print("\n" + "="*60)
        print("Running final evaluation with TTA...")
        print("="*60)
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_ema_model.pth'))
        model.load_state_dict(checkpoint['model'])
        _, tta_acc, _, tta_preds, tta_labels = validate(
            model, val_loader, criterion, device, class_names, use_tta=True
        )
        print(f"\nTTA Val Accuracy: {tta_acc:.4f}")
        tracker.all_preds = tta_preds
        tracker.all_labels = tta_labels

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
