"""
Multi-modal Fusion Training Script
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multimodal_fusion import MultiModalFusionClassifier, AttentionFusionClassifier, EarlyFusionClassifier
from models.ema import ModelEMA
from training.train_mobilenet_improved import MetricsTracker


class PairedImageDataset(Dataset):
    """Dataset for paired radio-optical images with labels"""
    def __init__(self, radio_dir, optical_dir, transform=None):
        self.radio_dir = Path(radio_dir)
        self.optical_dir = Path(optical_dir)
        self.transform = transform

        # Get class folders
        self.classes = sorted([d.name for d in self.radio_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Build file list
        self.samples = []
        for cls in self.classes:
            radio_class_dir = self.radio_dir / cls
            optical_class_dir = self.optical_dir / cls

            if not optical_class_dir.exists():
                continue

            radio_files = sorted(list(radio_class_dir.glob('*.png')) + list(radio_class_dir.glob('*.jpg')))
            optical_files = sorted(list(optical_class_dir.glob('*.png')) + list(optical_class_dir.glob('*.jpg')))

            # Match by index
            num_pairs = min(len(radio_files), len(optical_files))
            for i in range(num_pairs):
                self.samples.append({
                    'radio': radio_files[i],
                    'optical': optical_files[i],
                    'label': self.class_to_idx[cls]
                })

        print(f"Found {len(self.samples)} paired samples across {len(self.classes)} classes")
        print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        radio_img = Image.open(sample['radio']).convert('RGB')
        optical_img = Image.open(sample['optical']).convert('RGB')

        if self.transform:
            radio_img = self.transform(radio_img)
            optical_img = self.transform(optical_img)

        return radio_img, optical_img, sample['label']


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc='Training')
    for radio, optical, labels in pbar:
        radio = radio.to(device)
        optical = optical.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(radio, optical)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}

    pbar = tqdm(loader, desc='Validating')
    for radio, optical, labels in pbar:
        radio = radio.to(device)
        optical = optical.to(device)
        labels = labels.to(device)

        outputs = model(radio, optical)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
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
    parser.add_argument('--train_radio_dir', type=str, required=True)
    parser.add_argument('--train_optical_dir', type=str, required=True)
    parser.add_argument('--val_radio_dir', type=str, required=True)
    parser.add_argument('--val_optical_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/multimodal_fusion')
    parser.add_argument('--fusion_type', type=str, default='concat', choices=['concat', 'attention', 'early'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Transforms
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
    train_dataset = PairedImageDataset(args.train_radio_dir, args.train_optical_dir, train_transform)
    val_dataset = PairedImageDataset(args.val_radio_dir, args.val_optical_dir, val_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model selection
    print(f"\nCreating {args.fusion_type} fusion model...")
    if args.fusion_type == 'concat':
        model = MultiModalFusionClassifier(num_classes=num_classes)
    elif args.fusion_type == 'attention':
        model = AttentionFusionClassifier(num_classes=num_classes)
    else:  # early
        model = EarlyFusionClassifier(num_classes=num_classes)
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

        val_loss, val_acc, per_class, preds, labels = validate(model, val_loader, criterion, device, class_names)
        _, ema_val_acc, _, ema_preds, _ = validate(ema.module, val_loader, criterion, device, class_names)

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

    tracker.save_plots(args.output_dir)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Best EMA Val Acc: {best_ema_acc:.4f}")


if __name__ == '__main__':
    main()
