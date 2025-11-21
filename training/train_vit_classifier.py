"""
ViT/ConvNeXt Classifier Training with comprehensive metrics
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_classifier import ViTClassifier, ConvNeXtClassifier, FocalLoss


class MetricsTracker:
    """Track all training metrics"""
    def __init__(self, class_names):
        self.class_names = class_names
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'per_class_acc': {c: [] for c in class_names}
        }
        self.best_val_acc = 0
        self.all_preds = []
        self.all_labels = []

    def update(self, train_loss, train_acc, val_loss, val_acc, per_class=None):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
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
        axes[0].legend()

        axes[1].plot(self.metrics['train_acc'], label='Train')
        axes[1].plot(self.metrics['val_acc'], label='Val')
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.close()

        # Per-class accuracy
        if any(self.metrics['per_class_acc'].values()):
            plt.figure(figsize=(12, 6))
            for c, accs in self.metrics['per_class_acc'].items():
                if accs:
                    plt.plot(accs, label=c)
            plt.title('Per-Class Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'), dpi=150)
            plt.close()

        # Confusion matrix
        if self.all_preds and self.all_labels:
            cm = confusion_matrix(self.all_labels, self.all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
            plt.close()

            # Classification report
            report = classification_report(self.all_labels, self.all_preds,
                                          target_names=self.class_names)
            with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)

        # Save metrics JSON
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating', leave=False):
            images, labels = images.to(device), labels.to(device)
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

    per_class = {c: class_correct[c] / max(class_total[c], 1) for c in class_names}
    return total_loss / total, correct / total, per_class, all_preds, all_labels


def train_classifier(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(config['train_dir'], transform=train_transform)
    val_dataset = datasets.ImageFolder(config['val_dir'], transform=val_transform)

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32),
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model selection
    model_type = config.get('model', 'convnext')
    if model_type == 'vit':
        model = ViTClassifier(num_classes=len(class_names), embed_dim=384, depth=6, num_heads=6)
    else:
        model = ConvNeXtClassifier(num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('epochs', 50))

    tracker = MetricsTracker(class_names)
    save_dir = config.get('save_dir', 'outputs/classifier')
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0
    epochs = config.get('epochs', 50)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, per_class, preds, labels = validate(model, val_loader, criterion, device, class_names)
        scheduler.step()

        tracker.update(train_loss, train_acc, val_loss, val_acc, per_class)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Per-class: {per_class}")

        if val_acc > best_acc:
            best_acc = val_acc
            tracker.all_preds = preds
            tracker.all_labels = labels
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_names': class_names
            }, os.path.join(save_dir, 'best_model.pth'))

    # Save final metrics
    tracker.save_plots(save_dir)
    print(f"\nTraining complete! Best Val Acc: {best_acc:.4f}")


if __name__ == '__main__':
    config = {
        'train_dir': 'dataset/train/radio',
        'val_dir': 'dataset/val/radio',
        'model': 'convnext',  # 'vit' or 'convnext'
        'batch_size': 32,
        'epochs': 50,
        'lr': 1e-4,
        'save_dir': 'outputs/classifier'
    }

    train_classifier(config)
