"""
Train and evaluate ensemble of multiple models
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mobilenet_classifier import MobileNetV2Classifier
from models.convnext_v2_classifier import ConvNeXtV2Classifier
from models.vit_classifier import ViTClassifier
from models.multimodal_fusion import MultiModalFusionClassifier
from models.ensemble import EnsembleClassifier, VotingEnsemble
from models.ema import ModelEMA


@torch.no_grad()
def evaluate_ensemble(ensemble, val_loader, device, class_names):
    """Evaluate ensemble model"""
    correct, total = 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(val_loader, desc='Evaluating'):
        images, labels = images.to(device), labels.to(device)

        if isinstance(ensemble, VotingEnsemble):
            preds = ensemble.predict(images)
        else:
            outputs = ensemble(images)
            preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total

    # Per-class accuracy
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}
    for pred, label in zip(all_preds, all_labels):
        c = class_names[label]
        class_total[c] += 1
        if pred == label:
            class_correct[c] += 1

    per_class = {c: class_correct[c] / max(class_total[c], 1) for c in class_names}

    return acc, per_class


def load_model(model_path, model_type, num_classes):
    """Load a trained model"""
    checkpoint = torch.load(model_path, map_location='cpu')

    if model_type == 'mobilenet':
        model = MobileNetV2Classifier(num_classes=num_classes, pretrained=False, in_channels=3)
    elif model_type == 'convnext':
        model = ConvNeXtV2Classifier(num_classes=num_classes, model_size='base', pretrained=False)
    elif model_type == 'vit':
        model = ViTClassifier(num_classes=num_classes, embed_dim=384, depth=6, num_heads=6)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint['model'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/ensemble')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                       help='Paths to trained model checkpoints')
    parser.add_argument('--model_types', type=str, nargs='+', required=True,
                       help='Types of models (mobilenet, convnext, vit, etc.)')
    parser.add_argument('--ensemble_type', type=str, default='soft_voting',
                       choices=['soft_voting', 'hard_voting', 'learnable'])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    assert len(args.model_paths) == len(args.model_types), \
        "Number of model paths must match number of model types"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Data
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
    class_names = val_dataset.classes
    num_classes = len(class_names)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load individual models
    print("\nLoading individual models...")
    models = []
    individual_accs = []

    for model_path, model_type in zip(args.model_paths, args.model_types):
        print(f"  Loading {model_type} from {model_path}")
        model = load_model(model_path, model_type, num_classes)
        model = model.to(device)
        model.eval()
        models.append(model)

        # Evaluate individual model
        acc, per_class = evaluate_ensemble(VotingEnsemble([model], voting='soft'),
                                          val_loader, device, class_names)
        individual_accs.append(acc)
        print(f"    Accuracy: {acc:.4f}")
        print(f"    Per-class: {per_class}")

    # Create ensemble
    print(f"\nCreating {args.ensemble_type} ensemble...")
    if args.ensemble_type in ['soft_voting', 'hard_voting']:
        voting_type = 'soft' if args.ensemble_type == 'soft_voting' else 'hard'
        ensemble = VotingEnsemble(models, voting=voting_type)
    elif args.ensemble_type == 'learnable':
        ensemble = EnsembleClassifier(models, learnable_weights=True)
        ensemble = ensemble.to(device)
    else:
        raise ValueError(f"Unknown ensemble type: {args.ensemble_type}")

    # Evaluate ensemble
    print("\nEvaluating ensemble...")
    if isinstance(ensemble, VotingEnsemble):
        ensemble_acc, per_class = evaluate_ensemble(ensemble, val_loader, device, class_names)
    else:
        # For learnable ensemble, you might want to train it first
        ensemble_acc, per_class = evaluate_ensemble(ensemble, val_loader, device, class_names)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print("\nIndividual Model Accuracies:")
    for model_type, acc in zip(args.model_types, individual_accs):
        print(f"  {model_type}: {acc:.4f}")

    print(f"\nEnsemble Accuracy: {ensemble_acc:.4f}")
    print(f"Per-class: {per_class}")

    improvement = ensemble_acc - max(individual_accs)
    print(f"\nImprovement over best individual model: {improvement:.4f} ({improvement*100:.2f}%)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        'ensemble_type': args.ensemble_type,
        'individual_models': {
            model_type: {'accuracy': float(acc)}
            for model_type, acc in zip(args.model_types, individual_accs)
        },
        'ensemble_accuracy': float(ensemble_acc),
        'per_class_accuracy': {c: float(a) for c, a in per_class.items()},
        'improvement': float(improvement)
    }

    with open(os.path.join(args.output_dir, 'ensemble_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}/ensemble_results.json")


if __name__ == '__main__':
    main()
