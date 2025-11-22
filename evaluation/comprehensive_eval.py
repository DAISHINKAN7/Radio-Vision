"""
Comprehensive Evaluation Script
Includes:
- Per-class metrics
- Normalized confusion matrix
- Reliability diagrams / Expected Calibration Error
- Inference speed benchmarks
- Domain shift analysis
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from scipy.special import softmax
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CalibrationMetrics:
    """Calculate Expected Calibration Error and plot reliability diagrams"""

    @staticmethod
    def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
        """
        Calculate Expected Calibration Error (ECE)

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (confidence scores)
            n_bins: Number of bins for calibration

        Returns:
            ECE score
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences = np.max(y_pred_proba, axis=1)
        predictions = np.argmax(y_pred_proba, axis=1)
        accuracies = (predictions == y_true)

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    @staticmethod
    def plot_reliability_diagram(y_true, y_pred_proba, save_path, n_bins=10):
        """Plot reliability diagram"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences = np.max(y_pred_proba, axis=1)
        predictions = np.argmax(y_pred_proba, axis=1)
        accuracies = (predictions == y_true)

        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                bin_confidences.append(avg_confidence_in_bin)
                bin_accuracies.append(accuracy_in_bin)
                bin_counts.append(np.sum(in_bin))
            else:
                bin_confidences.append(None)
                bin_accuracies.append(None)
                bin_counts.append(0)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Reliability diagram
        bin_centers = (bin_lowers + bin_uppers) / 2
        valid_indices = [i for i, c in enumerate(bin_confidences) if c is not None]
        valid_centers = [bin_centers[i] for i in valid_indices]
        valid_confs = [bin_confidences[i] for i in valid_indices]
        valid_accs = [bin_accuracies[i] for i in valid_indices]

        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.bar(valid_centers, valid_accs, width=1.0/n_bins, alpha=0.5, edgecolor='black', label='Model')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Confidence histogram
        ax2.hist(confidences, bins=n_bins, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


class InferenceSpeedBenchmark:
    """Benchmark inference speed on different devices"""

    @staticmethod
    def benchmark_model(model, input_size=(1, 3, 224, 224), num_runs=100, warmup=10, device='cpu'):
        """
        Benchmark model inference speed

        Args:
            model: Model to benchmark
            input_size: Input tensor size
            num_runs: Number of inference runs
            warmup: Number of warmup runs
            device: Device to run on

        Returns:
            Dictionary with timing statistics
        """
        model = model.to(device)
        model.eval()

        dummy_input = torch.randn(input_size).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device == 'cuda':
                    torch.cuda.synchronize()

                start = time.time()
                _ = model(dummy_input)

                if device == 'cuda':
                    torch.cuda.synchronize()

                times.append(time.time() - start)

        times = np.array(times) * 1000  # Convert to ms

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
            'fps': 1000.0 / np.mean(times)
        }


def comprehensive_evaluation(model, val_loader, class_names, device, save_dir):
    """
    Run comprehensive evaluation

    Args:
        model: Trained model
        val_loader: Validation dataloader
        class_names: List of class names
        device: Device to run on
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 2:
                images, labels = batch
                images = images.to(device)
            elif len(batch) == 3:
                # Multimodal
                radio, optical, labels = batch
                images = (radio.to(device), optical.to(device))
            else:
                raise ValueError("Unexpected batch format")

            labels = labels.to(device)

            if isinstance(images, tuple):
                outputs = model(*images)
            else:
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 1. Classification Report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 2. Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names))
    )

    metrics_dict = {
        'overall_accuracy': np.mean(all_preds == all_labels),
        'per_class': {}
    }

    for i, cls in enumerate(class_names):
        metrics_dict['per_class'][cls] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    # 3. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Normalized Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # 4. Calibration metrics
    print("\n" + "="*60)
    print("CALIBRATION METRICS")
    print("="*60)
    ece = CalibrationMetrics.expected_calibration_error(all_labels, all_probs, n_bins=10)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    CalibrationMetrics.plot_reliability_diagram(
        all_labels, all_probs,
        os.path.join(save_dir, 'reliability_diagram.png'),
        n_bins=10
    )

    metrics_dict['ece'] = float(ece)

    # 5. Inference speed benchmarks
    print("\n" + "="*60)
    print("INFERENCE SPEED BENCHMARKS")
    print("="*60)

    speed_metrics = {}

    # CPU benchmark
    print("Benchmarking on CPU...")
    cpu_stats = InferenceSpeedBenchmark.benchmark_model(model, device='cpu')
    speed_metrics['cpu'] = cpu_stats
    print(f"  Mean: {cpu_stats['mean_ms']:.2f} ms")
    print(f"  FPS: {cpu_stats['fps']:.2f}")

    # GPU benchmark if available
    if torch.cuda.is_available():
        print("Benchmarking on GPU...")
        gpu_stats = InferenceSpeedBenchmark.benchmark_model(model, device='cuda')
        speed_metrics['gpu'] = gpu_stats
        print(f"  Mean: {gpu_stats['mean_ms']:.2f} ms")
        print(f"  FPS: {gpu_stats['fps']:.2f}")

    metrics_dict['inference_speed'] = speed_metrics

    # Save final metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\nAll results saved to: {save_dir}")

    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['mobilenet', 'convnext', 'vit', 'multimodal', 'ensemble'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print("Comprehensive Evaluation Script")
    print("="*60)

    # This is a template - you'll need to load your specific model type here
    print("Note: This script needs to be adapted for your specific model architecture")
    print("See the comprehensive_evaluation() function for usage")
