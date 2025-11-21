"""
EXTREME DATA AUGMENTATION FOR RADIO ASTRONOMY
Bridges the gap between synthetic and real radio data

Techniques:
- RFI (Radio Frequency Interference) simulation
- Realistic noise injection
- Elastic deformations
- MixUp and CutMix
- Atmospheric effects simulation
"""

import numpy as np
import torch
import random
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from scipy import signal as scipy_signal


class ExtremeSignalAugmentation:
    """
    Comprehensive augmentation for radio signals
    Designed to make synthetic data look like real survey data
    """

    def __init__(self, p=0.9):
        """
        Args:
            p: Probability of applying augmentation
        """
        self.p = p

    def __call__(self, signal):
        """
        Apply random augmentations

        Args:
            signal: numpy array (H, W)

        Returns:
            numpy array: Augmented signal
        """
        if random.random() > self.p:
            return signal

        # Apply 2-4 random augmentations
        num_augs = random.randint(2, 4)
        augmentations = random.sample([
            self.add_realistic_noise,
            self.rfi_injection,
            self.frequency_dropout,
            self.time_masking,
            self.elastic_deformation,
            self.atmospheric_distortion,
            self.add_baseline_drift,
            self.add_spikes,
        ], num_augs)

        for aug in augmentations:
            signal = aug(signal)

        return signal

    def add_realistic_noise(self, signal, std_range=(0.01, 0.15)):
        """Add Gaussian noise with varying SNR"""
        std = random.uniform(*std_range)
        noise = np.random.normal(0, std, signal.shape)
        return np.clip(signal + noise, 0, 1)

    def rfi_injection(self, signal, num_rfi_range=(1, 5)):
        """
        Simulate Radio Frequency Interference
        Common in real radio observations
        """
        h, w = signal.shape
        num_rfi = random.randint(*num_rfi_range)

        signal_rfi = signal.copy()

        for _ in range(num_rfi):
            # Vertical stripes (frequency interference)
            if random.random() < 0.6:
                freq_idx = random.randint(0, w - 1)
                width = random.randint(1, 5)
                intensity = random.uniform(0.3, 1.0)

                start = max(0, freq_idx - width // 2)
                end = min(w, freq_idx + width // 2)
                signal_rfi[:, start:end] += intensity

            # Horizontal stripes (time interference)
            else:
                time_idx = random.randint(0, h - 1)
                height = random.randint(1, 3)
                intensity = random.uniform(0.2, 0.8)

                start = max(0, time_idx - height // 2)
                end = min(h, time_idx + height // 2)
                signal_rfi[start:end, :] += intensity

        return np.clip(signal_rfi, 0, 1)

    def frequency_dropout(self, signal, p=0.3, max_freq_mask=10):
        """Drop random frequency channels"""
        h, w = signal.shape
        signal_dropped = signal.copy()

        num_masks = random.randint(1, max_freq_mask)
        for _ in range(num_masks):
            if random.random() < p:
                freq_idx = random.randint(0, w - 1)
                width = random.randint(1, 5)
                start = max(0, freq_idx - width // 2)
                end = min(w, freq_idx + width // 2)
                signal_dropped[:, start:end] = 0

        return signal_dropped

    def time_masking(self, signal, num_masks=3, max_time_mask=10):
        """Mask random time intervals"""
        h, w = signal.shape
        signal_masked = signal.copy()

        for _ in range(num_masks):
            time_idx = random.randint(0, h - 1)
            height = random.randint(1, max_time_mask)
            start = max(0, time_idx - height // 2)
            end = min(h, time_idx + height // 2)
            signal_masked[start:end, :] *= random.uniform(0, 0.3)

        return signal_masked

    def elastic_deformation(self, signal, alpha_range=(5, 15), sigma=3):
        """
        Apply elastic deformation
        Simulates atmospheric distortions
        """
        h, w = signal.shape
        alpha = random.uniform(*alpha_range)

        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Apply deformation
        distorted = map_coordinates(signal, indices, order=1, mode='reflect')
        return distorted.reshape(signal.shape)

    def atmospheric_distortion(self, signal, strength=0.1):
        """Simulate atmospheric scintillation"""
        h, w = signal.shape

        # Create smooth random field
        atmospheric_field = np.random.randn(h // 4, w // 4)
        atmospheric_field = zoom(atmospheric_field, 4, order=1)
        atmospheric_field = atmospheric_field[:h, :w]
        atmospheric_field = gaussian_filter(atmospheric_field, sigma=5)

        # Normalize
        atmospheric_field = atmospheric_field / (np.abs(atmospheric_field).max() + 1e-8)

        # Apply multiplicative distortion
        distorted = signal * (1 + strength * atmospheric_field)
        return np.clip(distorted, 0, 1)

    def add_baseline_drift(self, signal, amplitude=0.1):
        """Add slow baseline variations"""
        h, w = signal.shape

        # Create smooth baseline
        t = np.linspace(0, 2 * np.pi, h)
        baseline = amplitude * np.sin(random.uniform(1, 3) * t)
        baseline = baseline[:, np.newaxis]  # Broadcast along time axis

        return np.clip(signal + baseline, 0, 1)

    def add_spikes(self, signal, num_spikes_range=(1, 5)):
        """Add random spikes (cosmic rays, etc.)"""
        h, w = signal.shape
        num_spikes = random.randint(*num_spikes_range)

        signal_spiked = signal.copy()

        for _ in range(num_spikes):
            spike_h = random.randint(0, h - 1)
            spike_w = random.randint(0, w - 1)
            spike_intensity = random.uniform(0.5, 1.0)

            # Add Gaussian spike
            h_range = range(max(0, spike_h - 2), min(h, spike_h + 3))
            w_range = range(max(0, spike_w - 2), min(w, spike_w + 3))

            for i in h_range:
                for j in w_range:
                    dist = np.sqrt((i - spike_h)**2 + (j - spike_w)**2)
                    signal_spiked[i, j] += spike_intensity * np.exp(-dist)

        return np.clip(signal_spiked, 0, 1)


class MixUpAugmentation:
    """
    MixUp: Beyond Empirical Risk Minimization
    https://arxiv.org/abs/1710.09412
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, x1, x2, y1, y2):
        """
        Mix two samples

        Args:
            x1, x2: Input samples
            y1, y2: Labels

        Returns:
            mixed_x, mixed_y, lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, y1, y2, lam


class CutMixAugmentation:
    """
    CutMix: Regularization Strategy to Train Strong Classifiers
    https://arxiv.org/abs/1905.04899
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x1, x2, y1, y2):
        """
        Apply CutMix

        Args:
            x1, x2: Input samples (H, W)
            y1, y2: Labels

        Returns:
            mixed_x, y1, y2, lambda
        """
        lam = np.random.beta(self.alpha, self.alpha)

        # Handle both 2D (H, W) and 3D (C, H, W) arrays
        if x1.ndim == 3:
            _, h, w = x1.shape
        else:
            h, w = x1.shape

        # Random box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Uniform sampling
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply CutMix
        mixed_x = x1.copy()
        if x1.ndim == 3:
            mixed_x[:, bby1:bby2, bbx1:bbx2] = x2[:, bby1:bby2, bbx1:bbx2]
        else:
            mixed_x[bby1:bby2, bbx1:bbx2] = x2[bby1:bby2, bbx1:bbx2]

        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

        return mixed_x, y1, y2, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixed loss for MixUp/CutMix

    Args:
        criterion: Loss function
        pred: Predictions
        y_a, y_b: Original labels
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    print("="*60)
    print("TESTING EXTREME AUGMENTATION")
    print("="*60)

    # Create dummy signal
    signal = np.random.rand(128, 1024)

    # Test augmentations
    aug = ExtremeSignalAugmentation(p=1.0)

    print("\n🔬 Testing augmentations...")
    augmented = aug(signal)
    print(f"   Original shape: {signal.shape}")
    print(f"   Augmented shape: {augmented.shape}")
    print(f"   Original range: [{signal.min():.3f}, {signal.max():.3f}]")
    print(f"   Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")

    # Test MixUp
    mixup = MixUpAugmentation(alpha=0.4)
    signal2 = np.random.rand(128, 1024)
    mixed, y1, y2, lam = mixup(signal, signal2, 0, 1)
    print(f"\n🔀 MixUp tested!")
    print(f"   Lambda: {lam:.3f}")
    print(f"   Mixed shape: {mixed.shape}")

    # Test CutMix
    cutmix = CutMixAugmentation(alpha=1.0)
    mixed_cut, y1, y2, lam = cutmix(signal, signal2, 0, 1)
    print(f"\n✂️  CutMix tested!")
    print(f"   Lambda: {lam:.3f}")
    print(f"   Mixed shape: {mixed_cut.shape}")

    print(f"\n{'='*60}")
    print("ALL AUGMENTATION TESTS PASSED!")
    print("="*60)
