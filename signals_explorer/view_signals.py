"""
QUICK SIGNAL VIEWER
Simple script to quickly view signals from HDF5 file

Run: python view_signals_simple.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

# Configuration - CHANGE THIS TO YOUR DATASET PATH
DATASET_PATH = '/Users/kunalajgaonkar/Desktop/deep_learning_app/radio_vision_dataset_10k'

print("🔍 Quick Signal Viewer\n")

# Load signals
print(f"📂 Loading: {DATASET_PATH}/signals.h5")
with h5py.File(f'{DATASET_PATH}/signals.h5', 'r') as f:
    signals = f['signals'][:]
    print(f"✅ Loaded {signals.shape[0]} signals")
    print(f"   Shape: {signals.shape}")
    print(f"   Range: [{signals.min():.3f}, {signals.max():.3f}]")

# Load metadata
print(f"\n📄 Loading: {DATASET_PATH}/metadata.json")
with open(f'{DATASET_PATH}/metadata.json', 'r') as f:
    metadata = json.load(f)
    print(f"✅ Loaded metadata for {len(metadata)} samples")

# Show first signal
print("\n" + "="*60)
print("FIRST SIGNAL (Sample #0)")
print("="*60)

signal = signals[0]
meta = metadata[0]

print(f"\nObject type: {meta['object_type']}")
print(f"Signal shape: {signal.shape}")
print(f"Min: {signal.min():.6f}")
print(f"Max: {signal.max():.6f}")
print(f"Mean: {signal.mean():.6f}")

print(f"\nFirst 10 values (row 0):")
print(signal[0, :10])

print(f"\nSignal data (first 5 rows, 10 columns):")
print(signal[:5, :10])

# Visualize
print("\n📊 Creating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Show 6 different samples
for i, ax in enumerate(axes.flatten()):
    idx = i * (len(signals) // 6)
    sample = signals[idx]
    sample_meta = metadata[idx]
    
    im = ax.imshow(sample, cmap='viridis', aspect='auto')
    ax.set_title(f'Sample #{idx}: {sample_meta["object_type"]}')
    ax.set_xlabel('Frequency channels')
    ax.set_ylabel('Time bins')
    plt.colorbar(im, ax=ax, label='Intensity')

plt.tight_layout()
plt.savefig('signals_preview.png', dpi=150, bbox_inches='tight')
print("💾 Saved: signals_preview.png")
plt.show()

print("\n✅ Done!")
print("\nFor more options, run: python explore_signals.py")