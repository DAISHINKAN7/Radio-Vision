"""
HDF5 SIGNAL EXPLORER
Explore and visualize signals from signals.h5 file

Run: python explore_signals.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import os

print("""
╔══════════════════════════════════════════════════════════╗
║   HDF5 SIGNAL EXPLORER                                   ║
║   View and analyze signals from your dataset             ║
╚══════════════════════════════════════════════════════════╝
""")

# Configuration
DATASET_PATH = '/Users/kunalajgaonkar/Desktop/deep_learning_app/radio_vision_dataset_10k'  # Change this to your dataset path
H5_FILE = os.path.join(DATASET_PATH, 'signals.h5')
METADATA_FILE = os.path.join(DATASET_PATH, 'metadata.json')


def load_signals(h5_path):
    """Load all signals from HDF5 file"""
    print(f"📂 Loading: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # List all datasets
        print("\n📊 Datasets in file:")
        for key in f.keys():
            print(f"   - {key}: {f[key].shape}, dtype: {f[key].dtype}")
        
        # Load signals
        signals = f['signals'][:]
        
    print(f"\n✅ Loaded signals!")
    print(f"   Shape: {signals.shape}")
    print(f"   Type: {signals.dtype}")
    print(f"   Size: {signals.nbytes / 1024 / 1024:.2f} MB")
    
    return signals


def load_metadata(metadata_path):
    """Load metadata JSON"""
    print(f"\n📄 Loading: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Loaded metadata for {len(metadata)} samples")
    
    return metadata


def analyze_signals(signals):
    """Analyze signal statistics"""
    print("\n" + "="*60)
    print("SIGNAL ANALYSIS")
    print("="*60)
    
    print(f"\n📊 Basic Statistics:")
    print(f"   Number of signals: {signals.shape[0]}")
    print(f"   Signal shape: {signals.shape[1:]} (height × width)")
    print(f"   Total pixels per signal: {signals.shape[1] * signals.shape[2]}")
    
    print(f"\n📈 Value Statistics:")
    print(f"   Min value: {signals.min():.6f}")
    print(f"   Max value: {signals.max():.6f}")
    print(f"   Mean value: {signals.mean():.6f}")
    print(f"   Std deviation: {signals.std():.6f}")
    
    print(f"\n💾 Memory:")
    print(f"   Total size: {signals.nbytes / 1024 / 1024:.2f} MB")
    print(f"   Per signal: {signals[0].nbytes / 1024:.2f} KB")


def view_signal(signals, metadata, sample_id):
    """View a single signal in detail"""
    print("\n" + "="*60)
    print(f"SIGNAL #{sample_id}")
    print("="*60)
    
    signal = signals[sample_id]
    meta = metadata[sample_id]
    
    print(f"\n📊 Metadata:")
    print(f"   Object type: {meta['object_type']}")
    print(f"   Sample ID: {meta['sample_id']}")
    
    print(f"\n📈 Signal Statistics:")
    print(f"   Shape: {signal.shape}")
    print(f"   Min: {signal.min():.6f}")
    print(f"   Max: {signal.max():.6f}")
    print(f"   Mean: {signal.mean():.6f}")
    print(f"   Std: {signal.std():.6f}")
    
    print(f"\n🔢 First 10 values (row 0):")
    print(f"   {signal[0, :10]}")
    
    print(f"\n🔢 Signal data preview (first 5×10):")
    print(signal[:5, :10])
    
    return signal, meta


def visualize_signal(signal, meta, save_path=None):
    """Visualize a single signal"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Signal as grayscale image
    im1 = axes[0].imshow(signal, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Signal #{meta["sample_id"]}: {meta["object_type"]}')
    axes[0].set_xlabel('Frequency channels')
    axes[0].set_ylabel('Time bins')
    plt.colorbar(im1, ax=axes[0], label='Intensity')
    
    # 2. Signal as heatmap
    im2 = axes[1].imshow(signal, cmap='hot', aspect='auto')
    axes[1].set_title('Heatmap View')
    axes[1].set_xlabel('Frequency channels')
    axes[1].set_ylabel('Time bins')
    plt.colorbar(im2, ax=axes[1], label='Intensity')
    
    # 3. Average spectrum
    avg_spectrum = signal.mean(axis=0)
    axes[2].plot(avg_spectrum)
    axes[2].set_title('Average Spectrum')
    axes[2].set_xlabel('Frequency channel')
    axes[2].set_ylabel('Average intensity')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()


def visualize_multiple_signals(signals, metadata, num_samples=6):
    """Visualize multiple signals"""
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
    axes = axes.flatten()
    
    # Show different samples
    indices = np.linspace(0, len(signals)-1, num_samples, dtype=int)
    
    for idx, ax in zip(indices, axes):
        signal = signals[idx]
        meta = metadata[idx]
        
        im = ax.imshow(signal, cmap='viridis', aspect='auto')
        ax.set_title(f'#{idx}: {meta["object_type"]}', fontsize=10)
        ax.set_xlabel('Freq', fontsize=8)
        ax.set_ylabel('Time', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('multiple_signals.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved: multiple_signals.png")
    plt.show()


def compare_object_types(signals, metadata):
    """Compare signals by object type"""
    
    # Group by object type
    types = {}
    for idx, meta in enumerate(metadata):
        obj_type = meta['object_type']
        if obj_type not in types:
            types[obj_type] = []
        types[obj_type].append(idx)
    
    print("\n" + "="*60)
    print("OBJECT TYPE COMPARISON")
    print("="*60)
    
    for obj_type, indices in types.items():
        print(f"\n🌌 {obj_type}: {len(indices)} samples")
    
    # Visualize one of each type
    fig, axes = plt.subplots(1, len(types), figsize=(15, 4))
    
    if len(types) == 1:
        axes = [axes]
    
    for ax, (obj_type, indices) in zip(axes, types.items()):
        # Show first sample of this type
        idx = indices[0]
        signal = signals[idx]
        
        im = ax.imshow(signal, cmap='viridis', aspect='auto')
        ax.set_title(f'{obj_type}\n(Sample #{idx})')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Time')
        plt.colorbar(im, ax=ax, label='Intensity')
    
    plt.tight_layout()
    plt.savefig('object_type_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved: object_type_comparison.png")
    plt.show()


def export_signal_to_csv(signal, output_path='signal_export.csv'):
    """Export a single signal to CSV"""
    np.savetxt(output_path, signal, delimiter=',', fmt='%.6f')
    print(f"💾 Exported to: {output_path}")


def export_signal_to_npy(signal, output_path='signal_export.npy'):
    """Export a single signal to NPY"""
    np.save(output_path, signal)
    print(f"💾 Exported to: {output_path}")


def interactive_explorer(signals, metadata):
    """Interactive signal explorer"""
    
    print("\n" + "="*60)
    print("INTERACTIVE EXPLORER")
    print("="*60)
    
    while True:
        print("\n🎯 Options:")
        print("   1. View specific signal")
        print("   2. View random signal")
        print("   3. View multiple signals")
        print("   4. Compare object types")
        print("   5. Export signal to CSV")
        print("   6. Export signal to NPY")
        print("   7. Show statistics")
        print("   q. Quit")
        
        choice = input("\nYour choice: ").strip()
        
        if choice == 'q':
            break
            
        elif choice == '1':
            sample_id = int(input(f"Enter sample ID (0-{len(signals)-1}): "))
            if 0 <= sample_id < len(signals):
                signal, meta = view_signal(signals, metadata, sample_id)
                visualize_signal(signal, meta, f'signal_{sample_id}.png')
            else:
                print("❌ Invalid sample ID")
                
        elif choice == '2':
            sample_id = np.random.randint(0, len(signals))
            print(f"📊 Random sample: {sample_id}")
            signal, meta = view_signal(signals, metadata, sample_id)
            visualize_signal(signal, meta, f'signal_{sample_id}.png')
            
        elif choice == '3':
            num = int(input("How many signals to view (2-12)? "))
            visualize_multiple_signals(signals, metadata, num)
            
        elif choice == '4':
            compare_object_types(signals, metadata)
            
        elif choice == '5':
            sample_id = int(input(f"Enter sample ID to export (0-{len(signals)-1}): "))
            if 0 <= sample_id < len(signals):
                export_signal_to_csv(signals[sample_id], f'signal_{sample_id}.csv')
            else:
                print("❌ Invalid sample ID")
                
        elif choice == '6':
            sample_id = int(input(f"Enter sample ID to export (0-{len(signals)-1}): "))
            if 0 <= sample_id < len(signals):
                export_signal_to_npy(signals[sample_id], f'signal_{sample_id}.npy')
            else:
                print("❌ Invalid sample ID")
                
        elif choice == '7':
            analyze_signals(signals)
            
        else:
            print("❌ Invalid choice")


def main():
    """Main function"""
    
    # Check if files exist
    if not os.path.exists(H5_FILE):
        print(f"\n❌ File not found: {H5_FILE}")
        print(f"\n💡 Update DATASET_PATH in the script to your dataset location")
        print(f"   Current: {DATASET_PATH}")
        return
    
    if not os.path.exists(METADATA_FILE):
        print(f"\n❌ File not found: {METADATA_FILE}")
        return
    
    # Load data
    signals = load_signals(H5_FILE)
    metadata = load_metadata(METADATA_FILE)
    
    # Quick analysis
    analyze_signals(signals)
    
    # Show first signal
    print("\n" + "="*60)
    print("PREVIEW: First Signal")
    print("="*60)
    signal, meta = view_signal(signals, metadata, 0)
    
    # Ask user what to do
    print("\n" + "="*60)
    print("WHAT WOULD YOU LIKE TO DO?")
    print("="*60)
    print("\n1. Interactive explorer (recommended)")
    print("2. Quick visualization of 6 samples")
    print("3. Compare all object types")
    print("4. Exit")
    
    choice = input("\nYour choice (1-4): ").strip()
    
    if choice == '1':
        interactive_explorer(signals, metadata)
    elif choice == '2':
        visualize_multiple_signals(signals, metadata, 6)
    elif choice == '3':
        compare_object_types(signals, metadata)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    # Check dependencies
    try:
        import h5py
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\n📦 Install with:")
        print("   pip install h5py matplotlib")
        exit(1)
    
    main()