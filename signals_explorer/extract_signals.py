"""
SIGNAL DATA EXTRACTOR
Extract and export signals from HDF5 to various formats

Run: python extract_signals.py
"""

import h5py
import numpy as np
import json
import os
from tqdm import tqdm

print("""
╔══════════════════════════════════════════════════════════╗
║   SIGNAL DATA EXTRACTOR                                  ║
║   Export signals to CSV, NPY, TXT formats                ║
╚══════════════════════════════════════════════════════════╝
""")

# Configuration
DATASET_PATH = '/Users/kunalajgaonkar/Desktop/deep_learning_app/radio_vision_dataset_10k'
OUTPUT_DIR = 'extracted_signals'


def extract_all_signals_to_csv(signals, metadata, output_dir):
    """Extract all signals to individual CSV files"""
    
    print(f"\n📤 Extracting {len(signals)} signals to CSV...")
    
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    for idx in tqdm(range(len(signals)), desc="Extracting"):
        signal = signals[idx]
        meta = metadata[idx]
        
        filename = f'{meta["object_type"]}_{idx:05d}.csv'
        filepath = os.path.join(csv_dir, filename)
        
        np.savetxt(filepath, signal, delimiter=',', fmt='%.6f')
    
    print(f"✅ Extracted to: {csv_dir}/")


def extract_all_signals_to_npy(signals, metadata, output_dir):
    """Extract all signals to individual NPY files"""
    
    print(f"\n📤 Extracting {len(signals)} signals to NPY...")
    
    npy_dir = os.path.join(output_dir, 'npy')
    os.makedirs(npy_dir, exist_ok=True)
    
    for idx in tqdm(range(len(signals)), desc="Extracting"):
        signal = signals[idx]
        meta = metadata[idx]
        
        filename = f'{meta["object_type"]}_{idx:05d}.npy'
        filepath = os.path.join(npy_dir, filename)
        
        np.save(filepath, signal)
    
    print(f"✅ Extracted to: {npy_dir}/")


def extract_sample_signals(signals, metadata, output_dir, num_samples=10):
    """Extract a few sample signals for inspection"""
    
    print(f"\n📤 Extracting {num_samples} sample signals...")
    
    sample_dir = os.path.join(output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Get random samples
    indices = np.random.choice(len(signals), num_samples, replace=False)
    
    for idx in indices:
        signal = signals[idx]
        meta = metadata[idx]
        
        # Save in multiple formats
        base_name = f'{meta["object_type"]}_{idx:05d}'
        
        # CSV
        csv_path = os.path.join(sample_dir, f'{base_name}.csv')
        np.savetxt(csv_path, signal, delimiter=',', fmt='%.6f')
        
        # NPY
        npy_path = os.path.join(sample_dir, f'{base_name}.npy')
        np.save(npy_path, signal)
        
        # TXT (with metadata)
        txt_path = os.path.join(sample_dir, f'{base_name}.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Sample ID: {idx}\n")
            f.write(f"Object Type: {meta['object_type']}\n")
            f.write(f"Shape: {signal.shape}\n")
            f.write(f"Min: {signal.min():.6f}\n")
            f.write(f"Max: {signal.max():.6f}\n")
            f.write(f"Mean: {signal.mean():.6f}\n")
            f.write(f"\nSignal Data:\n")
            np.savetxt(f, signal, fmt='%.6f')
    
    print(f"✅ Extracted to: {sample_dir}/")
    print(f"   Formats: CSV, NPY, TXT")


def extract_by_object_type(signals, metadata, output_dir):
    """Extract signals organized by object type"""
    
    print(f"\n📤 Extracting signals by object type...")
    
    # Group by type
    types = {}
    for idx, meta in enumerate(metadata):
        obj_type = meta['object_type']
        if obj_type not in types:
            types[obj_type] = []
        types[obj_type].append(idx)
    
    for obj_type, indices in types.items():
        print(f"\n   {obj_type}: {len(indices)} samples")
        
        type_dir = os.path.join(output_dir, 'by_type', obj_type)
        os.makedirs(type_dir, exist_ok=True)
        
        # Extract first 10 of each type
        for i, idx in enumerate(indices[:10]):
            signal = signals[idx]
            
            csv_path = os.path.join(type_dir, f'{obj_type}_{i:03d}.csv')
            np.savetxt(csv_path, signal, delimiter=',', fmt='%.6f')
    
    print(f"✅ Extracted to: {output_dir}/by_type/")


def create_summary_file(signals, metadata, output_dir):
    """Create summary file with all signal statistics"""
    
    print(f"\n📊 Creating summary file...")
    
    summary_path = os.path.join(output_dir, 'summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DATASET SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total signals: {len(signals)}\n")
        f.write(f"Signal shape: {signals.shape[1:]}\n")
        f.write(f"Data type: {signals.dtype}\n\n")
        
        f.write("="*60 + "\n")
        f.write("STATISTICS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Overall Min: {signals.min():.6f}\n")
        f.write(f"Overall Max: {signals.max():.6f}\n")
        f.write(f"Overall Mean: {signals.mean():.6f}\n")
        f.write(f"Overall Std: {signals.std():.6f}\n\n")
        
        # By object type
        types = {}
        for idx, meta in enumerate(metadata):
            obj_type = meta['object_type']
            if obj_type not in types:
                types[obj_type] = []
            types[obj_type].append(idx)
        
        f.write("="*60 + "\n")
        f.write("BY OBJECT TYPE\n")
        f.write("="*60 + "\n\n")
        
        for obj_type, indices in types.items():
            obj_signals = signals[indices]
            f.write(f"{obj_type}:\n")
            f.write(f"  Count: {len(indices)}\n")
            f.write(f"  Min: {obj_signals.min():.6f}\n")
            f.write(f"  Max: {obj_signals.max():.6f}\n")
            f.write(f"  Mean: {obj_signals.mean():.6f}\n")
            f.write(f"  Std: {obj_signals.std():.6f}\n\n")
    
    print(f"✅ Summary saved: {summary_path}")


def main():
    """Main extraction function"""
    
    h5_file = os.path.join(DATASET_PATH, 'signals.h5')
    metadata_file = os.path.join(DATASET_PATH, 'metadata.json')
    
    # Check files exist
    if not os.path.exists(h5_file):
        print(f"❌ File not found: {h5_file}")
        print(f"\n💡 Update DATASET_PATH in the script")
        return
    
    # Load data
    print(f"📂 Loading: {h5_file}")
    with h5py.File(h5_file, 'r') as f:
        signals = f['signals'][:]
    print(f"✅ Loaded {len(signals)} signals")
    
    print(f"\n📄 Loading: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"✅ Loaded metadata")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Show menu
    print("\n" + "="*60)
    print("EXTRACTION OPTIONS")
    print("="*60)
    print("\n1. Extract 10 sample signals (quick)")
    print("2. Extract by object type (10 per type)")
    print("3. Extract ALL signals to CSV (may take time)")
    print("4. Extract ALL signals to NPY (faster)")
    print("5. Create summary file only")
    print("6. Extract everything (all options)")
    
    choice = input("\nYour choice (1-6): ").strip()
    
    if choice == '1':
        extract_sample_signals(signals, metadata, OUTPUT_DIR, 10)
        create_summary_file(signals, metadata, OUTPUT_DIR)
        
    elif choice == '2':
        extract_by_object_type(signals, metadata, OUTPUT_DIR)
        create_summary_file(signals, metadata, OUTPUT_DIR)
        
    elif choice == '3':
        extract_all_signals_to_csv(signals, metadata, OUTPUT_DIR)
        create_summary_file(signals, metadata, OUTPUT_DIR)
        
    elif choice == '4':
        extract_all_signals_to_npy(signals, metadata, OUTPUT_DIR)
        create_summary_file(signals, metadata, OUTPUT_DIR)
        
    elif choice == '5':
        create_summary_file(signals, metadata, OUTPUT_DIR)
        
    elif choice == '6':
        extract_sample_signals(signals, metadata, OUTPUT_DIR, 10)
        extract_by_object_type(signals, metadata, OUTPUT_DIR)
        create_summary_file(signals, metadata, OUTPUT_DIR)
        
        response = input("\nExtract ALL signals? This will take time (y/n): ")
        if response.lower() == 'y':
            extract_all_signals_to_npy(signals, metadata, OUTPUT_DIR)
    
    else:
        print("❌ Invalid choice")
        return
    
    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print(f"\n📂 Contents:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        level = root.replace(OUTPUT_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files) - 5} more files')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()