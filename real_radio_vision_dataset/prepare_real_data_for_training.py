"""
PREPARE REAL DATA FOR TRAINING
Converts real collected images to HDF5 format for training

Run: python prepare_real_data_for_training.py
"""

import os
import json
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

print("""
╔══════════════════════════════════════════════════════════╗
║   PREPARE REAL DATA FOR TRAINING                         ║
║   Converts radio PNG → signals, copies optical JPG       ║
╚══════════════════════════════════════════════════════════╝
""")

# Configuration
INPUT_DIR = 'real_radio_vision_dataset_complete'  # or 'real_radio_data_comprehensive'
OUTPUT_DIR = INPUT_DIR + '_processed'


def radio_image_to_signal(img_path, target_shape=(128, 1024)):
    """
    Convert radio PNG image to signal array
    
    Args:
        img_path: Path to radio PNG
        target_shape: Target signal shape (128, 1024)
    
    Returns:
        numpy array (128, 1024)
    """
    
    # Load image
    img = Image.open(img_path).convert('L')  # Grayscale
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Resize to target shape
    from scipy.ndimage import zoom
    h, w = img_array.shape
    target_h, target_w = target_shape
    
    zoom_h = target_h / h
    zoom_w = target_w / w
    
    signal = zoom(img_array, (zoom_h, zoom_w), order=1)
    
    # Ensure exact shape
    signal = signal[:target_h, :target_w]
    
    return signal.astype(np.float32)


def prepare_dataset():
    """Convert real data to training format"""
    
    print(f"\n📂 Input directory: {INPUT_DIR}")
    
    # Check if input exists
    metadata_path = os.path.join(INPUT_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"\n❌ Metadata not found: {metadata_path}")
        print(f"   Make sure you've collected data first!")
        print(f"   Run: python collect_all_objects_real.py")
        return
    
    # Load metadata
    print(f"\n📄 Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Found {len(metadata)} samples")
    
    # Count by type
    type_counts = {}
    for item in metadata:
        obj_type = item['object_type']
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
    
    print(f"\n📊 By object type:")
    for obj_type, count in type_counts.items():
        print(f"   {obj_type}: {count}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_dir = os.path.join(OUTPUT_DIR, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Create HDF5 file for signals
    h5_path = os.path.join(OUTPUT_DIR, 'signals.h5')
    
    print(f"\n🔄 Converting data...")
    print(f"   Radio PNG → Signals (HDF5)")
    print(f"   Optical JPG → images/")
    
    with h5py.File(h5_path, 'w') as h5f:
        # Create dataset
        signals_dataset = h5f.create_dataset(
            'signals',
            shape=(len(metadata), 128, 1024),
            dtype=np.float32
        )
        
        # Process each sample
        new_metadata = []
        
        for idx, item in enumerate(tqdm(metadata, desc="Processing")):
            
            # Get paths
            radio_path = os.path.join(INPUT_DIR, item['radio_image_path'])
            optical_path = os.path.join(INPUT_DIR, item['optical_image_path'])
            
            if not os.path.exists(radio_path):
                print(f"\n⚠️  Radio image not found: {radio_path}")
                continue
            
            if not os.path.exists(optical_path):
                print(f"\n⚠️  Optical image not found: {optical_path}")
                continue
            
            try:
                # Convert radio to signal
                signal = radio_image_to_signal(radio_path)
                signals_dataset[idx] = signal
                
                # Copy optical image
                new_optical_name = f'sample_{idx:05d}.jpg'
                new_optical_path = os.path.join(images_dir, new_optical_name)
                shutil.copy(optical_path, new_optical_path)
                
                # Update metadata
                new_item = {
                    'sample_id': idx,
                    'object_type': item['object_type'],
                    'ra': item.get('ra', 0),
                    'dec': item.get('dec', 0),
                    'optical_source': item.get('optical_source', 'unknown'),
                    'radio_source': item.get('radio_source', 'unknown'),
                    'source': item.get('source', 'real_surveys')
                }
                new_metadata.append(new_item)
                
            except Exception as e:
                print(f"\n⚠️  Error processing sample {idx}: {e}")
                continue
    
    # Save new metadata
    new_metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(new_metadata_path, 'w') as f:
        json.dump(new_metadata, f, indent=2)
    
    print(f"\n✅ Conversion complete!")
    print(f"\n📊 Output statistics:")
    print(f"   Total samples: {len(new_metadata)}")
    print(f"   signals.h5: {len(new_metadata)} signals (128×1024 each)")
    print(f"   images/: {len(os.listdir(images_dir))} optical images")
    
    # Final summary
    final_type_counts = {}
    for item in new_metadata:
        obj_type = item['object_type']
        final_type_counts[obj_type] = final_type_counts.get(obj_type, 0) + 1
    
    print(f"\n📊 Final by type:")
    for obj_type, count in final_type_counts.items():
        print(f"   {obj_type}: {count}")
    
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print(f"   ├── signals.h5")
    print(f"   ├── images/")
    print(f"   └── metadata.json")
    
    print(f"\n✅ Ready for training!")
    print(f"\n🚀 Next step:")
    print(f"   python training/train_classifier.py --dataset-path {OUTPUT_DIR}")
    
    return OUTPUT_DIR


if __name__ == "__main__":
    try:
        import scipy
        print("✅ scipy installed")
    except ImportError:
        print("\n❌ scipy not installed!")
        print("   Install: pip install scipy")
        exit(1)
    
    print("\n⚠️  This will create processed training data")
    print(f"   Input: {INPUT_DIR}/")
    print(f"   Output: {OUTPUT_DIR}/")
    print()
    
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        output = prepare_dataset()
        
        if output:
            print(f"\n🎉 Success!")
            print(f"\nYour dataset is ready at: {output}/")
            print(f"\nUpload to RunPod and train!")
    else:
        print("Cancelled")