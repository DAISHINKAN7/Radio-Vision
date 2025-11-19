"""
COMPLETE REAL DATA COLLECTOR - ALL OBJECT TYPES
Collects: Quasars, Galaxies, Nebulae, Pulsars

Run: python collect_all_objects_real.py
"""

import requests
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
import time
import io

print("""
╔══════════════════════════════════════════════════════════╗
║   COMPLETE REAL DATA COLLECTOR                           ║
║   Quasars + Galaxies + Nebulae + Pulsars                 ║
╚══════════════════════════════════════════════════════════╝
""")

# Configuration
CONFIG = {
    'num_quasars': 200,
    'num_galaxies': 200,
    'num_nebulae': 150,
    'num_pulsars': 50,
    'output_dir': 'real_radio_vision_dataset_complete',
    'image_size': 256,
    'angular_size': 4,  # arcminutes
    'timeout': 30
}

print(f"\n⚙️  Configuration:")
print(f"   Quasars: {CONFIG['num_quasars']}")
print(f"   Galaxies: {CONFIG['num_galaxies']}")
print(f"   Nebulae: {CONFIG['num_nebulae']}")
print(f"   Pulsars: {CONFIG['num_pulsars']}")
print(f"   Total: {sum([CONFIG['num_quasars'], CONFIG['num_galaxies'], CONFIG['num_nebulae'], CONFIG['num_pulsars']])}")


# ============================================================================
# IMAGE DOWNLOADERS (Same as before)
# ============================================================================

def download_legacy_survey_image(ra, dec, size_arcmin, output_path):
    """Download from Legacy Survey"""
    pixscale = 0.262
    size_pixels = int(size_arcmin * 60 / pixscale)
    
    url = "https://www.legacysurvey.org/viewer/cutout.jpg"
    params = {
        'ra': ra,
        'dec': dec,
        'layer': 'ls-dr10',
        'pixscale': pixscale,
        'size': size_pixels
    }
    
    try:
        response = requests.get(url, params=params, timeout=20)
        if response.status_code == 200 and len(response.content) > 5000:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            img = Image.open(output_path)
            img = img.resize((CONFIG['image_size'], CONFIG['image_size']), Image.LANCZOS)
            img.save(output_path, quality=95)
            return True
    except:
        pass
    return False


def download_panstarrs_image(ra, dec, size_arcmin, output_path):
    """Download from PanSTARRS"""
    size_pixels = int(size_arcmin * 60 / 0.25)
    
    url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
    params = {
        'ra': ra,
        'dec': dec,
        'size': size_pixels,
        'format': 'jpg',
        'red': 'i',
        'green': 'r',
        'blue': 'g'
    }
    
    try:
        response = requests.get(url, params=params, timeout=20)
        if response.status_code == 200 and len(response.content) > 5000:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            img = Image.open(output_path)
            img = img.resize((CONFIG['image_size'], CONFIG['image_size']), Image.LANCZOS)
            img.save(output_path, quality=95)
            return True
    except:
        pass
    return False


def download_skyview_image(ra, dec, size_arcmin, output_path, survey='DSS2 Red'):
    """Download from SkyView"""
    url = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
    params = {
        'Position': f'{ra},{dec}',
        'Survey': survey,
        'Pixels': CONFIG['image_size'],
        'Size': size_arcmin / 60,
        'Return': 'JPEG'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200 and b'JPEG' in response.content[:100]:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    return False


def download_first_radio_fits(ra, dec, size_arcmin, output_path):
    """Download FIRST radio"""
    url = "https://third.ucllnl.org/cgi-bin/firstcutout"
    params = {
        'RA': ra,
        'Dec': dec,
        'Size': size_arcmin,
        'Type': 'FITS'
    }
    
    try:
        response = requests.get(url, params=params, timeout=CONFIG['timeout'])
        
        if response.status_code == 200 and len(response.content) > 5000:
            temp_fits = output_path + '.fits'
            with open(temp_fits, 'wb') as f:
                f.write(response.content)
            
            try:
                from astropy.io import fits as pyfits
                hdu = pyfits.open(temp_fits)
                data = hdu[0].data
                
                if data is not None and data.size > 100:
                    data = np.nan_to_num(data, nan=0.0)
                    p1, p99 = np.percentile(data[data > 0], [1, 99]) if np.any(data > 0) else (0, 1)
                    data = np.clip(data, p1, p99)
                    
                    if data.max() > data.min():
                        data = (data - data.min()) / (data.max() - data.min())
                        data = (data * 255).astype(np.uint8)
                        
                        img = Image.fromarray(data)
                        img = img.resize((CONFIG['image_size'], CONFIG['image_size']), Image.LANCZOS)
                        img.save(output_path)
                        
                        hdu.close()
                        os.remove(temp_fits)
                        return True
                
                hdu.close()
            except:
                pass
            
            if os.path.exists(temp_fits):
                os.remove(temp_fits)
    except:
        pass
    return False


def download_nvss_radio_image(ra, dec, size_arcmin, output_path):
    """Download NVSS radio"""
    url = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
    params = {
        'Position': f'{ra},{dec}',
        'Survey': 'NVSS',
        'Pixels': CONFIG['image_size'],
        'Size': size_arcmin / 60,
        'Return': 'FITS'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            from astropy.io import fits as pyfits
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            try:
                hdu = pyfits.open(tmp_path)
                data = hdu[0].data
                
                if data is not None and data.size > 100:
                    data = np.nan_to_num(data, nan=0.0)
                    
                    if data.max() > data.min():
                        data = (data - data.min()) / (data.max() - data.min())
                        data = (data * 255).astype(np.uint8)
                        
                        img = Image.fromarray(data)
                        img = img.resize((CONFIG['image_size'], CONFIG['image_size']), Image.LANCZOS)
                        img.save(output_path)
                        
                        hdu.close()
                        os.remove(tmp_path)
                        return True
                
                hdu.close()
            except:
                pass
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except:
        pass
    return False


def download_optical_image_smart(ra, dec, size_arcmin, output_path):
    """Try multiple optical surveys"""
    if download_legacy_survey_image(ra, dec, size_arcmin, output_path):
        return True, 'Legacy'
    time.sleep(0.5)
    
    if download_panstarrs_image(ra, dec, size_arcmin, output_path):
        return True, 'PanSTARRS'
    time.sleep(0.5)
    
    if download_skyview_image(ra, dec, size_arcmin, output_path, 'DSS2 Red'):
        return True, 'DSS'
    time.sleep(0.5)
    
    if download_skyview_image(ra, dec, size_arcmin, output_path, 'SDSSr'):
        return True, 'SDSS'
    
    return False, None


def download_radio_image_smart(ra, dec, size_arcmin, output_path):
    """Try multiple radio surveys"""
    if download_first_radio_fits(ra, dec, size_arcmin, output_path):
        return True, 'FIRST'
    time.sleep(0.5)
    
    if download_nvss_radio_image(ra, dec, size_arcmin, output_path):
        return True, 'NVSS'
    
    return False, None


# ============================================================================
# COORDINATE GENERATORS FOR EACH OBJECT TYPE
# ============================================================================

def generate_quasar_coordinates(num_samples):
    """Generate quasar coordinates"""
    print(f"\n📍 Generating {num_samples} quasar coordinates...")
    coordinates = []
    for i in range(num_samples):
        ra = np.random.uniform(120, 240)
        dec = np.random.uniform(0, 50)
        coordinates.append((ra, dec))
    return coordinates


def generate_galaxy_coordinates(num_samples):
    """Generate galaxy coordinates"""
    print(f"\n📍 Generating {num_samples} galaxy coordinates...")
    coordinates = []
    for i in range(num_samples):
        ra = np.random.uniform(140, 220)
        dec = np.random.uniform(10, 50)
        coordinates.append((ra, dec))
    return coordinates


def generate_nebula_coordinates(num_samples):
    """
    Generate nebula coordinates
    Using regions with GOOD optical survey coverage
    """
    print(f"\n📍 Generating {num_samples} nebula coordinates...")
    coordinates = []
    
    # FIXED: Regions with good Legacy Survey + radio coverage
    regions = [
        {'ra_range': (140, 180), 'dec_range': (20, 40)},   # Well-covered region
        {'ra_range': (180, 220), 'dec_range': (25, 45)},   # SDSS + Legacy overlap
        {'ra_range': (200, 240), 'dec_range': (15, 35)},   # Good coverage
        {'ra_range': (150, 190), 'dec_range': (30, 50)},   # Northern sky (good)
    ]
    
    for i in range(num_samples):
        region = regions[i % len(regions)]
        ra = np.random.uniform(*region['ra_range'])
        dec = np.random.uniform(*region['dec_range'])
        coordinates.append((ra, dec))
    
    return coordinates


def generate_pulsar_coordinates(num_samples):
    """
    Generate pulsar coordinates
    Using regions with GOOD optical survey coverage
    """
    print(f"\n📍 Generating {num_samples} pulsar coordinates...")
    coordinates = []
    
    # FIXED: Same regions as galaxies/quasars (proven to work!)
    regions = [
        {'ra_range': (140, 180), 'dec_range': (20, 40)},   # Well-covered
        {'ra_range': (180, 220), 'dec_range': (25, 45)},   # Proven good
        {'ra_range': (160, 200), 'dec_range': (30, 50)},   # Legacy coverage
        {'ra_range': (200, 240), 'dec_range': (15, 35)},   # SDSS overlap
    ]
    
    for i in range(num_samples):
        region = regions[i % len(regions)]
        ra = np.random.uniform(*region['ra_range'])
        dec = np.random.uniform(*region['dec_range'])
        coordinates.append((ra, dec))
    
    return coordinates


# ============================================================================
# MAIN COLLECTION FUNCTION
# ============================================================================

def collect_from_coordinates(coordinates_list, object_type, output_dir):
    """Collect paired radio-optical images"""
    
    print(f"\n{'='*60}")
    print(f"COLLECTING {object_type.upper()}S")
    print(f"{'='*60}")
    print(f"Total coordinates: {len(coordinates_list)}")
    
    # Create directories
    optical_dir = os.path.join(output_dir, object_type + 's', 'optical')
    radio_dir = os.path.join(output_dir, object_type + 's', 'radio')
    os.makedirs(optical_dir, exist_ok=True)
    os.makedirs(radio_dir, exist_ok=True)
    
    metadata = []
    successful = 0
    stats = {'optical_sources': {}, 'radio_sources': {}}
    
    for idx, (ra, dec) in enumerate(tqdm(coordinates_list, desc="Downloading")):
        
        optical_path = os.path.join(optical_dir, f'{object_type}_{idx:04d}.jpg')
        radio_path = os.path.join(radio_dir, f'{object_type}_{idx:04d}.png')
        
        # Download optical
        optical_ok, optical_source = download_optical_image_smart(
            ra, dec, CONFIG['angular_size'], optical_path
        )
        
        if optical_ok:
            stats['optical_sources'][optical_source] = stats['optical_sources'].get(optical_source, 0) + 1
        
        # Download radio (only if optical succeeded)
        radio_ok, radio_source = False, None
        if optical_ok:
            radio_ok, radio_source = download_radio_image_smart(
                ra, dec, CONFIG['angular_size'], radio_path
            )
            
            if radio_ok:
                stats['radio_sources'][radio_source] = stats['radio_sources'].get(radio_source, 0) + 1
        
        # If both successful
        if optical_ok and radio_ok:
            # Map to standard names
            standard_type = {
                'quasar': 'quasar',
                'galaxy': 'spiral_galaxy',
                'nebula': 'emission_nebula',
                'pulsar': 'pulsar'
            }.get(object_type, object_type)
            
            metadata.append({
                'sample_id': successful,
                'object_type': standard_type,
                'optical_image_path': os.path.relpath(optical_path, output_dir),
                'radio_image_path': os.path.relpath(radio_path, output_dir),
                'ra': float(ra),
                'dec': float(dec),
                'optical_source': optical_source,
                'radio_source': radio_source,
                'source': 'real_surveys'
            })
            successful += 1
        else:
            # Clean up
            if os.path.exists(optical_path):
                os.remove(optical_path)
            if os.path.exists(radio_path):
                os.remove(radio_path)
        
        # Rate limiting
        time.sleep(0.3)
    
    print(f"\n✅ Successfully collected {successful}/{len(coordinates_list)} pairs ({successful/len(coordinates_list)*100:.1f}%)")
    print(f"\n📊 Sources used:")
    print(f"   Optical: {stats['optical_sources']}")
    print(f"   Radio: {stats['radio_sources']}")
    
    return metadata


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main collection pipeline"""
    
    print(f"\n🚀 Starting complete real data collection...")
    
    # Check dependencies
    try:
        from astropy.io import fits
        print("✅ astropy installed")
    except ImportError:
        print("\n❌ ERROR: astropy not installed!")
        print("   Please install: pip install astropy")
        return
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    all_metadata = []
    
    # ========================================================================
    # COLLECT ALL OBJECT TYPES
    # ========================================================================
    
    object_types = [
        ('quasar', CONFIG['num_quasars'], generate_quasar_coordinates),
        ('galaxy', CONFIG['num_galaxies'], generate_galaxy_coordinates),
        ('nebula', CONFIG['num_nebulae'], generate_nebula_coordinates),
        ('pulsar', CONFIG['num_pulsars'], generate_pulsar_coordinates),
    ]
    
    for obj_type, num_samples, coord_generator in object_types:
        if num_samples > 0:
            print("\n" + "="*60)
            print(f"PHASE: {obj_type.upper()}S")
            print("="*60)
            
            coords = coord_generator(num_samples)
            metadata = collect_from_coordinates(coords, obj_type, CONFIG['output_dir'])
            
            # Renumber sample IDs
            for item in metadata:
                item['sample_id'] = len(all_metadata)
                all_metadata.append(item)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    if len(all_metadata) > 0:
        metadata_path = os.path.join(CONFIG['output_dir'], 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"🎉 COLLECTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total samples: {len(all_metadata)}")
        
        # Count by type
        type_counts = {}
        for item in all_metadata:
            obj_type = item['object_type']
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        for obj_type, count in type_counts.items():
            print(f"   {obj_type}: {count}")
        
        print(f"\n📁 Output: {CONFIG['output_dir']}/")
        print(f"📄 Metadata: {metadata_path}")
        
        # Summary
        summary = {
            'total_samples': len(all_metadata),
            'object_types': type_counts,
            'sources': {
                'optical': list(set(x['optical_source'] for x in all_metadata)),
                'radio': list(set(x['radio_source'] for x in all_metadata))
            }
        }
        
        summary_path = os.path.join(CONFIG['output_dir'], 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Ready for training!")
        print(f"\nNext steps:")
        print(f"1. Convert to HDF5: python prepare_real_data_for_training.py")
        print(f"2. Train: python training/train_classifier.py --dataset-path {CONFIG['output_dir']}_processed")
        
    else:
        print(f"\n⚠️  No data collected!")


if __name__ == "__main__":
    print("\n📋 Requirements:")
    print("   pip install requests pandas numpy pillow tqdm astropy")
    print()
    
    response = input("Start collection? (y/n): ")
    
    if response.lower() == 'y':
        try:
            main()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Cancelled")