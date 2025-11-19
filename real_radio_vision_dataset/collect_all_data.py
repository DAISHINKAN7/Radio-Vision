"""
ROBUST REAL RADIO ASTRONOMY DATA COLLECTOR
Handles SDSS API issues and column name variations

Run: python collect_real_data_robust.py
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
║   REAL RADIO ASTRONOMY DATA COLLECTOR (ROBUST)           ║
║   Downloads from SDSS (optical) + FIRST (radio)          ║
╚══════════════════════════════════════════════════════════╝
""")

# Configuration
CONFIG = {
    'num_quasars': 200,  # Reduced to avoid SDSS overload
    'num_galaxies': 200,
    'output_dir': 'real_radio_vision_dataset',
    'image_size': 256,
    'timeout': 30,
    'retry_attempts': 3
}

print("\n⚙️  Configuration:")
print(f"   Quasars: {CONFIG['num_quasars']}")
print(f"   Galaxies: {CONFIG['num_galaxies']}")
print(f"   Output: {CONFIG['output_dir']}/")
print(f"   Image size: {CONFIG['image_size']}×{CONFIG['image_size']}")


def download_sdss_catalog(object_type='quasar', num_samples=200):
    """Download catalog from SDSS with retry logic"""
    
    print(f"\n📡 Downloading {object_type} catalog from SDSS...")
    
    if object_type == 'quasar':
        # Simplified quasar query
        query = f"""
        SELECT TOP {num_samples}
            p.ra, p.dec, s.z as redshift,
            p.psfMag_i, p.objid
        FROM PhotoObj AS p
        JOIN SpecObj AS s ON s.bestobjid = p.objid
        WHERE s.class = 'QSO'
            AND s.z > 0.1 AND s.z < 3.0
            AND p.psfMag_i < 19.0
            AND p.psfMag_i > 0
        """
    else:  # galaxy
        # Simplified galaxy query
        query = f"""
        SELECT TOP {num_samples}
            ra, dec, z as redshift,
            petroMag_r, objid
        FROM PhotoObj
        WHERE type = 3
            AND clean = 1
            AND petroMag_r BETWEEN 14 AND 17
            AND petroMag_r > 0
            AND z > 0.01 AND z < 0.2
        """
    
    url = "http://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
    
    for attempt in range(CONFIG['retry_attempts']):
        try:
            print(f"   Attempt {attempt + 1}/{CONFIG['retry_attempts']}... (may take 30-90 seconds)")
            
            params = {'cmd': query, 'format': 'csv'}
            response = requests.get(url, params=params, timeout=120)
            
            if response.status_code == 200:
                # Parse CSV
                df = pd.read_csv(io.StringIO(response.text))
                
                # Check if we got data
                if len(df) == 0:
                    print(f"   ⚠️  No results returned")
                    continue
                
                # Check for required columns
                if 'ra' not in df.columns or 'dec' not in df.columns:
                    print(f"   ⚠️  Missing columns. Got: {list(df.columns)}")
                    # Try to find alternative column names
                    if 'RA' in df.columns:
                        df = df.rename(columns={'RA': 'ra', 'DEC': 'dec'})
                    else:
                        continue
                
                # Remove rows with NaN coordinates
                df = df.dropna(subset=['ra', 'dec'])
                
                if len(df) > 0:
                    print(f"✅ Downloaded {len(df)} {object_type}s")
                    return df
                else:
                    print(f"   ⚠️  No valid data after cleaning")
                    
            else:
                print(f"   ⚠️  HTTP {response.status_code}")
                if attempt < CONFIG['retry_attempts'] - 1:
                    print(f"   ⏳ Waiting 10 seconds before retry...")
                    time.sleep(10)
                    
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            if attempt < CONFIG['retry_attempts'] - 1:
                print(f"   ⏳ Waiting 10 seconds before retry...")
                time.sleep(10)
    
    print(f"❌ Failed to download {object_type} catalog after {CONFIG['retry_attempts']} attempts")
    return None


def download_optical_image(ra, dec, output_path):
    """Download optical image from SDSS"""
    
    url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
    params = {
        'ra': ra,
        'dec': dec,
        'scale': 0.396,
        'width': CONFIG['image_size'],
        'height': CONFIG['image_size']
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200 and len(response.content) > 1000:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    
    return False


def download_radio_image(ra, dec, output_path):
    """Download radio image from FIRST survey"""
    
    # FIRST Cutout Service
    url = "https://third.ucllnl.org/cgi-bin/firstcutout"
    params = {
        'RA': ra,
        'Dec': dec,
        'Size': 4,  # arcminutes
        'Type': 'FITS'
    }
    
    try:
        response = requests.get(url, params=params, timeout=CONFIG['timeout'])
        
        if response.status_code == 200 and len(response.content) > 1000:
            # Save temporary FITS
            temp_fits = output_path + '.fits'
            with open(temp_fits, 'wb') as f:
                f.write(response.content)
            
            # Convert to PNG
            try:
                from astropy.io import fits as pyfits
                hdu = pyfits.open(temp_fits)
                data = hdu[0].data
                
                if data is not None and data.size > 0:
                    # Normalize
                    data = np.nan_to_num(data, nan=0.0)
                    if data.max() > data.min():
                        data = (data - data.min()) / (data.max() - data.min())
                        data = (data * 255).astype(np.uint8)
                        
                        # Save PNG
                        img = Image.fromarray(data)
                        img = img.resize((CONFIG['image_size'], CONFIG['image_size']), Image.LANCZOS)
                        img.save(output_path)
                        
                        hdu.close()
                        os.remove(temp_fits)
                        return True
                        
                hdu.close()
            except Exception as e:
                pass
            
            if os.path.exists(temp_fits):
                os.remove(temp_fits)
                
    except Exception as e:
        pass
    
    return False


def collect_object_type(object_type, num_samples):
    """Collect data for one object type"""
    
    print(f"\n{'='*60}")
    print(f"COLLECTING {object_type.upper()}S")
    print(f"{'='*60}")
    
    # Create directories
    base_dir = os.path.join(CONFIG['output_dir'], object_type + 's')
    optical_dir = os.path.join(base_dir, 'optical')
    radio_dir = os.path.join(base_dir, 'radio')
    
    os.makedirs(optical_dir, exist_ok=True)
    os.makedirs(radio_dir, exist_ok=True)
    
    # Download catalog
    df = download_sdss_catalog(object_type, num_samples)
    
    if df is None or len(df) == 0:
        print(f"❌ No catalog data for {object_type}")
        return []
    
    # Verify required columns
    required_cols = ['ra', 'dec']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return []
    
    # Save catalog
    catalog_path = os.path.join(base_dir, f'{object_type}_catalog.csv')
    df.to_csv(catalog_path, index=False)
    print(f"💾 Saved catalog: {catalog_path}")
    
    # Download images
    print(f"\n📸 Downloading paired images...")
    print(f"   Estimated time: ~{len(df) * 3} seconds")
    
    metadata = []
    successful = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Progress"):
        try:
            ra = float(row['ra'])
            dec = float(row['dec'])
        except (KeyError, ValueError) as e:
            print(f"\n⚠️  Skipping row {idx}: {e}")
            continue
        
        optical_path = os.path.join(optical_dir, f'{object_type}_{idx:04d}.jpg')
        radio_path = os.path.join(radio_dir, f'{object_type}_{idx:04d}.png')
        
        # Download optical
        optical_ok = download_optical_image(ra, dec, optical_path)
        
        # Download radio (only if optical succeeded)
        radio_ok = False
        if optical_ok:
            radio_ok = download_radio_image(ra, dec, radio_path)
        
        # If both successful, add to metadata
        if optical_ok and radio_ok:
            metadata.append({
                'sample_id': successful,
                'object_type': 'spiral_galaxy' if object_type == 'galaxy' else object_type,
                'optical_image_path': os.path.relpath(optical_path, CONFIG['output_dir']),
                'radio_image_path': os.path.relpath(radio_path, CONFIG['output_dir']),
                'ra': float(ra),
                'dec': float(dec),
                'redshift': float(row.get('redshift', 0)),
                'source': 'SDSS+FIRST'
            })
            successful += 1
        else:
            # Clean up failed downloads
            if os.path.exists(optical_path):
                os.remove(optical_path)
            if os.path.exists(radio_path):
                os.remove(radio_path)
        
        # Rate limiting
        time.sleep(0.2)
    
    print(f"\n✅ Successfully downloaded {successful}/{len(df)} pairs ({successful/len(df)*100:.1f}%)")
    
    return metadata


def main():
    """Main data collection pipeline"""
    
    print(f"\n🚀 Starting data collection...")
    
    # Check dependencies
    try:
        from astropy.io import fits
        print("✅ astropy installed")
    except ImportError:
        print("\n❌ ERROR: astropy not installed!")
        print("   Please install: pip install astropy")
        return
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    all_metadata = []
    
    # Collect quasars
    if CONFIG['num_quasars'] > 0:
        print("\n" + "="*60)
        print("PHASE 1: QUASARS")
        print("="*60)
        quasar_metadata = collect_object_type('quasar', CONFIG['num_quasars'])
        all_metadata.extend(quasar_metadata)
    
    # Collect galaxies
    if CONFIG['num_galaxies'] > 0:
        print("\n" + "="*60)
        print("PHASE 2: GALAXIES")
        print("="*60)
        galaxy_metadata = collect_object_type('galaxy', CONFIG['num_galaxies'])
        
        # Renumber sample IDs
        for item in galaxy_metadata:
            item['sample_id'] = len(all_metadata)
            all_metadata.append(item)
    
    # Save complete metadata
    if len(all_metadata) > 0:
        metadata_path = os.path.join(CONFIG['output_dir'], 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"🎉 DATA COLLECTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total samples: {len(all_metadata)}")
        
        quasar_count = sum(1 for x in all_metadata if x['object_type'] == 'quasar')
        galaxy_count = sum(1 for x in all_metadata if x['object_type'] == 'spiral_galaxy')
        
        print(f"   Quasars: {quasar_count}")
        print(f"   Galaxies: {galaxy_count}")
        print(f"\n📁 Output directory: {CONFIG['output_dir']}/")
        print(f"📄 Metadata file: {metadata_path}")
        
        # Create summary
        summary = {
            'total_samples': len(all_metadata),
            'object_types': {
                'quasar': quasar_count,
                'spiral_galaxy': galaxy_count
            },
            'sources': ['SDSS', 'FIRST'],
            'image_size': CONFIG['image_size']
        }
        
        summary_path = os.path.join(CONFIG['output_dir'], 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary saved: {summary_path}")
        
        print(f"\n✅ Ready to integrate!")
        print(f"\nNext steps:")
        print(f"1. Run: python integrate_real_data.py")
        print(f"2. Train: python training/train_classifier.py --dataset-path real_radio_vision_dataset_processed")
        
    else:
        print(f"\n⚠️  No data collected!")
        print(f"   Possible reasons:")
        print(f"   - SDSS server is down/overloaded")
        print(f"   - Internet connection issues")
        print(f"   - FIRST survey timeout")
        print(f"\n   Try again later or with fewer samples")
    
    return all_metadata


if __name__ == "__main__":
    print("\n📋 Requirements:")
    print("   ✅ pip install requests pandas numpy pillow tqdm astropy")
    print("   ✅ Stable internet connection")
    print("   ⏱️  ~30-60 minutes")
    print()
    
    response = input("Press Enter to start (or Ctrl+C to cancel)...")
    
    try:
        metadata = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()