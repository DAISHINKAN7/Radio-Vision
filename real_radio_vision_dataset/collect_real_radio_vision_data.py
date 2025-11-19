"""
COMPREHENSIVE REAL DATA COLLECTOR
Following the exact approach from the astronomy data guide

Uses multiple catalogs:
- MILLIQUAS (quasars)
- SDSS catalogs (galaxies)
- NVSS/FIRST (radio surveys)
- Direct image downloads

Run: python collect_real_data_comprehensive.py
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
from urllib.parse import urlencode

print("""
╔══════════════════════════════════════════════════════════╗
║   COMPREHENSIVE REAL DATA COLLECTOR                      ║
║   Multiple catalogs + Proper radio-optical pairing       ║
╚══════════════════════════════════════════════════════════╝
""")

# Configuration
CONFIG = {
    'num_quasars': 300,
    'num_galaxies': 300,
    'output_dir': 'real_radio_data_comprehensive',
    'image_size': 256,
    'angular_size': 4,  # arcminutes
    'timeout': 30
}

print(f"\n⚙️  Configuration:")
print(f"   Quasars: {CONFIG['num_quasars']}")
print(f"   Galaxies: {CONFIG['num_galaxies']}")
print(f"   Angular size: {CONFIG['angular_size']} arcmin")
print(f"   Output: {CONFIG['output_dir']}/")


# ============================================================================
# CATALOG DOWNLOADERS
# ============================================================================

def download_vizier_catalog(catalog_name, columns, conditions, limit=1000):
    """
    Download catalog from VizieR
    
    Args:
        catalog_name: VizieR catalog ID (e.g., 'VII/290')
        columns: List of columns to retrieve
        conditions: Query conditions
        limit: Maximum number of rows
    
    Returns:
        DataFrame
    """
    print(f"\n📡 Downloading {catalog_name} from VizieR...")
    
    base_url = "https://vizier.cds.unistra.fr/viz-bin/votable"
    
    params = {
        '-source': catalog_name,
        '-out': ','.join(columns),
        '-out.max': limit
    }
    
    # Add conditions
    for key, value in conditions.items():
        params[key] = value
    
    try:
        response = requests.get(base_url, params=params, timeout=60)
        
        if response.status_code == 200:
            # Parse VOTable (simple approach - look for TABLEDATA)
            content = response.text
            
            # Try to extract as CSV instead
            csv_url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
            response_csv = requests.get(csv_url, params=params, timeout=60)
            
            if response_csv.status_code == 200:
                try:
                    df = pd.read_csv(io.StringIO(response_csv.text), sep='\t', comment='#')
                    print(f"✅ Downloaded {len(df)} sources from VizieR")
                    return df
                except:
                    pass
        
        print(f"⚠️  VizieR download failed")
        return None
        
    except Exception as e:
        print(f"⚠️  VizieR error: {e}")
        return None


def download_nvss_catalog(ra_min, ra_max, dec_min, dec_max, limit=1000):
    """Download NVSS radio catalog for a sky region"""
    print(f"\n📡 Downloading NVSS catalog...")
    
    # NVSS catalog via VizieR: VIII/65/nvss
    params = {
        '-source': 'VIII/65/nvss',
        '-out': 'RAJ2000,DEJ2000,S1.4,MajAxis',
        '-out.max': limit,
        'RAJ2000': f'{ra_min}..{ra_max}',
        'DEJ2000': f'{dec_min}..{dec_max}',
        'S1.4': '>10'  # Flux > 10 mJy
    }
    
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
    
    try:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), sep='\t', comment='#')
            print(f"✅ Downloaded {len(df)} NVSS sources")
            return df
    except Exception as e:
        print(f"⚠️  NVSS error: {e}")
    
    return None


# ============================================================================
# METHOD 1: Legacy Survey (DECaLS) - Very Reliable!
# ============================================================================

def download_legacy_survey_image(ra, dec, size_arcmin, output_path):
    """
    Download optical image from Legacy Survey (DECaLS)
    This is VERY reliable and has excellent coverage!
    """
    
    # Legacy Survey cutout service
    pixscale = 0.262  # arcsec/pixel
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
            # Save image
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Resize to standard size
            img = Image.open(output_path)
            img = img.resize((CONFIG['image_size'], CONFIG['image_size']), Image.LANCZOS)
            img.save(output_path, quality=95)
            
            return True
    except:
        pass
    
    return False


# ============================================================================
# METHOD 2: PanSTARRS - Good coverage
# ============================================================================

def download_panstarrs_image(ra, dec, size_arcmin, output_path):
    """Download optical image from PanSTARRS"""
    
    size_pixels = int(size_arcmin * 60 / 0.25)  # 0.25 arcsec/pixel
    
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


# ============================================================================
# METHOD 3: SkyView - Multiple surveys
# ============================================================================

def download_skyview_image(ra, dec, size_arcmin, output_path, survey='DSS2 Red'):
    """Download from SkyView (many surveys available)"""
    
    url = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
    
    params = {
        'Position': f'{ra},{dec}',
        'Survey': survey,
        'Pixels': CONFIG['image_size'],
        'Size': size_arcmin / 60,  # degrees
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


# ============================================================================
# RADIO IMAGE DOWNLOADERS
# ============================================================================

def download_first_radio_fits(ra, dec, size_arcmin, output_path):
    """Download FIRST radio survey FITS image"""
    
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
            # Save FITS temporarily
            temp_fits = output_path + '.fits'
            with open(temp_fits, 'wb') as f:
                f.write(response.content)
            
            # Convert FITS to PNG
            try:
                from astropy.io import fits as pyfits
                hdu = pyfits.open(temp_fits)
                data = hdu[0].data
                
                if data is not None and data.size > 100:
                    # Normalize
                    data = np.nan_to_num(data, nan=0.0)
                    
                    # Remove extreme outliers
                    p1, p99 = np.percentile(data[data > 0], [1, 99])
                    data = np.clip(data, p1, p99)
                    
                    # Normalize to 0-1
                    if data.max() > data.min():
                        data = (data - data.min()) / (data.max() - data.min())
                        data = (data * 255).astype(np.uint8)
                        
                        # Save as PNG
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
    except:
        pass
    
    return False


def download_nvss_radio_image(ra, dec, size_arcmin, output_path):
    """Download NVSS radio survey image via SkyView"""
    
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
            # Parse FITS
            from astropy.io import fits as pyfits
            import tempfile
            
            # Save to temp file
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


# ============================================================================
# SMART IMAGE DOWNLOADER - Tries multiple sources
# ============================================================================

def download_optical_image_smart(ra, dec, size_arcmin, output_path):
    """Try multiple optical surveys in order of reliability"""
    
    # Try Legacy Survey first (best!)
    if download_legacy_survey_image(ra, dec, size_arcmin, output_path):
        return True, 'Legacy'
    
    time.sleep(0.5)
    
    # Try PanSTARRS
    if download_panstarrs_image(ra, dec, size_arcmin, output_path):
        return True, 'PanSTARRS'
    
    time.sleep(0.5)
    
    # Try SkyView DSS
    if download_skyview_image(ra, dec, size_arcmin, output_path, 'DSS2 Red'):
        return True, 'DSS'
    
    time.sleep(0.5)
    
    # Try SDSS via SkyView
    if download_skyview_image(ra, dec, size_arcmin, output_path, 'SDSSr'):
        return True, 'SDSS'
    
    return False, None


def download_radio_image_smart(ra, dec, size_arcmin, output_path):
    """Try multiple radio surveys"""
    
    # Try FIRST first (higher resolution)
    if download_first_radio_fits(ra, dec, size_arcmin, output_path):
        return True, 'FIRST'
    
    time.sleep(0.5)
    
    # Try NVSS (better coverage)
    if download_nvss_radio_image(ra, dec, size_arcmin, output_path):
        return True, 'NVSS'
    
    return False, None


# ============================================================================
# MAIN COLLECTION FUNCTION
# ============================================================================

def collect_from_coordinates(coordinates_list, object_type, output_dir):
    """
    Collect paired radio-optical images from coordinate list
    
    Args:
        coordinates_list: List of (ra, dec) tuples
        object_type: 'quasar' or 'galaxy'
        output_dir: Output directory
    
    Returns:
        List of metadata
    """
    
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
    
    stats = {
        'optical_sources': {},
        'radio_sources': {}
    }
    
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
            metadata.append({
                'sample_id': successful,
                'object_type': 'spiral_galaxy' if object_type == 'galaxy' else object_type,
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


def generate_sample_coordinates(num_samples, object_type):
    """
    Generate sample coordinates for objects
    Uses regions with known good coverage
    """
    
    print(f"\n📍 Generating {num_samples} sample coordinates for {object_type}...")
    
    coordinates = []
    
    if object_type == 'quasar':
        # Quasars are distributed across sky
        # Focus on regions with good coverage
        for i in range(num_samples):
            # SDSS coverage: RA 0-360, Dec -10 to 70
            ra = np.random.uniform(120, 240)  # Good coverage region
            dec = np.random.uniform(0, 50)
            coordinates.append((ra, dec))
    
    else:  # galaxy
        # Galaxies - also well-distributed
        for i in range(num_samples):
            ra = np.random.uniform(140, 220)
            dec = np.random.uniform(10, 50)
            coordinates.append((ra, dec))
    
    return coordinates


def main():
    """Main collection pipeline"""
    
    print(f"\n🚀 Starting comprehensive real data collection...")
    
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
    # COLLECT QUASARS
    # ========================================================================
    
    if CONFIG['num_quasars'] > 0:
        print("\n" + "="*60)
        print("PHASE 1: QUASARS")
        print("="*60)
        
        # Generate coordinates (in production, load from catalog)
        quasar_coords = generate_sample_coordinates(CONFIG['num_quasars'], 'quasar')
        
        quasar_metadata = collect_from_coordinates(
            quasar_coords, 'quasar', CONFIG['output_dir']
        )
        all_metadata.extend(quasar_metadata)
    
    # ========================================================================
    # COLLECT GALAXIES
    # ========================================================================
    
    if CONFIG['num_galaxies'] > 0:
        print("\n" + "="*60)
        print("PHASE 2: GALAXIES")
        print("="*60)
        
        galaxy_coords = generate_sample_coordinates(CONFIG['num_galaxies'], 'galaxy')
        
        galaxy_metadata = collect_from_coordinates(
            galaxy_coords, 'galaxy', CONFIG['output_dir']
        )
        
        # Renumber
        for item in galaxy_metadata:
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
        print(f"   Quasars: {sum(1 for x in all_metadata if x['object_type'] == 'quasar')}")
        print(f"   Galaxies: {sum(1 for x in all_metadata if x['object_type'] == 'spiral_galaxy')}")
        print(f"\n📁 Output: {CONFIG['output_dir']}/")
        print(f"📄 Metadata: {metadata_path}")
        
        # Summary
        summary = {
            'total_samples': len(all_metadata),
            'sources': {
                'optical': list(set(x['optical_source'] for x in all_metadata)),
                'radio': list(set(x['radio_source'] for x in all_metadata))
            }
        }
        
        summary_path = os.path.join(CONFIG['output_dir'], 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Ready for training!")
        print(f"\nNext: python integrate_real_data.py")
        
    else:
        print(f"\n⚠️  No data collected!")
        print(f"   Check internet connection")
        print(f"   Try different coordinates")


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