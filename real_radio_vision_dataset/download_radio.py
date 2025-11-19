"""
Download FIRST radio survey data
"""
import requests
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def download_first_radio(catalog_df, output_dir='data/quasars/radio'):
    """
    Download FIRST radio images for coordinates
    
    Args:
        catalog_df: DataFrame with 'ra' and 'dec' columns
        output_dir: Output directory
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("📡 Downloading FIRST radio images...")
    
    for idx, row in tqdm(catalog_df.iterrows(), total=len(catalog_df)):
        ra, dec = row['ra'], row['dec']
        
        # FIRST Cutout Service
        size = 256  # pixels
        
        url = "https://third.ucllnl.org/cgi-bin/firstcutout"
        params = {
            'RA': ra,
            'Dec': dec,
            'Size': size / 60,  # arcminutes
            'Type': 'FITS'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 1000:
                # Save FITS
                fits_path = f'{output_dir}/radio_{idx:04d}.fits'
                with open(fits_path, 'wb') as f:
                    f.write(response.content)
                
                # Convert to PNG
                try:
                    hdu = fits.open(fits_path)
                    data = hdu[0].data
                    
                    # Normalize
                    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                    data = (data * 255).astype(np.uint8)
                    
                    # Save PNG
                    img = Image.fromarray(data)
                    img = img.resize((256, 256))
                    png_path = f'{output_dir}/radio_{idx:04d}.png'
                    img.save(png_path)
                    
                    hdu.close()
                    os.remove(fits_path)  # Clean up FITS
                except:
                    pass
        except:
            pass
    
    num_downloaded = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"✅ Downloaded {num_downloaded} radio images")


if __name__ == "__main__":
    import pandas as pd
    
    # Load catalog
    df = pd.read_csv('data/quasars/quasar_catalog.csv')
    
    # Download radio images
    download_first_radio(df)
    print("🎉 Radio data downloaded!")