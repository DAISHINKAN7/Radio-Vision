"""
Download SDSS quasar catalog and images
"""
import requests
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
from tqdm import tqdm

def download_sdss_quasars(num_samples=1000, output_dir='data/quasars'):
    """Download SDSS quasar data"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("📡 Downloading SDSS quasar catalog...")
    
    # SDSS DR16 Quasar catalog
    query = f"""
    SELECT TOP {num_samples}
        ra, dec, z as redshift,
        psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z,
        objid, specobjid
    FROM SpecObj
    WHERE class = 'QSO'
        AND z > 0.1 AND z < 5.0
        AND psfMag_i < 19.0
    """
    
    # SDSS SkyServer
    url = "http://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
    params = {'cmd': query, 'format': 'csv'}
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        # Save catalog
        catalog_path = f'{output_dir}/quasar_catalog.csv'
        with open(catalog_path, 'w') as f:
            f.write(response.text)
        
        df = pd.read_csv(catalog_path)
        print(f"✅ Downloaded {len(df)} quasars")
        
        # Download images
        print("📸 Downloading optical images...")
        download_sdss_images(df, output_dir)
        
        return df
    else:
        print(f"❌ Failed: {response.status_code}")
        return None


def download_sdss_images(df, output_dir):
    """Download SDSS optical images"""
    
    images_dir = f'{output_dir}/optical'
    os.makedirs(images_dir, exist_ok=True)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ra, dec = row['ra'], row['dec']
        
        # SDSS Image cutout service
        scale = 0.396  # arcsec/pixel
        width = height = 256  # pixels
        
        url = f"http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        params = {
            'ra': ra,
            'dec': dec,
            'scale': scale,
            'width': width,
            'height': height
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                img_path = f'{images_dir}/quasar_{idx:04d}.jpg'
                with open(img_path, 'wb') as f:
                    f.write(response.content)
        except:
            pass
    
    print(f"✅ Downloaded {len(os.listdir(images_dir))} images")


if __name__ == "__main__":
    # Download 1000 quasars
    df = download_sdss_quasars(num_samples=1000)
    print("🎉 Quasar data downloaded!")