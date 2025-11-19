"""
Download galaxy data (SDSS optical + radio)
"""
import requests
import pandas as pd
import os
from tqdm import tqdm

def download_galaxy_catalog(num_samples=1000, output_dir='data/galaxies'):
    """Download SDSS galaxy catalog"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🌀 Downloading galaxy catalog...")
    
    query = f"""
    SELECT TOP {num_samples}
        ra, dec, z as redshift,
        petroMag_r, petroR50_r,
        psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z,
        objid
    FROM PhotoObj
    WHERE type = 3
        AND clean = 1
        AND petroMag_r BETWEEN 14 AND 18
        AND petroR50_r > 2
        AND z > 0.01 AND z < 0.3
    """
    
    url = "http://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
    params = {'cmd': query, 'format': 'csv'}
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        catalog_path = f'{output_dir}/galaxy_catalog.csv'
        with open(catalog_path, 'w') as f:
            f.write(response.text)
        
        df = pd.read_csv(catalog_path)
        print(f"✅ Downloaded {len(df)} galaxies")
        
        # Download images
        download_sdss_images(df, output_dir, object_type='galaxy')
        
        return df
    else:
        print(f"❌ Failed: {response.status_code}")
        return None


def download_sdss_images(df, output_dir, object_type='galaxy'):
    """Download SDSS images for galaxies"""
    
    images_dir = f'{output_dir}/optical'
    os.makedirs(images_dir, exist_ok=True)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ra, dec = row['ra'], row['dec']
        
        url = f"http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        params = {
            'ra': ra,
            'dec': dec,
            'scale': 0.396,
            'width': 256,
            'height': 256
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                img_path = f'{images_dir}/{object_type}_{idx:04d}.jpg'
                with open(img_path, 'wb') as f:
                    f.write(response.content)
        except:
            pass
    
    print(f"✅ Downloaded {len(os.listdir(images_dir))} images")


if __name__ == "__main__":
    df = download_galaxy_catalog(num_samples=1000)
    print("🎉 Galaxy data downloaded!")