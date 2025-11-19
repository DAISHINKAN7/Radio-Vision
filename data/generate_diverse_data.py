"""
IMPROVED DATASET GENERATOR WITH HIGH DIVERSITY
Creates realistic varied signals and images for each object type

This fixes the low-diversity issue by:
1. Random parameters for each sample
2. Realistic variation based on astronomy
3. Different morphologies within each class
4. Noise and artifacts
5. Real astrophysical parameters

Run: python generate_diverse_dataset.py
"""

import numpy as np
from PIL import Image
import h5py
import json
import os
from tqdm import tqdm

print("""
╔══════════════════════════════════════════════════════════╗
║   IMPROVED DIVERSE DATASET GENERATOR                     ║
║   Creates realistic varied astronomical data             ║
╚══════════════════════════════════════════════════════════╝
""")

# Configuration
CONFIG = {
    'num_samples': 2000,  # 500 per class
    'output_dir': 'radio_vision_dataset_diverse',
    'signal_shape': (128, 1024),
    'image_size': 256
}


class AstronomicalObjectGenerator:
    """Generate realistic varied astronomical objects"""
    
    def __init__(self):
        self.signal_shape = CONFIG['signal_shape']
        self.img_size = CONFIG['image_size']
    
    # ================================================================
    # SPIRAL GALAXIES - High Diversity
    # ================================================================
    
    def generate_spiral_galaxy_signal(self, seed):
        """Generate spiral galaxy radio signal with variation"""
        np.random.seed(seed)
        
        signal = np.zeros(self.signal_shape)
        h, w = self.signal_shape
        
        # Random parameters
        center_h = h // 2 + np.random.randint(-10, 10)
        center_w = w // 2 + np.random.randint(-200, 200)
        
        # Galaxy properties - HIGHLY VARIABLE
        brightness = np.random.uniform(0.3, 0.9)  # Different brightnesses
        size = np.random.uniform(30, 80)  # Different sizes
        num_arms = np.random.choice([2, 3, 4])  # Different spiral arms
        arm_width = np.random.uniform(5, 15)
        rotation = np.random.uniform(0, 2*np.pi)
        inclination = np.random.uniform(0, np.pi/3)  # Face-on to edge-on
        
        # Create spiral structure
        for i in range(h):
            for j in range(w):
                y = i - center_h
                x = j - center_w
                
                # Apply inclination (elliptical appearance)
                x_incl = x
                y_incl = y * np.cos(inclination)
                
                dist = np.sqrt(x_incl**2 + y_incl**2)
                angle = np.arctan2(y_incl, x_incl) + rotation
                
                # Spiral pattern with multiple arms
                spiral_angle = num_arms * angle - dist / 20
                spiral = np.sin(spiral_angle)**2
                
                # Radial profile with variation
                profile = brightness * np.exp(-dist / size)
                
                # Combine
                signal[i, j] = profile * (0.3 + 0.7 * spiral)
        
        # Add realistic features
        # Core region
        core_size = np.random.uniform(3, 8)
        for i in range(h):
            for j in range(w):
                y = i - center_h
                x = j - center_w
                dist = np.sqrt(x**2 + y**2)
                if dist < core_size:
                    signal[i, j] += np.random.uniform(0.2, 0.4) * np.exp(-dist)
        
        # Add noise and structure
        noise = np.random.randn(h, w) * 0.02
        signal += noise
        
        # Random hot spots (star formation regions)
        num_hotspots = np.random.randint(2, 8)
        for _ in range(num_hotspots):
            hs_h = np.random.randint(0, h)
            hs_w = np.random.randint(0, w)
            hs_size = np.random.uniform(2, 5)
            for i in range(max(0, hs_h-5), min(h, hs_h+5)):
                for j in range(max(0, hs_w-20), min(w, hs_w+20)):
                    dist = np.sqrt((i-hs_h)**2 + (j-hs_w)**2)
                    if dist < hs_size:
                        signal[i, j] += 0.15 * np.exp(-dist)
        
        return np.clip(signal, 0, 1)
    
    def generate_spiral_galaxy_image(self, seed):
        """Generate spiral galaxy optical image with variation"""
        np.random.seed(seed + 10000)
        
        img = np.zeros((self.img_size, self.img_size, 3))
        center = self.img_size // 2
        
        # Variable parameters
        brightness = np.random.uniform(100, 200)
        size = np.random.uniform(40, 90)
        num_arms = np.random.choice([2, 3, 4])
        arm_color = np.random.choice(['blue', 'mixed', 'red'])
        inclination = np.random.uniform(0, 60)  # degrees
        rotation = np.random.uniform(0, 360)
        bulge_color = np.random.uniform(0.6, 1.0)  # Yellow to red
        
        for i in range(self.img_size):
            for j in range(self.img_size):
                y = (i - center) * np.cos(np.radians(inclination))
                x = j - center
                
                dist = np.sqrt(x**2 + y**2)
                angle = np.arctan2(y, x) + np.radians(rotation)
                
                # Spiral arms
                spiral_angle = num_arms * angle - dist / 15
                spiral = np.sin(spiral_angle)**2
                
                # Radial brightness
                radial = brightness * np.exp(-dist / size)
                
                # Arm brightness
                arm_bright = radial * (0.2 + 0.8 * spiral)
                
                # Color based on distance and arms
                if dist < 15:  # Bulge - yellow/red
                    img[i, j] = [arm_bright, arm_bright * 0.8, arm_bright * bulge_color]
                else:  # Arms - blue/white or red
                    if arm_color == 'blue':
                        img[i, j] = [arm_bright * 0.7, arm_bright * 0.9, arm_bright]
                    elif arm_color == 'red':
                        img[i, j] = [arm_bright, arm_bright * 0.6, arm_bright * 0.4]
                    else:  # mixed
                        img[i, j] = [arm_bright, arm_bright * 0.8, arm_bright * 0.9]
        
        # Add dust lanes (dark regions)
        if np.random.rand() > 0.5:
            num_dust = np.random.randint(1, 3)
            for _ in range(num_dust):
                dust_angle = np.random.uniform(0, 2*np.pi)
                for i in range(self.img_size):
                    for j in range(self.img_size):
                        y = i - center
                        x = j - center
                        dist = np.sqrt(x**2 + y**2)
                        angle = np.arctan2(y, x)
                        if abs(angle - dust_angle) < 0.3 and dist > 20 and dist < 60:
                            img[i, j] *= 0.5
        
        # Add star forming regions (bright blue knots)
        num_sf = np.random.randint(5, 15)
        for _ in range(num_sf):
            sf_i = np.random.randint(20, self.img_size-20)
            sf_j = np.random.randint(20, self.img_size-20)
            sf_bright = np.random.uniform(150, 255)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = sf_i + di, sf_j + dj
                    if 0 <= ni < self.img_size and 0 <= nj < self.img_size:
                        dist = np.sqrt(di**2 + dj**2)
                        if dist < 2:
                            img[ni, nj] = [sf_bright*0.8, sf_bright*0.9, sf_bright]
        
        # Add background stars
        num_stars = np.random.randint(20, 50)
        for _ in range(num_stars):
            star_i = np.random.randint(0, self.img_size)
            star_j = np.random.randint(0, self.img_size)
            star_bright = np.random.uniform(150, 255)
            color_type = np.random.choice(['white', 'blue', 'red', 'yellow'])
            
            if color_type == 'white':
                color = [star_bright, star_bright, star_bright]
            elif color_type == 'blue':
                color = [star_bright*0.7, star_bright*0.9, star_bright]
            elif color_type == 'red':
                color = [star_bright, star_bright*0.5, star_bright*0.3]
            else:  # yellow
                color = [star_bright, star_bright*0.9, star_bright*0.6]
            
            img[star_i, star_j] = color
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    # ================================================================
    # QUASARS - High Diversity
    # ================================================================
    
    def generate_quasar_signal(self, seed):
        """Generate quasar radio signal with variation"""
        np.random.seed(seed)
        
        signal = np.zeros(self.signal_shape)
        h, w = self.signal_shape
        
        center_h = h // 2 + np.random.randint(-5, 5)
        center_w = w // 2 + np.random.randint(-100, 100)
        
        # Quasar properties - HIGHLY VARIABLE
        core_brightness = np.random.uniform(0.6, 1.0)
        core_size = np.random.uniform(3, 10)
        has_jets = np.random.rand() > 0.5
        jet_angle = np.random.uniform(0, 2*np.pi)
        jet_length = np.random.uniform(20, 60)
        jet_width = np.random.uniform(2, 6)
        
        # Core emission (point source)
        for i in range(h):
            for j in range(w):
                y = i - center_h
                x = j - center_w
                dist = np.sqrt(x**2 + y**2)
                
                signal[i, j] = core_brightness * np.exp(-dist / core_size)
        
        # Jets (if present)
        if has_jets:
            jet_brightness = np.random.uniform(0.3, 0.6)
            
            # Two jets in opposite directions
            for direction in [1, -1]:
                jet_x = direction * np.cos(jet_angle)
                jet_y = direction * np.sin(jet_angle)
                
                for length in range(int(jet_length)):
                    jet_h = int(center_h + jet_y * length)
                    jet_w = int(center_w + jet_x * length)
                    
                    if 0 <= jet_h < h and 0 <= jet_w < w:
                        # Jet brightness decreases with distance
                        brightness = jet_brightness * np.exp(-length / jet_length * 2)
                        
                        # Add width to jet
                        for dh in range(-int(jet_width), int(jet_width)+1):
                            for dw in range(-int(jet_width*5), int(jet_width*5)+1):
                                nh, nw = jet_h + dh, jet_w + dw
                                if 0 <= nh < h and 0 <= nw < w:
                                    dist_from_jet = np.sqrt(dh**2 + dw**2)
                                    if dist_from_jet < jet_width:
                                        signal[nh, nw] += brightness * np.exp(-dist_from_jet/jet_width)
        
        # Add host galaxy (faint extended emission)
        if np.random.rand() > 0.3:
            host_size = np.random.uniform(15, 30)
            host_brightness = np.random.uniform(0.1, 0.3)
            for i in range(h):
                for j in range(w):
                    y = i - center_h
                    x = j - center_w
                    dist = np.sqrt(x**2 + y**2)
                    signal[i, j] += host_brightness * np.exp(-dist / host_size)
        
        # Noise
        noise = np.random.randn(h, w) * 0.03
        signal += noise
        
        return np.clip(signal, 0, 1)
    
    def generate_quasar_image(self, seed):
        """Generate quasar optical image with variation"""
        np.random.seed(seed + 10000)
        
        img = np.zeros((self.img_size, self.img_size, 3))
        center = self.img_size // 2
        
        # Variable properties
        core_brightness = np.random.uniform(200, 255)
        core_size = np.random.uniform(2, 5)
        has_host = np.random.rand() > 0.3
        host_brightness = np.random.uniform(50, 120)
        host_size = np.random.uniform(20, 40)
        color_temp = np.random.choice(['hot', 'warm', 'cool'])
        
        # Host galaxy (if present)
        if has_host:
            for i in range(self.img_size):
                for j in range(self.img_size):
                    y = i - center
                    x = j - center
                    dist = np.sqrt(x**2 + y**2)
                    
                    brightness = host_brightness * np.exp(-dist / host_size)
                    
                    # Host is typically yellowish/reddish
                    img[i, j] = [brightness, brightness * 0.8, brightness * 0.6]
        
        # Bright quasar core
        for i in range(self.img_size):
            for j in range(self.img_size):
                y = i - center
                x = j - center
                dist = np.sqrt(x**2 + y**2)
                
                if dist < core_size * 2:
                    brightness = core_brightness * np.exp(-dist / core_size)
                    
                    if color_temp == 'hot':  # Blue-white
                        img[i, j] += [brightness*0.9, brightness*0.95, brightness]
                    elif color_temp == 'warm':  # White
                        img[i, j] += [brightness, brightness, brightness*0.95]
                    else:  # Yellow-white
                        img[i, j] += [brightness, brightness*0.95, brightness*0.8]
        
        # Diffraction spikes
        if np.random.rand() > 0.5:
            spike_length = np.random.randint(20, 40)
            spike_brightness = np.random.uniform(100, 180)
            
            # Horizontal and vertical spikes
            for offset in range(-spike_length, spike_length):
                # Horizontal
                if 0 <= center + offset < self.img_size:
                    brightness = spike_brightness * (1 - abs(offset)/spike_length)
                    img[center, center + offset] += [brightness, brightness, brightness]
                
                # Vertical
                if 0 <= center + offset < self.img_size:
                    brightness = spike_brightness * (1 - abs(offset)/spike_length)
                    img[center + offset, center] += [brightness, brightness, brightness]
        
        # Background stars
        num_stars = np.random.randint(30, 60)
        for _ in range(num_stars):
            star_i = np.random.randint(0, self.img_size)
            star_j = np.random.randint(0, self.img_size)
            if np.sqrt((star_i-center)**2 + (star_j-center)**2) > 20:  # Not too close to quasar
                star_bright = np.random.uniform(100, 200)
                img[star_i, star_j] = [star_bright, star_bright, star_bright*0.9]
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    # ================================================================
    # Continue with EMISSION_NEBULA and PULSAR...
    # (Similar high-diversity implementations)
    # ================================================================
    
    def generate_emission_nebula_signal(self, seed):
        """Generate emission nebula with high variation"""
        np.random.seed(seed)
        signal = np.zeros(self.signal_shape)
        h, w = self.signal_shape
        
        # Highly variable nebula
        num_clouds = np.random.randint(3, 8)
        
        for _ in range(num_clouds):
            center_h = np.random.randint(20, h-20)
            center_w = np.random.randint(100, w-100)
            cloud_size = np.random.uniform(15, 40)
            cloud_brightness = np.random.uniform(0.2, 0.6)
            cloud_shape = np.random.uniform(0.5, 2.0)  # Elongation
            
            for i in range(h):
                for j in range(w):
                    y = (i - center_h) * cloud_shape
                    x = j - center_w
                    dist = np.sqrt(x**2 + y**2)
                    
                    signal[i, j] += cloud_brightness * np.exp(-dist / cloud_size)
        
        # Add filaments
        num_filaments = np.random.randint(2, 5)
        for _ in range(num_filaments):
            start_h = np.random.randint(0, h)
            start_w = np.random.randint(0, w)
            angle = np.random.uniform(0, 2*np.pi)
            length = np.random.randint(20, 60)
            
            for l in range(length):
                fh = int(start_h + l * np.sin(angle))
                fw = int(start_w + l * np.cos(angle))
                if 0 <= fh < h and 0 <= fw < w:
                    signal[fh, fw] += 0.3
        
        signal += np.random.randn(h, w) * 0.02
        return np.clip(signal, 0, 1)
    
    def generate_emission_nebula_image(self, seed):
        """Generate emission nebula optical image with variation"""
        np.random.seed(seed + 10000)
        img = np.zeros((self.img_size, self.img_size, 3))
        
        # Random colors (H-alpha red, OIII green-blue, SII red)
        primary_color = np.random.choice(['red', 'green', 'blue', 'mixed'])
        
        num_clouds = np.random.randint(5, 12)
        for _ in range(num_clouds):
            center_i = np.random.randint(0, self.img_size)
            center_j = np.random.randint(0, self.img_size)
            cloud_size = np.random.uniform(20, 50)
            brightness = np.random.uniform(80, 180)
            
            for i in range(self.img_size):
                for j in range(self.img_size):
                    dist = np.sqrt((i-center_i)**2 + (j-center_j)**2)
                    val = brightness * np.exp(-dist / cloud_size)
                    
                    if primary_color == 'red':
                        img[i, j] += [val, val*0.3, val*0.2]
                    elif primary_color == 'green':
                        img[i, j] += [val*0.4, val, val*0.6]
                    elif primary_color == 'blue':
                        img[i, j] += [val*0.3, val*0.6, val]
                    else:
                        img[i, j] += [val*np.random.rand(), val*np.random.rand(), val*np.random.rand()]
        
        # Stars
        num_stars = np.random.randint(20, 40)
        for _ in range(num_stars):
            star_i = np.random.randint(0, self.img_size)
            star_j = np.random.randint(0, self.img_size)
            star_bright = np.random.uniform(150, 255)
            img[star_i, star_j] = [star_bright, star_bright, star_bright*0.9]
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def generate_pulsar_signal(self, seed):
        """Generate pulsar with periodic variation"""
        np.random.seed(seed)
        signal = np.zeros(self.signal_shape)
        h, w = self.signal_shape
        
        center_h = h // 2 + np.random.randint(-5, 5)
        period = np.random.randint(10, 30)  # Variable period
        brightness = np.random.uniform(0.5, 0.9)
        pulse_width = np.random.uniform(2, 8)
        
        for i in range(h):
            # Periodic pulses
            pulse = brightness * np.exp(-((i % period) - period/2)**2 / pulse_width)
            
            for j in range(w):
                signal[i, j] = pulse
        
        # Add dispersion (frequency-dependent delay)
        for j in range(w):
            delay = int((j / w) * 5)  # Different delays at different frequencies
            signal[:, j] = np.roll(signal[:, j], delay)
        
        signal += np.random.randn(h, w) * 0.03
        return np.clip(signal, 0, 1)
    
    def generate_pulsar_image(self, seed):
        """Generate pulsar optical image"""
        np.random.seed(seed + 10000)
        img = np.zeros((self.img_size, self.img_size, 3))
        center = self.img_size // 2
        
        # Faint point source
        core_brightness = np.random.uniform(120, 180)
        for i in range(self.img_size):
            for j in range(self.img_size):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                if dist < 3:
                    brightness = core_brightness * np.exp(-dist)
                    img[i, j] = [brightness, brightness*0.9, brightness]
        
        # Background stars
        num_stars = np.random.randint(40, 70)
        for _ in range(num_stars):
            star_i = np.random.randint(0, self.img_size)
            star_j = np.random.randint(0, self.img_size)
            star_bright = np.random.uniform(100, 200)
            img[star_i, star_j] = [star_bright, star_bright, star_bright*0.9]
        
        return np.clip(img, 0, 255).astype(np.uint8)


def generate_diverse_dataset():
    """Generate complete diverse dataset"""
    
    print(f"\n🎨 Generating DIVERSE dataset with {CONFIG['num_samples']} samples...")
    print("   Each sample is UNIQUE with realistic variation!")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    images_dir = os.path.join(CONFIG['output_dir'], 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    generator = AstronomicalObjectGenerator()
    
    # Object types
    object_types = ['spiral_galaxy', 'quasar', 'emission_nebula', 'pulsar']
    samples_per_type = CONFIG['num_samples'] // len(object_types)
    
    # Create HDF5 file
    h5_path = os.path.join(CONFIG['output_dir'], 'signals.h5')
    
    with h5py.File(h5_path, 'w') as h5f:
        signals_dataset = h5f.create_dataset(
            'signals',
            shape=(CONFIG['num_samples'], *CONFIG['signal_shape']),
            dtype=np.float32
        )
        
        metadata = []
        sample_id = 0
        
        for obj_type in object_types:
            print(f"\n📊 Generating {samples_per_type} {obj_type}s...")
            
            for i in tqdm(range(samples_per_type), desc=obj_type):
                seed = sample_id * 1000 + i
                
                # Generate signal
                if obj_type == 'spiral_galaxy':
                    signal = generator.generate_spiral_galaxy_signal(seed)
                    img = generator.generate_spiral_galaxy_image(seed)
                elif obj_type == 'quasar':
                    signal = generator.generate_quasar_signal(seed)
                    img = generator.generate_quasar_image(seed)
                elif obj_type == 'emission_nebula':
                    signal = generator.generate_emission_nebula_signal(seed)
                    img = generator.generate_emission_nebula_image(seed)
                else:  # pulsar
                    signal = generator.generate_pulsar_signal(seed)
                    img = generator.generate_pulsar_image(seed)
                
                # Save signal
                signals_dataset[sample_id] = signal
                
                # Save image
                img_filename = f'sample_{sample_id:05d}.jpg'
                img_path = os.path.join(images_dir, img_filename)
                Image.fromarray(img).save(img_path, quality=95)
                
                # Metadata
                metadata.append({
                    'sample_id': sample_id,
                    'object_type': obj_type
                })
                
                sample_id += 1
        
        # Save metadata
        metadata_path = os.path.join(CONFIG['output_dir'], 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Generated {CONFIG['num_samples']} DIVERSE samples!")
    print(f"📁 Output: {CONFIG['output_dir']}/")
    print(f"\n🎯 Key improvements:")
    print(f"   ✅ Every sample is UNIQUE")
    print(f"   ✅ Realistic variation within each class")
    print(f"   ✅ Different sizes, brightnesses, morphologies")
    print(f"   ✅ Random features (jets, arms, colors, etc.)")
    print(f"   ✅ Much better for training!")


if __name__ == "__main__":
    print("\n⚠️  This will create a NEW diverse dataset")
    print("   Much better quality than the original!")
    print()
    
    response = input("Generate diverse dataset? (y/n): ")
    
    if response.lower() == 'y':
        generate_diverse_dataset()
        print("\n🎉 Done! Use this dataset for training!")
    else:
        print("Cancelled")