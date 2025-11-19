"""
Radio Vision Dataset Generator
Similar to ImageNet + EEG paradigm:
- Radio signals (like EEG) → Astronomical images
- Large-scale dataset (10,000+ samples)
- Multiple frequency bands, noise levels, and augmentations
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import json
import os
from tqdm import tqdm
import random
from scipy import signal as scipy_signal
from scipy.fft import fft2, ifft2, fftshift
import h5py

class RadioSignalGenerator:
    """Generate realistic radio telescope signals"""
    
    def __init__(self, signal_length=1024, num_frequencies=128):
        self.signal_length = signal_length
        self.num_frequencies = num_frequencies
        
    def generate_interferometry_signal(self, object_type, intensity=1.0):
        """Generate realistic radio interferometry signal"""
        t = np.linspace(0, 10, self.signal_length)
        frequencies = np.linspace(0, 5, self.num_frequencies)
        
        # Create 2D frequency-time signal (spectrogram-like)
        signal_2d = np.zeros((self.num_frequencies, self.signal_length))
        
        if object_type == 'spiral_galaxy':
            # Rotating structure shows up as periodic variations
            for i, f in enumerate(frequencies):
                base_freq = 0.5 + i * 0.01
                signal_2d[i] = intensity * np.sin(2 * np.pi * base_freq * t) * np.exp(-((i - 64)**2) / 500)
                # Add rotation signature
                signal_2d[i] += 0.3 * np.sin(2 * np.pi * base_freq * t * 1.5 + i/10)
                
        elif object_type == 'emission_nebula':
            # Diffuse, broad spectrum emission
            for i, f in enumerate(frequencies):
                base_freq = 0.3
                signal_2d[i] = intensity * 0.8 * np.random.normal(0.5, 0.2, self.signal_length)
                # H-alpha line emission (specific frequency)
                if 50 < i < 70:
                    signal_2d[i] += 0.5 * np.ones(self.signal_length)
                    
        elif object_type == 'quasar':
            # Strong point source with jets (highly variable)
            for i, f in enumerate(frequencies):
                base_freq = 1.0
                # Core emission (broad spectrum)
                signal_2d[i] = intensity * 1.2 * np.ones(self.signal_length)
                # Synchrotron radiation (power-law spectrum)
                signal_2d[i] *= (1 + i)**(-0.7)
                # Add variability
                signal_2d[i] += 0.3 * np.sin(2 * np.pi * base_freq * t) * np.random.randn()
                
        elif object_type == 'pulsar':
            # Periodic pulses
            pulse_period = self.signal_length // 8
            for i, f in enumerate(frequencies):
                # Create periodic pulses
                pulses = np.zeros(self.signal_length)
                for j in range(0, self.signal_length, pulse_period):
                    if j + 5 < self.signal_length:
                        pulses[j:j+5] = intensity * 1.5
                signal_2d[i] = pulses
                # Dispersion: higher frequencies arrive earlier
                signal_2d[i] = np.roll(pulses, -int(i/4))
        
        # Add realistic noise
        noise = np.random.normal(0, 0.1, signal_2d.shape)
        signal_2d += noise
        
        # Normalize
        signal_2d = np.clip(signal_2d, 0, None)
        signal_2d = signal_2d / (np.max(signal_2d) + 1e-8)
        
        return signal_2d
    
    def add_atmospheric_noise(self, signal, noise_level=0.2):
        """Add realistic atmospheric interference"""
        # RFI (Radio Frequency Interference)
        rfi = np.random.uniform(0, noise_level, signal.shape)
        rfi_mask = np.random.random(signal.shape) < 0.05  # 5% RFI
        
        # Ionospheric scintillation
        scintillation = noise_level * 0.5 * np.sin(
            2 * np.pi * np.random.uniform(0.1, 0.5) * np.arange(signal.shape[1])
        )
        
        noisy_signal = signal.copy()
        noisy_signal[rfi_mask] += rfi[rfi_mask]
        noisy_signal += scintillation[np.newaxis, :]
        
        return np.clip(noisy_signal, 0, 1)
    
    def signal_to_image(self, signal):
        """Convert signal to RGB image representation"""
        # Normalize to 0-255
        img_array = (signal * 255).astype(np.uint8)
        
        # Create false-color image (typical for radio astronomy)
        img_rgb = np.zeros((signal.shape[0], signal.shape[1], 3), dtype=np.uint8)
        img_rgb[:, :, 0] = img_array * 0.3  # Red
        img_rgb[:, :, 1] = img_array * 0.7  # Green (emphasize)
        img_rgb[:, :, 2] = img_array * 1.0  # Blue (strongest)
        
        return img_rgb


class OpticalImageGenerator:
    """Generate realistic optical astronomical images"""
    
    def __init__(self, image_size=256):
        self.image_size = image_size
        
    def generate_spiral_galaxy(self, brightness=1.0, rotation=0, num_arms=2):
        """Generate spiral galaxy with parameters"""
        img = np.zeros((self.image_size, self.image_size, 3))
        center = self.image_size // 2
        
        # Core
        y, x = np.ogrid[:self.image_size, :self.image_size]
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Bright core
        core = np.exp(-dist_from_center**2 / (10**2))
        
        # Spiral arms
        theta = np.arctan2(y - center, x - center) + rotation
        r = dist_from_center
        
        arms = np.zeros((self.image_size, self.image_size))
        for i in range(num_arms):
            arm_angle = 2 * np.pi * i / num_arms
            spiral = np.sin(3 * theta - r / 20 + arm_angle)
            arms += np.maximum(0, spiral) * np.exp(-r / 80)
        
        # Combine
        galaxy = core + arms * 0.7
        galaxy = galaxy / np.max(galaxy) * brightness
        
        # Add color gradient
        img[:, :, 0] = galaxy * 0.9  # Red (outer regions)
        img[:, :, 1] = galaxy * 0.8  # Green
        img[:, :, 2] = galaxy * 1.0  # Blue (core)
        
        # Add stars
        img = self._add_star_field(img, num_stars=100)
        
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    def generate_emission_nebula(self, brightness=1.0, colors=None):
        """Generate colorful emission nebula"""
        img = np.zeros((self.image_size, self.image_size, 3))
        
        if colors is None:
            colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0)]
        
        # Multiple emission regions
        for color in colors:
            cx = random.randint(self.image_size//4, 3*self.image_size//4)
            cy = random.randint(self.image_size//4, 3*self.image_size//4)
            
            y, x = np.ogrid[:self.image_size, :self.image_size]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            emission = np.exp(-dist**2 / (50**2)) * brightness
            
            for i in range(3):
                img[:, :, i] += emission * color[i]
        
        # Add turbulence
        img = self._add_turbulence(img)
        
        # Add stars
        img = self._add_star_field(img, num_stars=150)
        
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    def generate_quasar(self, brightness=1.5, jet_angle=45):
        """Generate quasar with jets"""
        img = np.zeros((self.image_size, self.image_size, 3))
        center = self.image_size // 2
        
        # Ultra-bright core
        y, x = np.ogrid[:self.image_size, :self.image_size]
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        
        core = np.exp(-dist_from_center**2 / (5**2)) * brightness
        
        # Jets
        angle_rad = np.radians(jet_angle)
        for direction in [angle_rad, angle_rad + np.pi]:
            dx = np.cos(direction)
            dy = np.sin(direction)
            
            for d in range(10, 100, 2):
                jet_x = int(center + d * dx)
                jet_y = int(center + d * dy)
                
                if 0 <= jet_x < self.image_size and 0 <= jet_y < self.image_size:
                    y_jet, x_jet = np.ogrid[:self.image_size, :self.image_size]
                    jet_dist = np.sqrt((x_jet - jet_x)**2 + (y_jet - jet_y)**2)
                    jet = np.exp(-jet_dist**2 / (3**2)) * brightness * (1 - d/100)
                    img[:, :, :] += jet[:, :, np.newaxis] * np.array([0.8, 0.8, 1.0])
        
        # Add core
        img[:, :, :] += core[:, :, np.newaxis] * np.array([1.0, 1.0, 1.0])
        
        # Add diffraction spikes
        img = self._add_diffraction_spikes(img, center, center)
        
        # Add stars
        img = self._add_star_field(img, num_stars=80)
        
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    def generate_pulsar(self, brightness=1.0, disk_angle=30):
        """Generate pulsar with accretion disk"""
        img = np.zeros((self.image_size, self.image_size, 3))
        center = self.image_size // 2
        
        # Neutron star (point source)
        y, x = np.ogrid[:self.image_size, :self.image_size]
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        
        star = np.exp(-dist_from_center**2 / (2**2)) * brightness
        
        # Accretion disk (elliptical)
        angle_rad = np.radians(disk_angle)
        x_rot = (x - center) * np.cos(angle_rad) - (y - center) * np.sin(angle_rad)
        y_rot = (x - center) * np.sin(angle_rad) + (y - center) * np.cos(angle_rad)
        
        disk_dist = np.sqrt(x_rot**2 + (y_rot * 0.3)**2)
        disk = np.maximum(0, 1 - disk_dist / 30) * brightness * 0.7
        disk[disk_dist < 5] = 0  # Inner hole
        
        # Combine
        img[:, :, 0] = star + disk * 0.9
        img[:, :, 1] = star + disk * 0.7
        img[:, :, 2] = star + disk * 1.0
        
        # Add stars
        img = self._add_star_field(img, num_stars=60)
        
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    def _add_star_field(self, img, num_stars=100):
        """Add random stars to image"""
        for _ in range(num_stars):
            x = random.randint(0, self.image_size - 1)
            y = random.randint(0, self.image_size - 1)
            brightness = random.uniform(0.5, 1.0)
            
            img[y, x, :] = brightness
            
            # Add slight blur for larger stars
            if random.random() < 0.3:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.image_size and 0 <= ny < self.image_size:
                            img[ny, nx, :] = brightness * 0.5
        
        return img
    
    def _add_turbulence(self, img, scale=0.1):
        """Add turbulent structure"""
        noise = np.random.randn(self.image_size, self.image_size, 3) * scale
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=3))
        img = np.array(img_pil) / 255.0 + noise
        return img
    
    def _add_diffraction_spikes(self, img, cx, cy):
        """Add diffraction spikes (from telescope)"""
        for angle in [0, 45, 90, 135]:
            angle_rad = np.radians(angle)
            for d in range(5, 50):
                x = int(cx + d * np.cos(angle_rad))
                y = int(cy + d * np.sin(angle_rad))
                
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    intensity = 1.0 * (1 - d/50)
                    img[y, x, :] = np.minimum(1, img[y, x, :] + intensity * 0.5)
        
        return img


class RadioVisionDatasetGenerator:
    """Main dataset generator combining signal and image generation"""
    
    def __init__(self, output_dir='radio_vision_dataset', image_size=256):
        self.output_dir = output_dir
        self.image_size = image_size
        self.signal_gen = RadioSignalGenerator()
        self.optical_gen = OpticalImageGenerator(image_size)
        
        # Object types with variations
        self.object_types = ['spiral_galaxy', 'emission_nebula', 'quasar', 'pulsar']
        
        # Create directories
        os.makedirs(f"{output_dir}/signals", exist_ok=True)
        os.makedirs(f"{output_dir}/optical_images", exist_ok=True)
        os.makedirs(f"{output_dir}/radio_images", exist_ok=True)
        
    def generate_sample(self, object_type, sample_id):
        """Generate one complete sample (signal + images)"""
        # Random parameters for variation
        intensity = random.uniform(0.7, 1.3)
        noise_level = random.uniform(0.1, 0.3)
        
        # Generate radio signal
        signal = self.signal_gen.generate_interferometry_signal(object_type, intensity)
        signal = self.signal_gen.add_atmospheric_noise(signal, noise_level)
        
        # Convert signal to image
        radio_img = self.signal_gen.signal_to_image(signal)
        
        # Generate optical image with variations
        if object_type == 'spiral_galaxy':
            rotation = random.uniform(0, 2*np.pi)
            num_arms = random.choice([2, 3, 4])
            optical_img = self.optical_gen.generate_spiral_galaxy(intensity, rotation, num_arms)
        elif object_type == 'emission_nebula':
            colors = [
                (random.uniform(0.8, 1.0), random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)),
                (random.uniform(0.2, 0.4), random.uniform(0.8, 1.0), random.uniform(0.2, 0.4)),
                (random.uniform(0.2, 0.4), random.uniform(0.2, 0.4), random.uniform(0.8, 1.0))
            ]
            optical_img = self.optical_gen.generate_emission_nebula(intensity, colors[:random.randint(2,3)])
        elif object_type == 'quasar':
            jet_angle = random.uniform(0, 180)
            optical_img = self.optical_gen.generate_quasar(intensity, jet_angle)
        elif object_type == 'pulsar':
            disk_angle = random.uniform(0, 90)
            optical_img = self.optical_gen.generate_pulsar(intensity, disk_angle)
        
        # Metadata
        metadata = {
            'object_type': object_type,
            'intensity': intensity,
            'noise_level': noise_level,
            'signal_shape': signal.shape,
            'image_size': self.image_size
        }
        
        return signal, radio_img, optical_img, metadata
    
    def generate_dataset(self, num_samples=10000, save_signals=True):
        """Generate complete dataset"""
        print(f"🔬 Generating Radio Vision Dataset")
        print(f"📊 Total samples: {num_samples}")
        print(f"📁 Output directory: {self.output_dir}")
        print("=" * 60)
        
        metadata_list = []
        
        # Optional: Save signals to HDF5 for efficiency
        if save_signals:
            hdf5_file = h5py.File(f"{self.output_dir}/signals.h5", 'w')
            signal_dataset = hdf5_file.create_dataset(
                'signals', 
                shape=(num_samples, 128, 1024),
                dtype='float32'
            )
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Sample object type (balanced dataset)
            object_type = self.object_types[i % len(self.object_types)]
            
            # Generate sample
            signal, radio_img, optical_img, metadata = self.generate_sample(object_type, i)
            
            # Save signal
            if save_signals:
                signal_dataset[i] = signal
            
            # Save images
            Image.fromarray(radio_img).save(f"{self.output_dir}/radio_images/{i:06d}.png")
            Image.fromarray(optical_img).save(f"{self.output_dir}/optical_images/{i:06d}.png")
            
            # Save metadata
            metadata['sample_id'] = i
            metadata['radio_image_path'] = f"radio_images/{i:06d}.png"
            metadata['optical_image_path'] = f"optical_images/{i:06d}.png"
            metadata_list.append(metadata)
        
        if save_signals:
            hdf5_file.close()
        
        # Save metadata
        with open(f"{self.output_dir}/metadata.json", 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        # Save dataset statistics
        stats = {
            'total_samples': num_samples,
            'object_types': self.object_types,
            'samples_per_type': num_samples // len(self.object_types),
            'image_size': self.image_size,
            'signal_shape': [128, 1024]
        }
        
        with open(f"{self.output_dir}/dataset_info.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ Dataset generation complete!")
        print(f"📊 Total samples: {num_samples}")
        print(f"📁 Location: {self.output_dir}")
        print(f"💾 Signals: signals.h5 ({num_samples} × 128 × 1024)")
        print(f"🖼️  Images: {num_samples * 2} PNG files")
        print("=" * 60)


if __name__ == "__main__":
    # Generate large-scale dataset
    generator = RadioVisionDatasetGenerator(
        output_dir='radio_vision_dataset_10k',
        image_size=256
    )
    
    # Generate 10,000 samples
    generator.generate_dataset(num_samples=10000, save_signals=True)
    
    print("\n🎉 Radio Vision Dataset ready for training!")
    print("📊 Dataset is similar to ImageNet + EEG paradigm")
    print("📡 Radio signals → 🖼️ Optical images")