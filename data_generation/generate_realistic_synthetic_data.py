"""
ADVANCED REALISTIC SYNTHETIC DATA GENERATOR
Creates synthetic data with real-world imperfections:
- Radio Frequency Interference (RFI)
- Atmospheric noise and scintillation
- Baseline variations
- Signal artifacts
- Realistic SNR variations
- Incomplete/corrupted data

This produces synthetic data that is NOT 100% perfect,
making transfer learning more effective.

Author: Radio Vision Team
"""

import numpy as np
import h5py
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import butter, filtfilt
import os
import warnings
warnings.filterwarnings('ignore')

print("""
╔════════════════════════════════════════════════════════════╗
║   ADVANCED REALISTIC SYNTHETIC DATA GENERATOR              ║
║   Produces imperfect synthetic data for better transfer    ║
╚════════════════════════════════════════════════════════════╝
""")

# Configuration
CONFIG = {
    # Dataset size
    'num_spiral_galaxies': 800,
    'num_emission_nebulae': 800,
    'num_quasars': 800,
    'num_pulsars': 800,
    
    # Output
    'output_dir': 'realistic_synthetic_dataset',
    'image_size': 256,
    'signal_shape': (128, 1024),  # freq x time
    
    # Realism parameters
    'noise_level': 0.15,           # Higher = more noise
    'rfi_probability': 0.4,        # 40% of samples have RFI
    'corruption_probability': 0.1, # 10% have some corruption
    'snr_range': (5, 25),         # Signal-to-noise ratio in dB
    
    # Quality variation
    'quality_distribution': {
        'excellent': 0.2,   # 20% high quality
        'good': 0.5,        # 50% medium quality
        'poor': 0.3         # 30% low quality (realistic!)
    }
}

print("\n⚙️  Configuration:")
print(f"   Total samples: {sum([CONFIG[k] for k in CONFIG if k.startswith('num_')])}")
print(f"   Noise level: {CONFIG['noise_level']}")
print(f"   RFI probability: {CONFIG['rfi_probability']}")
print(f"   Quality distribution: Excellent={CONFIG['quality_distribution']['excellent']}, "
      f"Good={CONFIG['quality_distribution']['good']}, "
      f"Poor={CONFIG['quality_distribution']['poor']}")


class RealisticSignalGenerator:
    """Generate realistic radio signals with imperfections"""
    
    def __init__(self, signal_shape=(128, 1024), noise_level=0.15):
        self.freq_bins, self.time_bins = signal_shape
        self.noise_level = noise_level
    
    def generate_spiral_galaxy(self, quality='good'):
        """Generate spiral galaxy signal"""
        signal = np.zeros((self.freq_bins, self.time_bins))
        
        # Continuum emission (broad spectrum)
        for i in range(3):
            freq_center = np.random.uniform(0.2, 0.8) * self.freq_bins
            freq_width = np.random.uniform(20, 40)
            time_center = self.time_bins // 2
            time_width = np.random.uniform(200, 400)
            
            freq_gaussian = np.exp(-0.5 * ((np.arange(self.freq_bins) - freq_center) / freq_width) ** 2)
            time_gaussian = np.exp(-0.5 * ((np.arange(self.time_bins) - time_center) / time_width) ** 2)
            
            component = np.outer(freq_gaussian, time_gaussian)
            signal += component * np.random.uniform(0.3, 0.8)
        
        # HI line emission (21 cm)
        hi_freq = int(self.freq_bins * 0.65)
        hi_profile = np.zeros(self.freq_bins)
        hi_profile[hi_freq-5:hi_freq+5] = np.random.uniform(0.5, 1.0)
        signal += np.outer(hi_profile, np.ones(self.time_bins)) * 0.4
        
        # Add rotation curve effect (frequency shift over time)
        if np.random.rand() > 0.5:
            for t in range(self.time_bins):
                shift = int(3 * np.sin(2 * np.pi * t / self.time_bins))
                signal[:, t] = np.roll(signal[:, t], shift)
        
        return self._apply_quality_effects(signal, quality)
    
    def generate_emission_nebula(self, quality='good'):
        """Generate emission nebula signal (HII region)"""
        signal = np.zeros((self.freq_bins, self.time_bins))
        
        # Multiple emission lines
        emission_lines = [
            (0.3, 0.8),   # H-alpha
            (0.45, 0.6),  # H-beta
            (0.6, 0.5),   # [OIII]
            (0.75, 0.4)   # [NII]
        ]
        
        for rel_freq, intensity in emission_lines:
            freq_center = int(rel_freq * self.freq_bins)
            line_width = np.random.uniform(3, 8)
            
            freq_profile = np.exp(-0.5 * ((np.arange(self.freq_bins) - freq_center) / line_width) ** 2)
            signal += np.outer(freq_profile, np.ones(self.time_bins)) * intensity
        
        # Continuum from ionized gas
        continuum = np.random.uniform(0.1, 0.3, (self.freq_bins, self.time_bins))
        continuum = gaussian_filter(continuum, sigma=5)
        signal += continuum
        
        # Spatial structure (clumpy)
        for _ in range(np.random.randint(3, 8)):
            t_center = np.random.randint(0, self.time_bins)
            t_width = np.random.uniform(50, 150)
            time_profile = np.exp(-0.5 * ((np.arange(self.time_bins) - t_center) / t_width) ** 2)
            signal *= (1 + 0.5 * time_profile)
        
        return self._apply_quality_effects(signal, quality)
    
    def generate_quasar(self, quality='good'):
        """Generate quasar signal (point source, variable)"""
        signal = np.zeros((self.freq_bins, self.time_bins))
        
        # Power-law spectrum (typical for AGN)
        spectrum = np.arange(self.freq_bins, 0, -1) ** (-0.7)
        spectrum = spectrum / spectrum.max()
        
        # Time variability (flickering)
        time_variability = np.ones(self.time_bins)
        
        # Add multiple variability timescales
        for period in [50, 150, 300]:
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.1, 0.3)
            time_variability += amplitude * np.sin(2 * np.pi * np.arange(self.time_bins) / period + phase)
        
        # Normalize
        time_variability = time_variability / time_variability.mean()
        
        signal = np.outer(spectrum, time_variability)
        
        # Add jets (sometimes)
        if np.random.rand() > 0.7:
            jet_freq = np.random.randint(20, 80)
            jet_signal = np.zeros_like(signal)
            jet_signal[jet_freq-5:jet_freq+5, :] = 0.3
            signal += jet_signal
        
        return self._apply_quality_effects(signal, quality)
    
    def generate_pulsar(self, quality='good'):
        """Generate pulsar signal (periodic pulses)"""
        signal = np.zeros((self.freq_bins, self.time_bins))
        
        # Pulsar period (in time bins)
        period = np.random.uniform(20, 100)
        pulse_width = np.random.uniform(0.02, 0.1) * period
        
        # Generate pulses
        num_pulses = int(self.time_bins / period)
        
        for i in range(num_pulses):
            pulse_center = int(i * period + np.random.uniform(-2, 2))  # Timing jitter
            
            if pulse_center < 0 or pulse_center >= self.time_bins:
                continue
            
            # Pulse profile (Gaussian)
            pulse_profile = np.exp(-0.5 * ((np.arange(self.time_bins) - pulse_center) / pulse_width) ** 2)
            
            # Dispersion (frequency-dependent delay)
            DM = np.random.uniform(10, 100)  # Dispersion measure
            
            for freq_idx in range(self.freq_bins):
                delay = int(DM * (1.0 - freq_idx / self.freq_bins) ** 2)
                signal[freq_idx, :] += np.roll(pulse_profile, delay) * (0.7 + 0.3 * freq_idx / self.freq_bins)
        
        # Scintillation (twinkling)
        scintillation = np.random.randn(self.freq_bins, self.time_bins) * 0.2
        scintillation = gaussian_filter(scintillation, sigma=3)
        signal *= (1 + scintillation)
        
        return self._apply_quality_effects(signal, quality)
    
    def _apply_quality_effects(self, signal, quality):
        """Apply realistic quality degradation"""
        
        # Quality-dependent parameters
        quality_params = {
            'excellent': {'noise': 0.05, 'rfi_prob': 0.1, 'corruption': 0.0},
            'good': {'noise': 0.15, 'rfi_prob': 0.4, 'corruption': 0.05},
            'poor': {'noise': 0.35, 'rfi_prob': 0.7, 'corruption': 0.15}
        }
        
        params = quality_params[quality]
        
        # 1. Gaussian noise
        noise = np.random.randn(*signal.shape) * params['noise']
        signal += noise
        
        # 2. RFI (Radio Frequency Interference)
        if np.random.rand() < params['rfi_prob']:
            signal = self._add_rfi(signal)
        
        # 3. Baseline variations
        baseline = self._generate_baseline(signal.shape)
        signal += baseline * 0.1
        
        # 4. Atmospheric effects
        signal = self._add_atmospheric_effects(signal)
        
        # 5. Instrumental artifacts
        if np.random.rand() < 0.3:
            signal = self._add_instrumental_artifacts(signal)
        
        # 6. Data corruption
        if np.random.rand() < params['corruption']:
            signal = self._add_corruption(signal)
        
        # Normalize
        signal = np.clip(signal, 0, None)
        if signal.max() > 0:
            signal = signal / signal.max()
        
        return signal
    
    def _add_rfi(self, signal):
        """Add Radio Frequency Interference"""
        num_rfi = np.random.randint(1, 5)
        
        for _ in range(num_rfi):
            rfi_type = np.random.choice(['vertical', 'horizontal', 'broadband'])
            
            if rfi_type == 'vertical':
                # Vertical stripes (frequency channel interference)
                freq_idx = np.random.randint(0, self.freq_bins)
                width = np.random.randint(1, 5)
                intensity = np.random.uniform(0.3, 1.5)
                signal[freq_idx:freq_idx+width, :] += intensity
                
            elif rfi_type == 'horizontal':
                # Horizontal stripes (time-domain bursts)
                time_idx = np.random.randint(0, self.time_bins)
                duration = np.random.randint(5, 50)
                intensity = np.random.uniform(0.2, 1.0)
                signal[:, time_idx:time_idx+duration] += intensity
                
            else:  # broadband
                # Broadband RFI
                t_start = np.random.randint(0, self.time_bins - 100)
                t_end = t_start + np.random.randint(50, 200)
                intensity = np.random.uniform(0.1, 0.5)
                signal[:, t_start:t_end] += intensity
        
        return signal
    
    def _generate_baseline(self, shape):
        """Generate baseline drift"""
        baseline = np.zeros(shape)
        
        # Frequency-dependent baseline
        for freq_idx in range(shape[0]):
            trend = np.polyval(np.random.randn(3) * 0.01, np.linspace(-1, 1, shape[1]))
            baseline[freq_idx, :] = trend
        
        return baseline
    
    def _add_atmospheric_effects(self, signal):
        """Add atmospheric scintillation and absorption"""
        # Scintillation (random amplitude variations)
        scintillation = np.random.randn(*signal.shape) * 0.05
        scintillation = gaussian_filter(scintillation, sigma=5)
        signal *= (1 + scintillation)
        
        # Atmospheric absorption (frequency dependent)
        absorption = 1 - 0.1 * np.random.rand(self.freq_bins, 1)
        signal *= absorption
        
        return signal
    
    def _add_instrumental_artifacts(self, signal):
        """Add instrumental artifacts"""
        # Dead channels
        if np.random.rand() > 0.5:
            dead_channel = np.random.randint(0, self.freq_bins)
            signal[dead_channel, :] = 0
        
        # Periodic artifacts (60 Hz hum, etc.)
        if np.random.rand() > 0.5:
            period = np.random.choice([50, 60, 100])  # Common interference frequencies
            artifact = 0.05 * np.sin(2 * np.pi * np.arange(self.time_bins) / period)
            signal += artifact
        
        # Spikes (cosmic rays, etc.)
        num_spikes = np.random.randint(0, 5)
        for _ in range(num_spikes):
            spike_freq = np.random.randint(0, self.freq_bins)
            spike_time = np.random.randint(0, self.time_bins)
            spike_intensity = np.random.uniform(1.0, 3.0)
            signal[spike_freq, spike_time] = spike_intensity
        
        return signal
    
    def _add_corruption(self, signal):
        """Add data corruption (missing data, bad regions)"""
        corruption_type = np.random.choice(['missing_block', 'missing_channels', 'missing_times'])
        
        if corruption_type == 'missing_block':
            # Missing rectangular block
            f_start = np.random.randint(0, self.freq_bins - 20)
            f_end = f_start + np.random.randint(10, 30)
            t_start = np.random.randint(0, self.time_bins - 50)
            t_end = t_start + np.random.randint(30, 100)
            signal[f_start:f_end, t_start:t_end] = 0
            
        elif corruption_type == 'missing_channels':
            # Missing frequency channels
            num_missing = np.random.randint(1, 10)
            missing_channels = np.random.choice(self.freq_bins, num_missing, replace=False)
            signal[missing_channels, :] = 0
            
        else:  # missing_times
            # Missing time samples
            num_missing = np.random.randint(5, 20)
            missing_times = np.random.choice(self.time_bins, num_missing, replace=False)
            signal[:, missing_times] = 0
        
        return signal


class RealisticImageGenerator:
    """Generate realistic optical images"""
    
    def __init__(self, image_size=256):
        self.image_size = image_size
    
    def generate_optical_image(self, object_type, quality='good'):
        """Generate optical image for object type"""
        
        if object_type == 'spiral_galaxy':
            return self._generate_galaxy_image(quality)
        elif object_type == 'emission_nebula':
            return self._generate_nebula_image(quality)
        elif object_type == 'quasar':
            return self._generate_quasar_image(quality)
        elif object_type == 'pulsar':
            return self._generate_pulsar_image(quality)
    
    def _generate_galaxy_image(self, quality):
        """Generate spiral galaxy optical image"""
        img = np.zeros((self.image_size, self.image_size, 3))
        
        center = self.image_size // 2
        
        # Bulge
        y, x = np.ogrid[:self.image_size, :self.image_size]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        bulge = np.exp(-r / 20)
        
        # Spiral arms
        theta = np.arctan2(y - center, x - center)
        spiral = np.zeros_like(r)
        
        for arm in range(2):
            arm_angle = arm * np.pi + 0.5 * r / 30
            spiral += np.exp(-((theta - arm_angle) % (2*np.pi) - np.pi)**2 / 0.3) * np.exp(-r / 60)
        
        # Combine
        galaxy = bulge + 0.5 * spiral
        galaxy = galaxy / galaxy.max()
        
        # Color (yellowish for old stars, blue for young)
        img[:, :, 0] = galaxy * 0.8  # R
        img[:, :, 1] = galaxy * 0.7  # G
        img[:, :, 2] = galaxy * 0.5 + spiral * 0.3  # B (blue in arms)
        
        # Add stars
        img = self._add_stars(img, num_stars=np.random.randint(10, 30))
        
        # Add noise and artifacts
        img = self._add_image_artifacts(img, quality)
        
        return np.clip(img, 0, 1)
    
    def _generate_nebula_image(self, quality):
        """Generate emission nebula image"""
        img = np.zeros((self.image_size, self.image_size, 3))
        
        # Create clumpy emission structure
        num_clumps = np.random.randint(5, 15)
        
        for _ in range(num_clumps):
            cx = np.random.randint(50, self.image_size - 50)
            cy = np.random.randint(50, self.image_size - 50)
            size = np.random.uniform(20, 60)
            
            y, x = np.ogrid[:self.image_size, :self.image_size]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            clump = np.exp(-r / size)
            
            # Different emission lines = different colors
            color = np.random.choice(['red', 'green', 'blue'])
            if color == 'red':
                img[:, :, 0] += clump * np.random.uniform(0.5, 1.0)  # H-alpha
            elif color == 'green':
                img[:, :, 1] += clump * np.random.uniform(0.3, 0.8)  # [OIII]
            else:
                img[:, :, 2] += clump * np.random.uniform(0.2, 0.6)
        
        # Add diffuse glow
        diffuse = gaussian_filter(img, sigma=15)
        img = img + 0.3 * diffuse
        
        # Filamentary structure
        for _ in range(3):
            line_img = np.random.randn(self.image_size, self.image_size)
            line_img = gaussian_filter(line_img, sigma=2) > 1.5
            img[line_img] *= 1.5
        
        # Add stars
        img = self._add_stars(img, num_stars=np.random.randint(20, 50))
        
        img = self._add_image_artifacts(img, quality)
        
        return np.clip(img, 0, 1)
    
    def _generate_quasar_image(self, quality):
        """Generate quasar (point source) image"""
        img = np.zeros((self.image_size, self.image_size, 3))
        
        center = self.image_size // 2
        
        # Central point source
        y, x = np.ogrid[:self.image_size, :self.image_size]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # PSF (point spread function)
        psf = np.exp(-r**2 / (2 * 3**2))
        psf += 0.1 * np.exp(-r**2 / (2 * 10**2))  # Diffraction rings
        
        # Blue/white color (hot)
        img[:, :, 0] = psf * 0.7
        img[:, :, 1] = psf * 0.8
        img[:, :, 2] = psf * 1.0
        
        # Host galaxy (faint)
        if np.random.rand() > 0.5:
            host = np.exp(-r / 30) * 0.3
            img[:, :, 0] += host * 0.8
            img[:, :, 1] += host * 0.7
            img[:, :, 2] += host * 0.5
        
        # Add stars
        img = self._add_stars(img, num_stars=np.random.randint(5, 15))
        
        img = self._add_image_artifacts(img, quality)
        
        return np.clip(img, 0, 1)
    
    def _generate_pulsar_image(self, quality):
        """Generate pulsar image (very faint point source)"""
        img = np.zeros((self.image_size, self.image_size, 3))
        
        center = self.image_size // 2
        
        # Very faint point source
        y, x = np.ogrid[:self.image_size, :self.image_size]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        psf = np.exp(-r**2 / (2 * 2**2)) * 0.4  # Faint!
        
        # White/blue (neutron star)
        img[:, :, 0] = psf * 0.8
        img[:, :, 1] = psf * 0.9
        img[:, :, 2] = psf * 1.0
        
        # Possible pulsar wind nebula
        if np.random.rand() > 0.7:
            pwn = np.exp(-r / 15) * 0.2
            img[:, :, 2] += pwn * 0.5  # Blue nebula
        
        # Many background stars (to make it harder)
        img = self._add_stars(img, num_stars=np.random.randint(30, 60))
        
        img = self._add_image_artifacts(img, quality)
        
        return np.clip(img, 0, 1)
    
    def _add_stars(self, img, num_stars=20):
        """Add background stars"""
        for _ in range(num_stars):
            sx = np.random.randint(0, self.image_size)
            sy = np.random.randint(0, self.image_size)
            brightness = np.random.uniform(0.1, 0.8)
            
            # Add small PSF for star
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if 0 <= sy+dy < self.image_size and 0 <= sx+dx < self.image_size:
                        r = np.sqrt(dx**2 + dy**2)
                        if r < 2.5:
                            img[sy+dy, sx+dx, :] += brightness * np.exp(-r**2 / 2)
        
        return img
    
    def _add_image_artifacts(self, img, quality):
        """Add realistic image artifacts"""
        
        quality_params = {
            'excellent': {'noise': 0.01, 'blur': 0, 'vignetting': 0},
            'good': {'noise': 0.03, 'blur': 0.5, 'vignetting': 0.1},
            'poor': {'noise': 0.08, 'blur': 1.5, 'vignetting': 0.25}
        }
        
        params = quality_params[quality]
        
        # Gaussian noise
        noise = np.random.randn(*img.shape) * params['noise']
        img += noise
        
        # Blur (atmospheric seeing, tracking errors)
        if params['blur'] > 0:
            for c in range(3):
                img[:, :, c] = gaussian_filter(img[:, :, c], sigma=params['blur'])
        
        # Vignetting (darkening at edges)
        if params['vignetting'] > 0:
            y, x = np.ogrid[:self.image_size, :self.image_size]
            center = self.image_size // 2
            r = np.sqrt((x - center)**2 + (y - center)**2)
            vignette = 1 - params['vignetting'] * (r / (self.image_size / 2))**2
            img *= vignette[:, :, np.newaxis]
        
        # Hot pixels
        if quality == 'poor':
            num_hot_pixels = np.random.randint(5, 20)
            for _ in range(num_hot_pixels):
                hx = np.random.randint(0, self.image_size)
                hy = np.random.randint(0, self.image_size)
                img[hy, hx, :] = 1.0
        
        # Cosmic rays
        if np.random.rand() > 0.5:
            num_cosmic_rays = np.random.randint(1, 5)
            for _ in range(num_cosmic_rays):
                cx = np.random.randint(5, self.image_size - 5)
                cy = np.random.randint(5, self.image_size - 5)
                length = np.random.randint(3, 10)
                angle = np.random.uniform(0, 2*np.pi)
                
                for i in range(length):
                    x = int(cx + i * np.cos(angle))
                    y = int(cy + i * np.sin(angle))
                    if 0 <= x < self.image_size and 0 <= y < self.image_size:
                        img[y, x, :] = 1.0
        
        return img
    
    def generate_radio_visualization(self, signal):
        """Generate radio image from signal (for visualization)"""
        # Simple visualization: show signal as grayscale image
        radio_img = signal.copy()
        
        # Enhance contrast
        radio_img = np.clip(radio_img, 0, None)
        if radio_img.max() > 0:
            radio_img = radio_img / radio_img.max()
        
        # Resize to image_size x image_size
        radio_img_resized = zoom(radio_img, 
                                  (self.image_size / signal.shape[0], 
                                   self.image_size / signal.shape[1]))
        
        return radio_img_resized


def generate_dataset():
    """Main dataset generation function"""
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    signal_gen = RealisticSignalGenerator(
        signal_shape=CONFIG['signal_shape'],
        noise_level=CONFIG['noise_level']
    )
    
    image_gen = RealisticImageGenerator(
        image_size=CONFIG['image_size']
    )
    
    all_metadata = []
    all_signals = []
    
    sample_id = 0
    
    # Generate each object type
    object_types = [
        ('spiral_galaxy', CONFIG['num_spiral_galaxies']),
        ('emission_nebula', CONFIG['num_emission_nebulae']),
        ('quasar', CONFIG['num_quasars']),
        ('pulsar', CONFIG['num_pulsars'])
    ]
    
    for object_type, num_samples in object_types:
        print(f"\n{'='*60}")
        print(f"GENERATING {object_type.upper().replace('_', ' ')}S ({num_samples} samples)")
        print(f"{'='*60}")
        
        # Create directories
        optical_dir = output_dir / object_type / 'optical'
        radio_dir = output_dir / object_type / 'radio'
        optical_dir.mkdir(parents=True, exist_ok=True)
        radio_dir.mkdir(parents=True, exist_ok=True)
        
        for i in tqdm(range(num_samples), desc=f"Generating {object_type}"):
            # Determine quality
            quality_roll = np.random.rand()
            if quality_roll < CONFIG['quality_distribution']['excellent']:
                quality = 'excellent'
            elif quality_roll < CONFIG['quality_distribution']['excellent'] + CONFIG['quality_distribution']['good']:
                quality = 'good'
            else:
                quality = 'poor'
            
            # Generate signal
            if object_type == 'spiral_galaxy':
                signal = signal_gen.generate_spiral_galaxy(quality)
            elif object_type == 'emission_nebula':
                signal = signal_gen.generate_emission_nebula(quality)
            elif object_type == 'quasar':
                signal = signal_gen.generate_quasar(quality)
            elif object_type == 'pulsar':
                signal = signal_gen.generate_pulsar(quality)
            
            # Generate optical image
            optical_img = image_gen.generate_optical_image(object_type, quality)
            
            # Generate radio image (visualization)
            radio_img = image_gen.generate_radio_visualization(signal)
            
            # Save images
            optical_path = optical_dir / f'{object_type}_{i:05d}.jpg'
            radio_path = radio_dir / f'{object_type}_{i:05d}.png'
            
            # Save optical (RGB)
            optical_pil = Image.fromarray((optical_img * 255).astype(np.uint8))
            optical_pil.save(optical_path, quality=95)
            
            # Save radio (grayscale)
            radio_pil = Image.fromarray((radio_img * 255).astype(np.uint8))
            radio_pil.save(radio_path)
            
            # Store signal and metadata
            all_signals.append(signal)
            
            # Create metadata entry
            metadata_entry = {
                'sample_id': sample_id,
                'object_type': object_type,
                'optical_image_path': str(optical_path.relative_to(output_dir)),
                'radio_image_path': str(radio_path.relative_to(output_dir)),
                'quality': quality,
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'redshift': np.random.uniform(0.01, 2.0) if object_type != 'pulsar' else 0.0,
                'source': 'synthetic_realistic'
            }
            
            all_metadata.append(metadata_entry)
            sample_id += 1
    
    # Save signals to HDF5
    print(f"\n{'='*60}")
    print("SAVING SIGNALS TO HDF5")
    print(f"{'='*60}")
    
    signals_array = np.array(all_signals)
    h5_path = output_dir / 'signals.h5'
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('signals', data=signals_array, compression='gzip', compression_opts=9)
    
    print(f"✅ Saved {len(signals_array)} signals")
    print(f"   Shape: {signals_array.shape}")
    print(f"   Size: {h5_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"✅ Saved metadata: {metadata_path}")
    
    # Save summary
    from collections import Counter
    quality_counts = Counter([m['quality'] for m in all_metadata])
    type_counts = Counter([m['object_type'] for m in all_metadata])
    
    summary = {
        'total_samples': len(all_metadata),
        'object_types': dict(type_counts),
        'quality_distribution': dict(quality_counts),
        'signal_shape': list(CONFIG['signal_shape']),
        'image_size': CONFIG['image_size'],
        'noise_level': CONFIG['noise_level'],
        'rfi_probability': CONFIG['rfi_probability'],
        'corruption_probability': CONFIG['corruption_probability']
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("🎉 DATASET GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nTotal samples: {len(all_metadata)}")
    print(f"\nObject types:")
    for obj_type, count in type_counts.items():
        print(f"   {obj_type.replace('_', ' ').title():20s}: {count:4d}")
    print(f"\nQuality distribution:")
    for quality, count in quality_counts.items():
        print(f"   {quality.title():10s}: {count:4d} ({count/len(all_metadata)*100:.1f}%)")
    print(f"\n📁 Output directory: {output_dir}/")
    print(f"📄 Metadata: {metadata_path}")
    print(f"📊 Signals: {h5_path}")
    print(f"📋 Summary: {summary_path}")
    
    # Generate sample visualization
    print(f"\n📸 Generating sample visualization...")
    visualize_samples(output_dir, all_metadata, all_signals)
    
    print(f"\n✅ Realistic synthetic dataset ready!")
    print(f"\n🎯 Expected performance:")
    print(f"   - NOT 100% accuracy (more realistic)")
    print(f"   - Better transfer to real data")
    print(f"   - Handles noise, RFI, artifacts")


def visualize_samples(output_dir, metadata, signals):
    """Create visualization of sample data"""
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for obj_idx, object_type in enumerate(['spiral_galaxy', 'emission_nebula', 'quasar', 'pulsar']):
        # Find a sample of this type
        sample = [m for m in metadata if m['object_type'] == object_type][0]
        signal = signals[sample['sample_id']]
        
        # Load images
        optical_img = Image.open(output_dir / sample['optical_image_path'])
        radio_img = Image.open(output_dir / sample['radio_image_path'])
        
        # Plot signal
        axes[obj_idx, 0].imshow(signal, aspect='auto', cmap='viridis')
        axes[obj_idx, 0].set_title(f"{object_type.replace('_', ' ').title()} - Signal")
        axes[obj_idx, 0].set_ylabel('Frequency')
        axes[obj_idx, 0].set_xlabel('Time')
        
        # Plot radio image
        axes[obj_idx, 1].imshow(radio_img, cmap='gray')
        axes[obj_idx, 1].set_title('Radio Image')
        axes[obj_idx, 1].axis('off')
        
        # Plot optical image
        axes[obj_idx, 2].imshow(optical_img)
        axes[obj_idx, 2].set_title('Optical Image')
        axes[obj_idx, 2].axis('off')
        
        # Plot quality info
        axes[obj_idx, 3].text(0.5, 0.5, 
                             f"Quality: {sample['quality']}\n"
                             f"RA: {sample['ra']:.2f}\n"
                             f"Dec: {sample['dec']:.2f}\n"
                             f"z: {sample['redshift']:.3f}",
                             ha='center', va='center', fontsize=10)
        axes[obj_idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'sample_visualization.png'}")
    plt.close()


if __name__ == "__main__":
    print("\n📋 This will generate:")
    print(f"   - {sum([CONFIG[k] for k in CONFIG if k.startswith('num_')])} total samples")
    print(f"   - Signal shape: {CONFIG['signal_shape']}")
    print(f"   - Image size: {CONFIG['image_size']}×{CONFIG['image_size']}")
    print(f"   - Quality: {CONFIG['quality_distribution']['excellent']*100:.0f}% excellent, "
          f"{CONFIG['quality_distribution']['good']*100:.0f}% good, "
          f"{CONFIG['quality_distribution']['poor']*100:.0f}% poor")
    print(f"\n⏱️  Estimated time: ~15-30 minutes")
    print()
    
    response = input("Press Enter to start generation (or Ctrl+C to cancel)...")
    
    try:
        generate_dataset()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()