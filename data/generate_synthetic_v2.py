"""
SYNTHETIC DATASET GENERATOR V2
Updated for new classes: galaxy, quasar, radio_galaxy, agn
With truly diverse morphologies, orientations, and structures

Run: python generate_synthetic_v2.py
"""

import numpy as np
from PIL import Image
import h5py
import json
import os
from tqdm import tqdm

print("""
=====================================================
   SYNTHETIC DATASET GENERATOR V2
   Classes: galaxy, quasar, radio_galaxy, agn
   ~5200 samples with diverse morphologies
=====================================================
""")

CONFIG = {
    'num_samples': 5200,  # 1300 per class to match real data
    'output_dir': 'synthetic_dataset_v2',
    'signal_shape': (128, 1024),
    'image_size': 256
}


class AstronomicalObjectGenerator:
    """Generate diverse astronomical objects for 4 classes"""

    def __init__(self):
        self.signal_shape = CONFIG['signal_shape']
        self.img_size = CONFIG['image_size']

    # ================================================================
    # GALAXY (includes spiral, elliptical, irregular morphologies)
    # ================================================================

    def generate_galaxy_signal(self, seed):
        """Generate galaxy radio signal - multiple morphologies"""
        np.random.seed(seed)
        signal = np.zeros(self.signal_shape)
        h, w = self.signal_shape

        center_h = h // 2 + np.random.randint(-10, 10)
        center_w = w // 2 + np.random.randint(-200, 200)

        # Random morphology type
        morph_type = np.random.choice(['spiral', 'elliptical', 'irregular', 'ring', 'barred'])

        brightness = np.random.uniform(0.3, 0.9)
        size = np.random.uniform(25, 80)
        inclination = np.random.uniform(0, np.pi/2.5)
        rotation = np.random.uniform(0, 2*np.pi)

        if morph_type == 'spiral':
            num_arms = np.random.choice([2, 3, 4, 5])
            arm_tightness = np.random.uniform(15, 35)

            for i in range(h):
                for j in range(w):
                    y = (i - center_h) * np.cos(inclination)
                    x = j - center_w
                    dist = np.sqrt(x**2 + y**2)
                    angle = np.arctan2(y, x) + rotation

                    spiral = np.sin(num_arms * angle - dist / arm_tightness)**2
                    profile = brightness * np.exp(-dist / size)
                    signal[i, j] = profile * (0.25 + 0.75 * spiral)

        elif morph_type == 'elliptical':
            ellipticity = np.random.uniform(0.3, 1.0)
            sersic_n = np.random.uniform(2, 6)

            for i in range(h):
                for j in range(w):
                    y = (i - center_h) / ellipticity
                    x = j - center_w
                    r = np.sqrt(x**2 + y**2)
                    signal[i, j] = brightness * np.exp(-(r / size)**(1/sersic_n))

        elif morph_type == 'irregular':
            num_blobs = np.random.randint(4, 10)
            for _ in range(num_blobs):
                blob_h = center_h + np.random.randint(-30, 30)
                blob_w = center_w + np.random.randint(-150, 150)
                blob_size = np.random.uniform(10, 30)
                blob_bright = brightness * np.random.uniform(0.5, 1.0)

                for i in range(h):
                    for j in range(w):
                        dist = np.sqrt((i-blob_h)**2 + (j-blob_w)**2)
                        signal[i, j] += blob_bright * np.exp(-dist / blob_size)

        elif morph_type == 'ring':
            ring_radius = np.random.uniform(30, 60)
            ring_width = np.random.uniform(5, 15)

            for i in range(h):
                for j in range(w):
                    y = (i - center_h) * np.cos(inclination)
                    x = j - center_w
                    dist = np.sqrt(x**2 + y**2)
                    ring = np.exp(-((dist - ring_radius)**2) / (2 * ring_width**2))
                    signal[i, j] = brightness * ring

        else:  # barred spiral
            bar_length = np.random.uniform(20, 50)
            bar_angle = rotation
            num_arms = 2

            for i in range(h):
                for j in range(w):
                    y = (i - center_h) * np.cos(inclination)
                    x = j - center_w

                    # Rotate to bar frame
                    x_rot = x * np.cos(bar_angle) + y * np.sin(bar_angle)
                    y_rot = -x * np.sin(bar_angle) + y * np.cos(bar_angle)

                    dist = np.sqrt(x**2 + y**2)
                    angle = np.arctan2(y, x) + rotation

                    # Bar component
                    bar = brightness * 0.8 * np.exp(-abs(y_rot)/10) * np.exp(-abs(x_rot)/bar_length)

                    # Spiral arms from bar ends
                    spiral = np.sin(num_arms * angle - dist / 25)**2
                    arm = brightness * np.exp(-dist / size) * spiral * (1 if dist > bar_length/2 else 0.3)

                    signal[i, j] = bar + arm

        # Add core
        core_size = np.random.uniform(3, 10)
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i-center_h)**2 + (j-center_w)**2)
                signal[i, j] += np.random.uniform(0.15, 0.35) * np.exp(-dist / core_size)

        # Star formation hotspots
        if morph_type in ['spiral', 'irregular', 'barred']:
            for _ in range(np.random.randint(3, 10)):
                hs_h, hs_w = np.random.randint(0, h), np.random.randint(0, w)
                for i in range(max(0, hs_h-4), min(h, hs_h+4)):
                    for j in range(max(0, hs_w-15), min(w, hs_w+15)):
                        dist = np.sqrt((i-hs_h)**2 + (j-hs_w)**2)
                        signal[i, j] += 0.12 * np.exp(-dist / 3)

        signal += np.random.randn(h, w) * np.random.uniform(0.01, 0.04)
        return np.clip(signal, 0, 1)

    def generate_galaxy_image(self, seed):
        """Generate galaxy optical image - multiple morphologies"""
        np.random.seed(seed + 10000)
        img = np.zeros((self.img_size, self.img_size, 3))
        center = self.img_size // 2 + np.random.randint(-20, 20)

        morph_type = np.random.choice(['spiral', 'elliptical', 'irregular', 'ring', 'barred'])
        brightness = np.random.uniform(120, 220)
        size = np.random.uniform(35, 85)
        inclination = np.random.uniform(0, 70)
        rotation = np.random.uniform(0, 360)

        # Color palette based on galaxy type
        if morph_type == 'elliptical':
            base_color = np.array([1.0, 0.85, 0.65])  # Yellow-red old stars
        elif morph_type in ['spiral', 'barred']:
            base_color = np.array([0.7, 0.85, 1.0])  # Blue-ish young stars
            bulge_color = np.array([1.0, 0.9, 0.7])
        else:
            base_color = np.array([0.8, 0.9, 1.0])

        cos_inc = np.cos(np.radians(inclination))
        rot_rad = np.radians(rotation)

        if morph_type == 'spiral':
            num_arms = np.random.choice([2, 3, 4])
            for i in range(self.img_size):
                for j in range(self.img_size):
                    y = (i - center) * cos_inc
                    x = j - center
                    dist = np.sqrt(x**2 + y**2)
                    angle = np.arctan2(y, x) + rot_rad

                    spiral = np.sin(num_arms * angle - dist / 18)**2
                    radial = brightness * np.exp(-dist / size)
                    val = radial * (0.2 + 0.8 * spiral)

                    if dist < 12:
                        img[i, j] = val * bulge_color
                    else:
                        img[i, j] = val * base_color

        elif morph_type == 'elliptical':
            ellip = np.random.uniform(0.4, 1.0)
            for i in range(self.img_size):
                for j in range(self.img_size):
                    y = (i - center) / ellip
                    x = j - center
                    dist = np.sqrt(x**2 + y**2)
                    val = brightness * np.exp(-dist / size)
                    img[i, j] = val * base_color

        elif morph_type == 'irregular':
            for _ in range(np.random.randint(5, 12)):
                blob_i = center + np.random.randint(-50, 50)
                blob_j = center + np.random.randint(-50, 50)
                blob_size = np.random.uniform(15, 40)
                blob_bright = brightness * np.random.uniform(0.5, 1.0)
                blob_color = base_color * np.random.uniform(0.8, 1.2, 3)

                for i in range(self.img_size):
                    for j in range(self.img_size):
                        dist = np.sqrt((i-blob_i)**2 + (j-blob_j)**2)
                        img[i, j] += blob_bright * np.exp(-dist / blob_size) * blob_color

        elif morph_type == 'ring':
            ring_r = np.random.uniform(40, 70)
            ring_w = np.random.uniform(8, 18)
            for i in range(self.img_size):
                for j in range(self.img_size):
                    y = (i - center) * cos_inc
                    x = j - center
                    dist = np.sqrt(x**2 + y**2)
                    ring = np.exp(-((dist - ring_r)**2) / (2 * ring_w**2))
                    img[i, j] = brightness * ring * base_color

        else:  # barred
            bar_len = np.random.uniform(25, 50)
            for i in range(self.img_size):
                for j in range(self.img_size):
                    y = (i - center) * cos_inc
                    x = j - center
                    x_rot = x * np.cos(rot_rad) + y * np.sin(rot_rad)
                    y_rot = -x * np.sin(rot_rad) + y * np.cos(rot_rad)

                    dist = np.sqrt(x**2 + y**2)
                    angle = np.arctan2(y, x) + rot_rad

                    bar = brightness * 0.7 * np.exp(-abs(y_rot)/12) * np.exp(-abs(x_rot)/bar_len)
                    spiral = np.sin(2 * angle - dist / 20)**2
                    arm = brightness * np.exp(-dist / size) * spiral * (1 if dist > bar_len/2 else 0.2)

                    if dist < 10:
                        img[i, j] = (bar + arm) * np.array([1.0, 0.9, 0.7])
                    else:
                        img[i, j] = (bar + arm) * base_color

        # Add star-forming regions
        if morph_type in ['spiral', 'irregular', 'barred']:
            for _ in range(np.random.randint(5, 18)):
                sf_i, sf_j = np.random.randint(30, self.img_size-30), np.random.randint(30, self.img_size-30)
                sf_bright = np.random.uniform(180, 255)
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        if 0 <= sf_i+di < self.img_size and 0 <= sf_j+dj < self.img_size:
                            d = np.sqrt(di**2 + dj**2)
                            if d < 3:
                                img[sf_i+di, sf_j+dj] += sf_bright * np.exp(-d) * np.array([0.7, 0.9, 1.0])

        # Background stars
        for _ in range(np.random.randint(25, 55)):
            si, sj = np.random.randint(0, self.img_size), np.random.randint(0, self.img_size)
            sb = np.random.uniform(120, 240)
            sc = np.random.choice([[1,1,1], [0.8,0.9,1], [1,0.9,0.7], [1,0.6,0.4]])
            img[si, sj] = sb * np.array(sc)

        return np.clip(img, 0, 255).astype(np.uint8)

    # ================================================================
    # QUASAR - bright AGN with point-like appearance
    # ================================================================

    def generate_quasar_signal(self, seed):
        """Generate quasar radio signal - bright compact source"""
        np.random.seed(seed)
        signal = np.zeros(self.signal_shape)
        h, w = self.signal_shape

        center_h = h // 2 + np.random.randint(-5, 5)
        center_w = w // 2 + np.random.randint(-100, 100)

        core_brightness = np.random.uniform(0.6, 1.0)
        core_size = np.random.uniform(3, 12)

        # Variability pattern (quasars are variable)
        variability = np.random.choice(['steady', 'flaring', 'fading'])

        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i-center_h)**2 + (j-center_w)**2)
                base = core_brightness * np.exp(-dist / core_size)

                if variability == 'flaring':
                    base *= (1 + 0.3 * np.sin(j / 50))
                elif variability == 'fading':
                    base *= (1 - 0.2 * j / w)

                signal[i, j] = base

        # Possible radio jets (30% of quasars)
        if np.random.rand() > 0.7:
            jet_angle = np.random.uniform(0, 2*np.pi)
            jet_length = np.random.uniform(25, 70)
            jet_bright = np.random.uniform(0.2, 0.5)
            jet_width = np.random.uniform(2, 5)

            for direction in [1, -1]:
                for l in range(int(jet_length)):
                    jh = int(center_h + direction * l * np.sin(jet_angle))
                    jw = int(center_w + direction * l * np.cos(jet_angle))
                    if 0 <= jh < h and 0 <= jw < w:
                        brightness = jet_bright * np.exp(-l / jet_length * 2)
                        for dh in range(-int(jet_width), int(jet_width)+1):
                            for dw in range(-int(jet_width*4), int(jet_width*4)+1):
                                nh, nw = jh + dh, jw + dw
                                if 0 <= nh < h and 0 <= nw < w:
                                    d = np.sqrt(dh**2 + dw**2)
                                    if d < jet_width:
                                        signal[nh, nw] += brightness * np.exp(-d/jet_width)

        # Faint host galaxy (sometimes visible)
        if np.random.rand() > 0.5:
            host_size = np.random.uniform(18, 35)
            host_bright = np.random.uniform(0.08, 0.2)
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i-center_h)**2 + (j-center_w)**2)
                    signal[i, j] += host_bright * np.exp(-dist / host_size)

        signal += np.random.randn(h, w) * np.random.uniform(0.02, 0.04)
        return np.clip(signal, 0, 1)

    def generate_quasar_image(self, seed):
        """Generate quasar optical image - very bright point source"""
        np.random.seed(seed + 10000)
        img = np.zeros((self.img_size, self.img_size, 3))
        center = self.img_size // 2 + np.random.randint(-15, 15)

        core_brightness = np.random.uniform(220, 255)
        core_size = np.random.uniform(2, 6)
        color_temp = np.random.choice(['blue', 'white', 'uv'])

        # Host galaxy (if visible)
        if np.random.rand() > 0.4:
            host_bright = np.random.uniform(40, 100)
            host_size = np.random.uniform(25, 50)
            for i in range(self.img_size):
                for j in range(self.img_size):
                    dist = np.sqrt((i-center)**2 + (j-center)**2)
                    val = host_bright * np.exp(-dist / host_size)
                    img[i, j] = [val, val * 0.85, val * 0.65]

        # Bright quasar core
        for i in range(self.img_size):
            for j in range(self.img_size):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                if dist < core_size * 3:
                    val = core_brightness * np.exp(-dist / core_size)
                    if color_temp == 'blue':
                        img[i, j] += [val*0.85, val*0.92, val]
                    elif color_temp == 'uv':
                        img[i, j] += [val*0.75, val*0.85, val]
                    else:
                        img[i, j] += [val, val, val*0.95]

        # Diffraction spikes (bright sources)
        if np.random.rand() > 0.4:
            spike_len = np.random.randint(25, 50)
            spike_bright = np.random.uniform(120, 200)
            for offset in range(-spike_len, spike_len):
                decay = 1 - abs(offset)/spike_len
                if 0 <= center + offset < self.img_size:
                    img[center, center + offset] += spike_bright * decay
                    img[center + offset, center] += spike_bright * decay

        # Background stars
        for _ in range(np.random.randint(35, 65)):
            si, sj = np.random.randint(0, self.img_size), np.random.randint(0, self.img_size)
            if np.sqrt((si-center)**2 + (sj-center)**2) > 25:
                sb = np.random.uniform(100, 200)
                img[si, sj] = [sb, sb, sb*0.95]

        return np.clip(img, 0, 255).astype(np.uint8)

    # ================================================================
    # RADIO GALAXY - Extended radio lobes, jets
    # ================================================================

    def generate_radio_galaxy_signal(self, seed):
        """Generate radio galaxy signal - lobes and jets"""
        np.random.seed(seed)
        signal = np.zeros(self.signal_shape)
        h, w = self.signal_shape

        center_h = h // 2 + np.random.randint(-8, 8)
        center_w = w // 2 + np.random.randint(-150, 150)

        # Radio galaxy type
        rg_type = np.random.choice(['FR1', 'FR2', 'WAT', 'NAT', 'compact'])

        core_bright = np.random.uniform(0.3, 0.7)
        core_size = np.random.uniform(3, 8)
        jet_angle = np.random.uniform(0, 2*np.pi)

        # Core
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i-center_h)**2 + (j-center_w)**2)
                signal[i, j] = core_bright * np.exp(-dist / core_size)

        if rg_type == 'FR1':
            # Edge-darkened lobes
            lobe_len = np.random.uniform(40, 80)
            lobe_bright = np.random.uniform(0.3, 0.6)
            lobe_width = np.random.uniform(8, 18)

            for direction in [1, -1]:
                for l in range(int(lobe_len)):
                    lh = int(center_h + direction * l * np.sin(jet_angle))
                    lw = int(center_w + direction * l * np.cos(jet_angle))
                    # Brightness decreases with distance (FR1)
                    bright = lobe_bright * (1 - l/lobe_len * 0.7)
                    width = lobe_width * (1 + l/lobe_len * 0.5)

                    for dh in range(-int(width), int(width)+1):
                        for dw in range(-int(width*4), int(width*4)+1):
                            nh, nw = lh + dh, lw + dw
                            if 0 <= nh < h and 0 <= nw < w:
                                d = np.sqrt(dh**2 + (dw/4)**2)
                                if d < width:
                                    signal[nh, nw] += bright * np.exp(-d/width)

        elif rg_type == 'FR2':
            # Edge-brightened lobes with hotspots
            lobe_len = np.random.uniform(50, 100)
            lobe_bright = np.random.uniform(0.2, 0.4)
            hotspot_bright = np.random.uniform(0.5, 0.8)

            for direction in [1, -1]:
                # Jet
                for l in range(int(lobe_len)):
                    jh = int(center_h + direction * l * np.sin(jet_angle))
                    jw = int(center_w + direction * l * np.cos(jet_angle))
                    if 0 <= jh < h and 0 <= jw < w:
                        signal[jh, jw] += lobe_bright * 0.5

                # Hotspot at end
                hs_h = int(center_h + direction * lobe_len * np.sin(jet_angle))
                hs_w = int(center_w + direction * lobe_len * np.cos(jet_angle))
                for dh in range(-10, 11):
                    for dw in range(-40, 41):
                        nh, nw = hs_h + dh, hs_w + dw
                        if 0 <= nh < h and 0 <= nw < w:
                            d = np.sqrt(dh**2 + (dw/4)**2)
                            signal[nh, nw] += hotspot_bright * np.exp(-d / 8)

        elif rg_type == 'WAT':
            # Wide-angle tail - bent jets
            lobe_len = np.random.uniform(40, 70)
            bend_angle = np.random.uniform(0.3, 0.8)

            for direction in [1, -1]:
                for l in range(int(lobe_len)):
                    # Increasing bend with distance
                    angle_offset = bend_angle * (l / lobe_len)**2 * direction
                    jh = int(center_h + l * np.sin(jet_angle + angle_offset))
                    jw = int(center_w + l * np.cos(jet_angle + angle_offset))
                    if 0 <= jh < h and 0 <= jw < w:
                        width = 3 + l / 10
                        for dh in range(-5, 6):
                            for dw in range(-20, 21):
                                nh, nw = jh + dh, jw + dw
                                if 0 <= nh < h and 0 <= nw < w:
                                    d = np.sqrt(dh**2 + (dw/4)**2)
                                    signal[nh, nw] += 0.4 * np.exp(-d / width)

        elif rg_type == 'NAT':
            # Narrow-angle tail - sharply bent
            lobe_len = np.random.uniform(35, 60)

            for l in range(int(lobe_len)):
                # Both jets bent in same direction
                jh = int(center_h + l * 0.3)
                jw = int(center_w + l)
                if 0 <= jh < h and 0 <= jw < w:
                    for dh in range(-4, 5):
                        for dw in range(-15, 16):
                            nh, nw = jh + dh, jw + dw
                            if 0 <= nh < h and 0 <= nw < w:
                                d = np.sqrt(dh**2 + (dw/4)**2)
                                signal[nh, nw] += 0.35 * np.exp(-d / 4)
        else:
            # Compact radio source
            ext_size = np.random.uniform(8, 20)
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i-center_h)**2 + (j-center_w)**2)
                    signal[i, j] += 0.5 * np.exp(-dist / ext_size)

        signal += np.random.randn(h, w) * np.random.uniform(0.02, 0.05)
        return np.clip(signal, 0, 1)

    def generate_radio_galaxy_image(self, seed):
        """Generate radio galaxy optical image - host galaxy"""
        np.random.seed(seed + 10000)
        img = np.zeros((self.img_size, self.img_size, 3))
        center = self.img_size // 2 + np.random.randint(-20, 20)

        # Radio galaxies are usually massive ellipticals
        brightness = np.random.uniform(140, 220)
        size = np.random.uniform(40, 80)
        ellipticity = np.random.uniform(0.5, 1.0)

        # Core AGN contribution
        agn_bright = np.random.uniform(0, 100)  # Variable AGN brightness

        for i in range(self.img_size):
            for j in range(self.img_size):
                y = (i - center) / ellipticity
                x = j - center
                dist = np.sqrt(x**2 + y**2)

                # Elliptical galaxy profile (de Vaucouleurs)
                val = brightness * np.exp(-(dist / size)**0.25 * 7.67)

                # Yellow-red color (old stellar population)
                img[i, j] = [val, val * 0.85, val * 0.6]

                # AGN point source
                if dist < 4:
                    agn_val = agn_bright * np.exp(-dist)
                    img[i, j] += [agn_val, agn_val, agn_val * 0.9]

        # Possible dust lane
        if np.random.rand() > 0.7:
            dust_angle = np.random.uniform(0, np.pi)
            for i in range(self.img_size):
                for j in range(self.img_size):
                    y, x = i - center, j - center
                    dist_to_line = abs(y * np.cos(dust_angle) - x * np.sin(dust_angle))
                    if dist_to_line < 3 and np.sqrt(x**2 + y**2) < 40:
                        img[i, j] *= 0.4

        # Background stars
        for _ in range(np.random.randint(30, 60)):
            si, sj = np.random.randint(0, self.img_size), np.random.randint(0, self.img_size)
            sb = np.random.uniform(100, 200)
            img[si, sj] = [sb, sb, sb * 0.9]

        return np.clip(img, 0, 255).astype(np.uint8)

    # ================================================================
    # AGN (Active Galactic Nucleus) - various types
    # ================================================================

    def generate_agn_signal(self, seed):
        """Generate AGN radio signal - Seyfert, LINER, etc."""
        np.random.seed(seed)
        signal = np.zeros(self.signal_shape)
        h, w = self.signal_shape

        center_h = h // 2 + np.random.randint(-8, 8)
        center_w = w // 2 + np.random.randint(-120, 120)

        agn_type = np.random.choice(['seyfert1', 'seyfert2', 'liner', 'blazar'])

        core_bright = np.random.uniform(0.5, 0.9)
        core_size = np.random.uniform(4, 12)

        # Core emission
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i-center_h)**2 + (j-center_w)**2)
                signal[i, j] = core_bright * np.exp(-dist / core_size)

        if agn_type == 'seyfert1':
            # Bright unobscured nucleus
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i-center_h)**2 + (j-center_w)**2)
                    if dist < core_size:
                        signal[i, j] += 0.3 * np.exp(-dist / (core_size * 0.3))

        elif agn_type == 'seyfert2':
            # Obscured nucleus - narrow-line region visible
            nlr_size = np.random.uniform(15, 30)
            nlr_angle = np.random.uniform(0, 2*np.pi)

            for i in range(h):
                for j in range(w):
                    y, x = i - center_h, j - center_w
                    angle = np.arctan2(y, x)
                    dist = np.sqrt(x**2 + y**2)

                    # Bi-conical NLR
                    angle_diff = abs(np.sin(angle - nlr_angle))
                    if angle_diff < 0.4 and dist < nlr_size:
                        signal[i, j] += 0.25 * np.exp(-dist / nlr_size)

        elif agn_type == 'liner':
            # Low-ionization nuclear emission region
            liner_size = np.random.uniform(8, 20)
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i-center_h)**2 + (j-center_w)**2)
                    signal[i, j] += 0.2 * np.exp(-dist / liner_size)

        else:  # blazar
            # Highly variable, beamed emission
            variability_amp = np.random.uniform(0.2, 0.5)
            variability_freq = np.random.uniform(20, 60)

            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i-center_h)**2 + (j-center_w)**2)
                    var = 1 + variability_amp * np.sin(j / variability_freq)
                    signal[i, j] *= var

        # Host galaxy emission
        host_size = np.random.uniform(20, 45)
        host_bright = np.random.uniform(0.1, 0.3)
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i-center_h)**2 + (j-center_w)**2)
                signal[i, j] += host_bright * np.exp(-dist / host_size)

        signal += np.random.randn(h, w) * np.random.uniform(0.02, 0.04)
        return np.clip(signal, 0, 1)

    def generate_agn_image(self, seed):
        """Generate AGN optical image"""
        np.random.seed(seed + 10000)
        img = np.zeros((self.img_size, self.img_size, 3))
        center = self.img_size // 2 + np.random.randint(-15, 15)

        agn_type = np.random.choice(['seyfert1', 'seyfert2', 'liner', 'blazar'])

        # Host galaxy (spiral or elliptical)
        host_type = np.random.choice(['spiral', 'elliptical'])
        host_bright = np.random.uniform(100, 180)
        host_size = np.random.uniform(40, 75)

        if host_type == 'spiral':
            num_arms = np.random.choice([2, 3])
            inclination = np.random.uniform(0, 60)
            rotation = np.random.uniform(0, 360)

            for i in range(self.img_size):
                for j in range(self.img_size):
                    y = (i - center) * np.cos(np.radians(inclination))
                    x = j - center
                    dist = np.sqrt(x**2 + y**2)
                    angle = np.arctan2(y, x) + np.radians(rotation)

                    spiral = np.sin(num_arms * angle - dist / 20)**2
                    val = host_bright * np.exp(-dist / host_size) * (0.3 + 0.7 * spiral)

                    if dist < 15:
                        img[i, j] = [val, val * 0.85, val * 0.65]
                    else:
                        img[i, j] = [val * 0.75, val * 0.85, val]
        else:
            ellip = np.random.uniform(0.5, 1.0)
            for i in range(self.img_size):
                for j in range(self.img_size):
                    y = (i - center) / ellip
                    x = j - center
                    dist = np.sqrt(x**2 + y**2)
                    val = host_bright * np.exp(-dist / host_size)
                    img[i, j] = [val, val * 0.85, val * 0.65]

        # AGN core
        if agn_type in ['seyfert1', 'blazar']:
            core_bright = np.random.uniform(200, 255)
            core_size = np.random.uniform(2, 5)
        else:
            core_bright = np.random.uniform(120, 180)
            core_size = np.random.uniform(3, 6)

        for i in range(self.img_size):
            for j in range(self.img_size):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                if dist < core_size * 2.5:
                    val = core_bright * np.exp(-dist / core_size)
                    # AGN cores are typically blue-white
                    img[i, j] += [val * 0.9, val * 0.95, val]

        # Diffraction spikes for bright AGN
        if agn_type in ['seyfert1', 'blazar'] and np.random.rand() > 0.5:
            spike_len = np.random.randint(15, 35)
            spike_bright = np.random.uniform(100, 160)
            for offset in range(-spike_len, spike_len):
                decay = 1 - abs(offset)/spike_len
                if 0 <= center + offset < self.img_size:
                    img[center, center + offset] += spike_bright * decay
                    img[center + offset, center] += spike_bright * decay

        # Background stars
        for _ in range(np.random.randint(25, 50)):
            si, sj = np.random.randint(0, self.img_size), np.random.randint(0, self.img_size)
            if np.sqrt((si-center)**2 + (sj-center)**2) > 20:
                sb = np.random.uniform(100, 200)
                img[si, sj] = [sb, sb, sb * 0.9]

        return np.clip(img, 0, 255).astype(np.uint8)


def generate_diverse_dataset():
    """Generate complete diverse dataset for 4 classes"""

    print(f"\nGenerating DIVERSE dataset with {CONFIG['num_samples']} samples...")
    print("Classes: galaxy, quasar, radio_galaxy, agn")
    print("Each sample is UNIQUE with realistic variation\n")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    images_dir = os.path.join(CONFIG['output_dir'], 'images')
    os.makedirs(images_dir, exist_ok=True)

    generator = AstronomicalObjectGenerator()

    object_types = ['galaxy', 'quasar', 'radio_galaxy', 'agn']
    samples_per_type = CONFIG['num_samples'] // len(object_types)

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
            print(f"\nGenerating {samples_per_type} {obj_type}s...")

            for i in tqdm(range(samples_per_type), desc=obj_type):
                seed = sample_id * 1000 + i + np.random.randint(0, 100000)

                if obj_type == 'galaxy':
                    signal = generator.generate_galaxy_signal(seed)
                    img = generator.generate_galaxy_image(seed)
                elif obj_type == 'quasar':
                    signal = generator.generate_quasar_signal(seed)
                    img = generator.generate_quasar_image(seed)
                elif obj_type == 'radio_galaxy':
                    signal = generator.generate_radio_galaxy_signal(seed)
                    img = generator.generate_radio_galaxy_image(seed)
                else:  # agn
                    signal = generator.generate_agn_signal(seed)
                    img = generator.generate_agn_image(seed)

                signals_dataset[sample_id] = signal

                img_filename = f'sample_{sample_id:05d}.jpg'
                img_path = os.path.join(images_dir, img_filename)
                Image.fromarray(img).save(img_path, quality=95)

                metadata.append({
                    'sample_id': sample_id,
                    'object_type': obj_type
                })

                sample_id += 1

        metadata_path = os.path.join(CONFIG['output_dir'], 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Generated {CONFIG['num_samples']} DIVERSE samples!")
    print(f"Output: {CONFIG['output_dir']}/")
    print(f"\nKey features:")
    print(f"  - 4 classes: galaxy, quasar, radio_galaxy, agn")
    print(f"  - {samples_per_type} samples per class")
    print(f"  - Multiple morphologies per class")
    print(f"  - Varied orientations, sizes, brightness")
    print(f"  - Galaxy: spiral, elliptical, irregular, ring, barred")
    print(f"  - Radio Galaxy: FR1, FR2, WAT, NAT, compact")
    print(f"  - AGN: seyfert1, seyfert2, liner, blazar")
    print(f"{'='*50}")


if __name__ == "__main__":
    print("\nThis will generate ~5200 diverse synthetic samples")
    print("Classes: galaxy, quasar, radio_galaxy, agn\n")

    response = input("Generate dataset? (y/n): ")

    if response.lower() == 'y':
        generate_diverse_dataset()
        print("\nDone! Use this dataset for pre-training!")
    else:
        print("Cancelled")
