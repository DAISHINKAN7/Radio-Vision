import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json
import random

class RadioVisionDataGenerator:
    """Generate synthetic radio and optical image pairs for celestial objects"""
    
    def __init__(self, image_size=256):
        self.image_size = image_size
        self.object_types = ['spiral_galaxy', 'emission_nebula', 'quasar', 'pulsar']
    
    def generate_spiral_galaxy(self):
        """Generate spiral galaxy radio and optical images"""
        radio = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        optical = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        center = self.image_size // 2
        
        # Radio: Interferometry pattern - concentric rings with spiral modulation
        for r in range(20, center - 20, 5):
            for theta in np.linspace(0, 2*np.pi, 360):
                # Spiral arm modulation
                intensity = 150 + 50 * np.sin(2*theta + r/10)
                x = int(center + r * np.cos(theta))
                y = int(center + r * np.sin(theta))
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    radio[y, x] = [0, int(intensity * 0.7), int(intensity)]  # Blue-cyan
        
        # Optical: Beautiful spiral structure
        img = Image.fromarray(optical)
        draw = ImageDraw.Draw(img)
        
        # Core
        draw.ellipse([center-15, center-15, center+15, center+15], fill=(255, 220, 150))
        
        # Spiral arms
        for arm in range(2):
            for r in range(20, 100, 2):
                theta_offset = arm * np.pi
                for dt in np.linspace(0, 2*np.pi, 180):
                    theta = dt + theta_offset + r/30
                    x = center + int(r * np.cos(theta))
                    y = center + int(r * np.sin(theta))
                    intensity = int(200 - r*1.5)
                    if 0 <= x < self.image_size and 0 <= y < self.image_size:
                        draw.point((x, y), fill=(intensity, intensity, 255))
        
        # Add stars
        for _ in range(50):
            x, y = random.randint(0, self.image_size-1), random.randint(0, self.image_size-1)
            draw.point((x, y), fill=(255, 255, 255))
        
        optical = np.array(img.filter(ImageFilter.GaussianBlur(radius=1)))
        
        return radio, optical
    
    def generate_emission_nebula(self):
        """Generate emission nebula radio and optical images"""
        radio = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        optical = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        center = self.image_size // 2
        
        # Radio: Diffuse cloud structure
        for _ in range(1000):
            x = int(np.random.normal(center, 40))
            y = int(np.random.normal(center, 40))
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                intensity = random.randint(100, 200)
                radio[y, x] = [0, int(intensity * 0.6), intensity]
        
        # Optical: Colorful nebula
        img = Image.fromarray(optical)
        draw = ImageDraw.Draw(img)
        
        # Multiple color clouds
        colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50)]
        for color in colors:
            cx = center + random.randint(-30, 30)
            cy = center + random.randint(-30, 30)
            for _ in range(500):
                x = int(np.random.normal(cx, 25))
                y = int(np.random.normal(cy, 25))
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    intensity = random.random()
                    draw.point((x, y), fill=tuple(int(c * intensity) for c in color))
        
        # Stars
        for _ in range(100):
            x, y = random.randint(0, self.image_size-1), random.randint(0, self.image_size-1)
            draw.point((x, y), fill=(255, 255, 255))
        
        optical = np.array(img.filter(ImageFilter.GaussianBlur(radius=2)))
        radio = np.array(Image.fromarray(radio).filter(ImageFilter.GaussianBlur(radius=3)))
        
        return radio, optical
    
    def generate_quasar(self):
        """Generate quasar radio and optical images"""
        radio = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        optical = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        center = self.image_size // 2
        
        # Radio: Bright core with jets
        img_radio = Image.fromarray(radio)
        draw_radio = ImageDraw.Draw(img_radio)
        
        # Core
        draw_radio.ellipse([center-10, center-10, center+10, center+10], fill=(0, 200, 255))
        
        # Jets
        for angle in [45, 225]:  # Two opposite jets
            for r in range(15, 80, 2):
                x = center + int(r * np.cos(np.radians(angle)))
                y = center + int(r * np.sin(np.radians(angle)))
                intensity = int(200 - r)
                draw_radio.ellipse([x-2, y-2, x+2, y+2], fill=(0, intensity, int(intensity*1.2)))
        
        radio = np.array(img_radio.filter(ImageFilter.GaussianBlur(radius=1)))
        
        # Optical: Bright point source
        img_optical = Image.fromarray(optical)
        draw_optical = ImageDraw.Draw(img_optical)
        
        # Intense core
        draw_optical.ellipse([center-8, center-8, center+8, center+8], fill=(255, 255, 255))
        draw_optical.ellipse([center-15, center-15, center+15, center+15], fill=(255, 200, 150))
        
        # Diffraction spikes
        for angle in [0, 90, 180, 270]:
            for r in range(10, 50):
                x = center + int(r * np.cos(np.radians(angle)))
                y = center + int(r * np.sin(np.radians(angle)))
                intensity = int(255 - r*4)
                if intensity > 0:
                    draw_optical.point((x, y), fill=(intensity, intensity, intensity))
        
        # Background stars
        for _ in range(80):
            x, y = random.randint(0, self.image_size-1), random.randint(0, self.image_size-1)
            draw_optical.point((x, y), fill=(255, 255, 255))
        
        optical = np.array(img_optical.filter(ImageFilter.GaussianBlur(radius=1)))
        
        return radio, optical
    
    def generate_pulsar(self):
        """Generate pulsar radio and optical images"""
        radio = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        optical = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        center = self.image_size // 2
        
        # Radio: Periodic beam pattern
        img_radio = Image.fromarray(radio)
        draw_radio = ImageDraw.Draw(img_radio)
        
        # Core
        draw_radio.ellipse([center-5, center-5, center+5, center+5], fill=(0, 255, 255))
        
        # Beams
        for beam_angle in [30, 210]:  # Two beams
            for r in range(10, 90, 3):
                # Periodic intensity (simulating pulses)
                intensity = int(180 + 50 * np.sin(r / 5))
                angle = np.radians(beam_angle)
                # Beam width increases with distance
                for da in np.linspace(-5, 5, 11):
                    a = angle + np.radians(da)
                    x = center + int(r * np.cos(a))
                    y = center + int(r * np.sin(a))
                    if 0 <= x < self.image_size and 0 <= y < self.image_size:
                        draw_radio.point((x, y), fill=(0, int(intensity*0.7), intensity))
        
        radio = np.array(img_radio.filter(ImageFilter.GaussianBlur(radius=1)))
        
        # Optical: Compact object with accretion disk
        img_optical = Image.fromarray(optical)
        draw_optical = ImageDraw.Draw(img_optical)
        
        # Central neutron star
        draw_optical.ellipse([center-3, center-3, center+3, center+3], fill=(255, 255, 255))
        
        # Accretion disk
        for r in range(5, 25):
            for theta in np.linspace(0, 2*np.pi, 180):
                x = int(center + r * np.cos(theta))
                y = int(center + r * 0.3 * np.sin(theta))  # Flattened disk
                intensity = int(200 - r*5)
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    draw_optical.point((x, y), fill=(intensity, intensity, int(intensity*0.8)))
        
        # Background stars
        for _ in range(60):
            x, y = random.randint(0, self.image_size-1), random.randint(0, self.image_size-1)
            draw_optical.point((x, y), fill=(255, 255, 255))
        
        optical = np.array(img_optical.filter(ImageFilter.GaussianBlur(radius=0.5)))
        
        return radio, optical
    
    def generate_pair(self, object_type=None):
        """Generate a matched radio-optical pair"""
        if object_type is None:
            object_type = random.choice(self.object_types)
        
        if object_type == 'spiral_galaxy':
            radio, optical = self.generate_spiral_galaxy()
        elif object_type == 'emission_nebula':
            radio, optical = self.generate_emission_nebula()
        elif object_type == 'quasar':
            radio, optical = self.generate_quasar()
        elif object_type == 'pulsar':
            radio, optical = self.generate_pulsar()
        else:
            raise ValueError(f"Unknown object type: {object_type}")
        
        # Add noise to make it realistic
        radio = self.add_noise(radio, noise_level=10)
        optical = self.add_noise(optical, noise_level=5)
        
        return radio, optical, object_type
    
    def add_noise(self, image, noise_level=10):
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image
    
    def generate_dataset(self, num_samples=1000, output_dir='data'):
        """Generate a full dataset"""
        import os
        os.makedirs(f"{output_dir}/radio", exist_ok=True)
        os.makedirs(f"{output_dir}/optical", exist_ok=True)
        
        metadata = []
        
        for i in range(num_samples):
            radio, optical, obj_type = self.generate_pair()
            
            # Save images
            Image.fromarray(radio).save(f"{output_dir}/radio/{i:05d}.png")
            Image.fromarray(optical).save(f"{output_dir}/optical/{i:05d}.png")
            
            metadata.append({
                'id': i,
                'object_type': obj_type,
                'radio_path': f"radio/{i:05d}.png",
                'optical_path': f"optical/{i:05d}.png"
            })
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples...")
        
        # Save metadata
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"   Total samples: {num_samples}")
        print(f"   Output directory: {output_dir}")

if __name__ == "__main__":
    print("🔬 Radio Vision Data Generator")
    print("=" * 50)
    
    generator = RadioVisionDataGenerator(image_size=256)
    
    # Generate sample pairs for each object type
    print("\nGenerating sample pairs...")
    for obj_type in generator.object_types:
        radio, optical, _ = generator.generate_pair(obj_type)
        Image.fromarray(radio).save(f"sample_{obj_type}_radio.png")
        Image.fromarray(optical).save(f"sample_{obj_type}_optical.png")
        print(f"✅ Generated {obj_type} pair")
    
    print("\nSample images saved!")
    print("Run with generate_dataset() to create full training set.")