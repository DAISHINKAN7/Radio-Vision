"""
STABLE DIFFUSION INTEGRATION - WORKING VERSION
Radio-guided optical image generation using Stable Diffusion

Install: pip install diffusers transformers accelerate
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler
from PIL import Image
import numpy as np


class StableDiffusionRadioAdapter:
    """Generate optical images from radio signals using Stable Diffusion"""
    
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        
        print(f"Loading Stable Diffusion from: {model_id}")
        
        # Load pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
        
        self.pipe = self.pipe.to(device)
        
        # Optimize
        if torch.cuda.is_available():
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        # Faster scheduler
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        print(f"✅ Stable Diffusion loaded on {device}")
    
    def signal_to_init_image(self, signal):
        """Convert radio signal to RGB image"""
        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()
        
        signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        signal_uint8 = (signal * 255).astype(np.uint8)
        
        img = Image.fromarray(signal_uint8)
        img = img.resize((512, 512), Image.LANCZOS)
        return img.convert('RGB')
    
    def generate(self, signal, prompt=None, negative_prompt=None, 
                 num_inference_steps=50, guidance_scale=7.5, strength=0.75):
        """Generate optical image"""
        
        init_image = self.signal_to_init_image(signal)
        
        if prompt is None:
            prompt = self._auto_prompt(signal)
        
        if negative_prompt is None:
            negative_prompt = "blurry, low quality, distorted, noise"
        
        print(f"Generating with: {prompt}")
        
        with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu'):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
            )
        
        return result.images[0]
    
    def _auto_prompt(self, signal):
        """Auto-generate prompt from signal"""
        import random
        
        mean_intensity = signal.mean()
        max_intensity = signal.max()
        std_intensity = signal.std()
        
        if max_intensity > 0.8 and std_intensity > 0.2:
            prompts = [
                "A distant quasar in deep space, bright point source, stars, astronomical image",
                "An active galactic nucleus, bright core, cosmic scene, space photography"
            ]
        elif std_intensity > 0.15:
            prompts = [
                "A spiral galaxy, elegant spiral arms, stars, cosmic dust, astronomical image",
                "A beautiful spiral galaxy with distinct arms, space photography"
            ]
        elif mean_intensity > 0.3:
            prompts = [
                "A glowing emission nebula, colorful gas clouds, stars, astronomical image",
                "A beautiful nebula with glowing gases, space photography"
            ]
        else:
            prompts = [
                "A deep space astronomical object, stars, cosmic scene, space photography",
                "A celestial object in the universe, astronomical observation"
            ]
        
        return random.choice(prompts)


def create_sd_generator(model_id="stabilityai/stable-diffusion-2-1", device=None):
    """Create Stable Diffusion generator"""
    return StableDiffusionRadioAdapter(model_id=model_id, device=device)


if __name__ == '__main__':
    print("Testing...")
    signal = np.random.rand(128, 1024)
    sd = create_sd_generator()
    result = sd.generate(signal, num_inference_steps=30)
    result.save('test_sd_output.png')
    print("✅ Saved to: test_sd_output.png")