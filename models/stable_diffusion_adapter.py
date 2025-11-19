"""
Stable Diffusion Integration for Radio Vision
Adapter for using Stable Diffusion with radio signals
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class StableDiffusionRadioAdapter:
    """
    Adapter for Stable Diffusion to work with radio signals
    Note: Requires diffusers library installation
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model_loaded = False
        
        try:
            from diffusers import StableDiffusionPipeline
            self.StableDiffusionPipeline = StableDiffusionPipeline
            print("✅ Diffusers library available")
        except ImportError:
            print("⚠️  Stable Diffusion requires: pip install diffusers transformers accelerate")
            self.StableDiffusionPipeline = None
    
    def load_model(self, model_id="runwayml/stable-diffusion-v1-5"):
        """Load Stable Diffusion model"""
        if self.StableDiffusionPipeline is None:
            raise ImportError("Please install: pip install diffusers transformers accelerate")
        
        try:
            self.pipeline = self.StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None
            )
            self.pipeline = self.pipeline.to(self.device)
            self.model_loaded = True
            print(f"✅ Stable Diffusion loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load Stable Diffusion: {e}")
            self.model_loaded = False
    
    def signal_to_prompt(self, signal, object_type):
        """Convert signal characteristics to text prompt"""
        
        # Analyze signal
        mean_intensity = float(np.mean(signal))
        peak_intensity = float(np.max(signal))
        std_intensity = float(np.std(signal))
        
        # Base prompts for each object type
        base_prompts = {
            'spiral_galaxy': "spiral galaxy with bright core and spiral arms",
            'emission_nebula': "colorful emission nebula with gas clouds",
            'quasar': "bright quasar with intense central point",
            'pulsar': "pulsar with bright emission and jets"
        }
        
        base = base_prompts.get(object_type, "astronomical object")
        
        # Add signal characteristics to prompt
        if peak_intensity > 0.8:
            prompt = f"bright, high-contrast {base}"
        elif peak_intensity < 0.3:
            prompt = f"faint, dim {base}"
        else:
            prompt = f"{base}"
        
        if std_intensity > 0.3:
            prompt += ", complex structure, detailed features"
        else:
            prompt += ", smooth, diffuse structure"
        
        # Add quality terms
        prompt += ", astronomy photo, Hubble telescope, high quality, detailed, 4k"
        
        # Negative prompt
        negative_prompt = "blurry, low quality, distorted, artifacts, noise"
        
        return prompt, negative_prompt
    
    def generate(self, signal, object_type, num_inference_steps=50, guidance_scale=7.5):
        """
        Generate optical image from radio signal
        
        Args:
            signal: numpy array (128, 1024)
            object_type: predicted object type
            num_inference_steps: number of diffusion steps
            guidance_scale: classifier-free guidance scale
        
        Returns:
            PIL Image
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Generate prompt from signal
        prompt, negative_prompt = self.signal_to_prompt(signal, object_type)
        
        # Generate image
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            )
        
        return result.images[0]


class EnsembleGenerator:
    """
    Ensemble of multiple generation models
    Combines Pix2Pix and Stable Diffusion outputs
    """
    
    def __init__(self, pix2pix_model, stable_diffusion_model, device='cuda'):
        self.pix2pix = pix2pix_model
        self.stable_diffusion = stable_diffusion_model
        self.device = device
        
        # Learned weights (can be trained)
        self.weights = {
            'pix2pix': 0.6,
            'stable_diffusion': 0.4
        }
    
    def generate(self, radio_img_tensor, signal, object_type):
        """
        Generate using ensemble of models
        
        Args:
            radio_img_tensor: Radio image tensor for Pix2Pix (1, 3, 256, 256)
            signal: Raw signal for Stable Diffusion (128, 1024)
            object_type: Classification result
        
        Returns:
            Blended PIL Image
        """
        
        # Generate with Pix2Pix
        with torch.no_grad():
            pix2pix_output = self.pix2pix(radio_img_tensor)
        
        # Convert to PIL
        pix2pix_img = self._tensor_to_pil(pix2pix_output, size=512)
        
        # Generate with Stable Diffusion
        if self.stable_diffusion.model_loaded:
            sd_img = self.stable_diffusion.generate(signal, object_type)
        else:
            # Fallback: just use Pix2Pix
            return pix2pix_img
        
        # Blend images
        blended = self._blend_images(pix2pix_img, sd_img)
        
        return blended
    
    def _tensor_to_pil(self, tensor, size=512):
        """Convert tensor to PIL Image"""
        img = (tensor.squeeze(0).cpu() + 1) / 2  # [-1,1] -> [0,1]
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        return pil_img.resize((size, size), Image.Resampling.LANCZOS)
    
    def _blend_images(self, img1, img2):
        """Blend two images using learned weights"""
        
        # Convert to numpy
        arr1 = np.array(img1).astype(np.float32) / 255.0
        arr2 = np.array(img2).astype(np.float32) / 255.0
        
        # Weighted blend
        w1 = self.weights['pix2pix']
        w2 = self.weights['stable_diffusion']
        
        blended = arr1 * w1 + arr2 * w2
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(blended)


# Quick usage example
if __name__ == "__main__":
    print("Testing Stable Diffusion adapter...")
    
    # Create adapter
    adapter = StableDiffusionRadioAdapter(device='cpu')  # Use 'cuda' for GPU
    
    print("\nTo use Stable Diffusion:")
    print("1. Install: pip install diffusers transformers accelerate")
    print("2. Load model: adapter.load_model()")
    print("3. Generate: image = adapter.generate(signal, 'spiral_galaxy')")
    
    print("\n⚠️  Note: Stable Diffusion requires ~10GB download and 8GB+ VRAM")
    print("✅ Pix2Pix works great without it!")