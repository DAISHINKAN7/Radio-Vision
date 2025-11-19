"""
Radio Vision Backend API - Production Grade
Complete system with:
- Signal classification
- Multi-format upload
- Real-time generation
- Batch processing
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io
import json
import h5py
from pathlib import Path
import asyncio
from datetime import datetime
import logging
import base64
import tempfile
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.signal_classifier import SignalClassifier, LightweightClassifier, preprocess_signal, extract_signal_features
from models.pix2pix_gan import UNetGenerator

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = None
        self.classifier = None
        self.pix2pix_generator = None
        self.temp_dir = Path(tempfile.gettempdir()) / 'radio_vision'
        self.temp_dir.mkdir(exist_ok=True)
        
        # Upload tracking
        self.uploaded_signals = {}
        self.generation_cache = {}
        
    async def initialize(self):
        """Initialize all models and dataset"""
        logger.info("🔧 Initializing Radio Vision backend (Production)...")
        
        # Load dataset
        try:
            self.dataset = RadioVisionDataset('radio_vision_dataset_10k')
            logger.info(f"✅ Dataset loaded: {len(self.dataset.metadata)} samples")
        except Exception as e:
            logger.warning(f"⚠️  Dataset not found: {e}")
            self.dataset = None
        
        # Load Signal Classifier
        try:
            self.classifier = LightweightClassifier(num_classes=4).to(self.device)
            try:
                checkpoint = torch.load('models/checkpoints/lightweight_best.pth', map_location=self.device)
                self.classifier.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Classifier loaded from checkpoint")
            except:
                logger.info("⚠️  No pretrained classifier found")
            self.classifier.eval()
        except Exception as e:
            logger.error(f"❌ Failed to load classifier: {e}")
            self.classifier = None
        
        # Load Pix2Pix Generator
        try:
            self.pix2pix_generator = UNetGenerator().to(self.device)
            try:
                checkpoint = torch.load('models/checkpoints/pix2pix_best.pth', map_location=self.device)
                self.pix2pix_generator.load_state_dict(checkpoint['generator'])
                logger.info("✅ Pix2Pix generator loaded")
            except:
                logger.info("⚠️  No pretrained Pix2Pix found")
            self.pix2pix_generator.eval()
        except Exception as e:
            logger.error(f"❌ Failed to load Pix2Pix: {e}")
            self.pix2pix_generator = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await state.initialize()
    yield

app = FastAPI(
    title="Radio Vision API - Production",
    description="Complete system with classification, upload, and generation",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class ClassificationResult(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    features: Optional[Dict[str, float]] = None

class GenerationRequest(BaseModel):
    signal_id: Optional[str] = None
    sample_id: Optional[int] = None
    model: str = 'pix2pix'
    settings: Optional[Dict] = {}

class BatchGenerationRequest(BaseModel):
    signal_ids: List[str]
    model: str = 'pix2pix'

# ============================================================================
# HELPER CLASSES
# ============================================================================

class RadioVisionDataset:
    """Dataset manager"""
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        
        with open(self.dataset_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.signals_file = h5py.File(self.dataset_path / 'signals.h5', 'r')
        self.signals = self.signals_file['signals']
    
    def get_sample(self, sample_id: int):
        if sample_id >= len(self.metadata):
            raise ValueError(f"Sample ID {sample_id} out of range")
        
        sample = self.metadata[sample_id]
        signal = self.signals[sample_id]
        
        return {
            'signal': signal,
            'metadata': sample,
            'optical_image_path': self.dataset_path / sample['optical_image_path'],
            'radio_image_path': self.dataset_path / sample['radio_image_path']
        }
    
    def query(self, object_type=None, limit=100, offset=0):
        filtered = self.metadata
        if object_type:
            filtered = [s for s in filtered if s['object_type'] == object_type]
        return filtered[offset:offset+limit]

# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def parse_uploaded_signal(file_content: bytes, filename: str) -> np.ndarray:
    """Parse uploaded signal file"""
    
    ext = Path(filename).suffix.lower()
    
    try:
        if ext == '.csv':
            # CSV file
            signal = np.loadtxt(io.BytesIO(file_content), delimiter=',')
        
        elif ext == '.npy':
            # NumPy binary
            signal = np.load(io.BytesIO(file_content))
        
        elif ext == '.txt':
            # Text file
            signal = np.loadtxt(io.BytesIO(file_content))
        
        elif ext in ['.png', '.jpg', '.jpeg']:
            # Image file - convert to signal
            img = Image.open(io.BytesIO(file_content)).convert('L')
            signal = np.array(img).astype(np.float32) / 255.0
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Preprocess
        signal = preprocess_signal(signal)
        
        return signal
    
    except Exception as e:
        raise ValueError(f"Failed to parse signal: {e}")


def signal_to_radio_image(signal: np.ndarray) -> Image.Image:
    """Convert signal to radio image visualization"""
    # Normalize
    signal_norm = ((signal - signal.min()) / (signal.max() - signal.min() + 1e-8) * 255).astype(np.uint8)
    
    # Create false-color
    img_rgb = np.zeros((signal.shape[0], signal.shape[1], 3), dtype=np.uint8)
    img_rgb[:, :, 0] = signal_norm * 0.3
    img_rgb[:, :, 1] = signal_norm * 0.7
    img_rgb[:, :, 2] = signal_norm * 1.0
    
    img = Image.fromarray(img_rgb)
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    
    return img


def radio_image_to_tensor(radio_img: Image.Image) -> torch.Tensor:
    """Convert radio image to tensor for generator"""
    # To tensor and normalize
    img_array = np.array(radio_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor * 2 - 1  # [0,1] -> [-1,1]
    
    return img_tensor.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    # Denormalize
    img = (tensor.squeeze(0).cpu() + 1) / 2  # [-1,1] -> [0,1]
    img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
    img = (img * 255).astype(np.uint8)
    
    return Image.fromarray(img)


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def calculate_metrics(generated: torch.Tensor, ground_truth: torch.Tensor) -> Dict:
    """Calculate image quality metrics"""
    # Move to CPU and convert to numpy
    gen_np = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
    gt_np = ground_truth.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Denormalize
    gen_np = (gen_np + 1) / 2
    gt_np = (gt_np + 1) / 2
    
    # MSE
    mse = np.mean((gen_np - gt_np) ** 2)
    
    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = 100.0
    
    # Simple SSIM approximation
    mu_gen = np.mean(gen_np)
    mu_gt = np.mean(gt_np)
    sigma_gen = np.std(gen_np)
    sigma_gt = np.std(gt_np)
    sigma_gen_gt = np.mean((gen_np - mu_gen) * (gt_np - mu_gt))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu_gen * mu_gt + c1) * (2 * sigma_gen_gt + c2)) / \
           ((mu_gen**2 + mu_gt**2 + c1) * (sigma_gen**2 + sigma_gt**2 + c2))
    
    return {
        'psnr': float(psnr),
        'ssim': float(np.clip(ssim, 0, 1)),
        'mse': float(mse)
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "Radio Vision API - Production",
        "version": "3.0.0",
        "status": "operational",
        "features": [
            "Signal Classification",
            "Multi-format Upload",
            "Pix2Pix Generation",
            "Batch Processing",
            "Real Metrics"
        ],
        "models_loaded": {
            "classifier": state.classifier is not None,
            "pix2pix": state.pix2pix_generator is not None
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": state.device,
        "dataset": "loaded" if state.dataset else "not_loaded",
        "models": {
            "classifier": state.classifier is not None,
            "pix2pix": state.pix2pix_generator is not None
        }
    }

# ============================================================================
# UPLOAD ENDPOINTS
# ============================================================================

@app.post("/api/upload/signal")
async def upload_signal(file: UploadFile = File(...)):
    """Upload and process signal file"""
    
    try:
        # Read file
        content = await file.read()
        
        # Parse signal
        signal = parse_uploaded_signal(content, file.filename)
        
        # Generate unique ID
        signal_id = str(uuid.uuid4())
        
        # Classify signal
        if state.classifier:
            classification = state.classifier.predict(signal)
        else:
            classification = {
                'predicted_class': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
        
        # Extract features
        features = extract_signal_features(signal)
        
        # Create radio image
        radio_img = signal_to_radio_image(signal)
        
        # Store
        state.uploaded_signals[signal_id] = {
            'signal': signal,
            'radio_image': radio_img,
            'classification': classification,
            'features': features,
            'filename': file.filename,
            'uploaded_at': datetime.now().isoformat()
        }
        
        return {
            'success': True,
            'signal_id': signal_id,
            'classification': classification,
            'features': features,
            'radio_image': image_to_base64(radio_img),
            'shape': list(signal.shape)
        }
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload/manual")
async def upload_manual_signal(data: dict):
    """Upload manually entered signal data"""
    
    try:
        # Parse signal array
        signal_data = np.array(data.get('signal', []))
        
        if signal_data.size == 0:
            raise ValueError("Empty signal data")
        
        # Preprocess
        signal = preprocess_signal(signal_data)
        
        # Generate ID
        signal_id = str(uuid.uuid4())
        
        # Classify
        if state.classifier:
            classification = state.classifier.predict(signal)
        else:
            classification = {'predicted_class': 'unknown', 'confidence': 0.0}
        
        # Create radio image
        radio_img = signal_to_radio_image(signal)
        
        # Store
        state.uploaded_signals[signal_id] = {
            'signal': signal,
            'radio_image': radio_img,
            'classification': classification,
            'source': 'manual',
            'uploaded_at': datetime.now().isoformat()
        }
        
        return {
            'success': True,
            'signal_id': signal_id,
            'classification': classification,
            'radio_image': image_to_base64(radio_img)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# CLASSIFICATION ENDPOINTS
# ============================================================================

@app.post("/api/classify/signal")
async def classify_uploaded_signal(signal_id: str):
    """Classify an uploaded signal"""
    
    if signal_id not in state.uploaded_signals:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    if not state.classifier:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    signal = state.uploaded_signals[signal_id]['signal']
    classification = state.classifier.predict(signal)
    
    # Update stored classification
    state.uploaded_signals[signal_id]['classification'] = classification
    
    return classification

# ============================================================================
# GENERATION ENDPOINTS
# ============================================================================

@app.post("/api/generate")
async def generate_from_signal(request: GenerationRequest):
    """Generate optical image from signal"""
    
    try:
        # Get signal
        if request.signal_id:
            if request.signal_id not in state.uploaded_signals:
                raise HTTPException(status_code=404, detail="Signal not found")
            signal_data = state.uploaded_signals[request.signal_id]
            radio_img = signal_data['radio_image']
        elif request.sample_id is not None:
            if not state.dataset:
                raise HTTPException(status_code=503, detail="Dataset not loaded")
            sample = state.dataset.get_sample(request.sample_id)
            radio_img = Image.open(sample['radio_image_path']).convert('RGB')
            radio_img = radio_img.resize((256, 256))
        else:
            raise HTTPException(status_code=400, detail="Must provide signal_id or sample_id")
        
        # Generate with Pix2Pix
        if not state.pix2pix_generator:
            raise HTTPException(status_code=503, detail="Generator not loaded")
        
        # Convert to tensor
        radio_tensor = radio_image_to_tensor(radio_img).to(state.device)
        
        # Generate
        with torch.no_grad():
            generated_tensor = state.pix2pix_generator(radio_tensor)
        
        # Convert to image
        generated_img = tensor_to_image(generated_tensor)
        
        # Calculate metrics if we have ground truth
        metrics = {}
        if request.sample_id is not None:
            sample = state.dataset.get_sample(request.sample_id)
            gt_img = Image.open(sample['optical_image_path']).convert('RGB')
            gt_img = gt_img.resize((256, 256))
            gt_tensor = radio_image_to_tensor(gt_img).to(state.device)
            metrics = calculate_metrics(generated_tensor, gt_tensor)
        
        return {
            'success': True,
            'generated_image': image_to_base64(generated_img),
            'metrics': metrics,
            'model': 'pix2pix'
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/batch")
async def generate_batch(request: BatchGenerationRequest):
    """Generate images for multiple signals"""
    
    results = []
    
    for signal_id in request.signal_ids:
        try:
            gen_request = GenerationRequest(signal_id=signal_id, model=request.model)
            result = await generate_from_signal(gen_request)
            results.append({
                'signal_id': signal_id,
                'success': True,
                'result': result
            })
        except Exception as e:
            results.append({
                'signal_id': signal_id,
                'success': False,
                'error': str(e)
            })
    
    return {
        'total': len(request.signal_ids),
        'successful': sum(1 for r in results if r['success']),
        'results': results
    }

# ============================================================================
# DATASET ENDPOINTS (existing)
# ============================================================================

@app.get("/dataset/info")
async def dataset_info():
    if not state.dataset:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    return {
        'total_samples': len(state.dataset.metadata),
        'object_types': {
            'spiral_galaxy': sum(1 for s in state.dataset.metadata if s['object_type'] == 'spiral_galaxy'),
            'emission_nebula': sum(1 for s in state.dataset.metadata if s['object_type'] == 'emission_nebula'),
            'quasar': sum(1 for s in state.dataset.metadata if s['object_type'] == 'quasar'),
            'pulsar': sum(1 for s in state.dataset.metadata if s['object_type'] == 'pulsar')
        }
    }

@app.post("/dataset/query")
async def query_dataset(data: dict):
    if not state.dataset:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    results = state.dataset.query(
        object_type=data.get('object_type'),
        limit=data.get('limit', 100),
        offset=data.get('offset', 0)
    )
    
    return {'results': results, 'total': len(results)}

@app.get("/dataset/sample/{sample_id}/images/optical")
async def get_optical_image(sample_id: int):
    if not state.dataset:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    sample = state.dataset.get_sample(sample_id)
    return FileResponse(sample['optical_image_path'])

@app.get("/dataset/sample/{sample_id}/images/radio")
async def get_radio_image(sample_id: int):
    if not state.dataset:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    sample = state.dataset.get_sample(sample_id)
    return FileResponse(sample['radio_image_path'])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)