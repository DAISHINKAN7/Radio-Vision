# Radio-Vision: Quick Start Guide

**Get up and running in 3 steps**

---

## Prerequisites

```bash
# Install dependencies
pip install torch torchvision numpy pandas matplotlib scikit-learn \
  pillow tqdm h5py scipy requests albumentations astropy astroquery

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Step 1: Collect Data (6 hours)

```bash
cd data_collection
python collect_5000_samples.py
```

**Output**: `radio_vision_dataset_5k/` with 5000+ samples

**Can skip?** Yes, if you already have a dataset

---

## Step 2: Train Classifier (8 hours)

```bash
cd training

# Full pipeline (pre-train + fine-tune)
python train_transfer_learning.py \
  --synthetic_path ../synthetic_dataset \
  --real_path ../radio_vision_dataset_5k \
  --output_dir ../outputs/classifier_v2 \
  --batch_size 16 \
  --pretrain_epochs 50 \
  --finetune_epochs 100

# OR quick (fine-tune only)
python train_transfer_learning.py \
  --skip_pretrain \
  --real_path ../radio_vision_dataset_5k \
  --output_dir ../outputs/classifier_quick \
  --finetune_epochs 150 \
  --batch_size 8
```

**Expected result**: 75-85% validation accuracy

**Outputs**:
- `outputs/classifier_v2/best_model.pt`
- `outputs/classifier_v2/training_curves.png`
- `outputs/classifier_v2/confusion_matrix.png`

---

## Step 3: Train GAN (12 hours)

```bash
cd training

python train_gan_improved.py \
  --train_path ../radio_vision_dataset_5k \
  --output_dir ../outputs/gan_improved \
  --batch_size 8 \
  --num_epochs 200 \
  --use_attention \
  --use_perceptual
```

**Expected result**: PSNR 25-28 dB, SSIM 0.72-0.78

**Outputs**:
- `outputs/gan_improved/best_generator.pt`
- `outputs/gan_improved/gan_training_curves.png`
- `outputs/gan_improved/samples/*.png`

---

## Monitor Progress

```bash
# Watch training
tail -f outputs/classifier_v2/*.log
tail -f outputs/gan_improved/*.log

# Check GPU usage
watch -n 1 nvidia-smi

# View results
ls outputs/classifier_v2/
ls outputs/gan_improved/samples/
```

---

## Common Issues

### Out of Memory
```bash
# Reduce batch size
--batch_size 4
```

### Data collection failing
```bash
# Increase delays in collect_5000_samples.py
CONFIG = {'rate_limit_delay': 1.0, 'timeout': 60}
```

### Training interrupted
```bash
# Just re-run - it will resume from last checkpoint
python train_transfer_learning.py ...  # Resumes automatically
```

---

## Expected Performance

| Metric | Before | After |
|--------|--------|-------|
| **Classifier Accuracy** | 13-39% | 75-85% |
| **Dataset Size** | 576 | 5000+ |
| **Model Parameters** | 22M | 3.5M |
| **GAN PSNR** | ~18 dB | 25-28 dB |
| **GAN SSIM** | ~0.45 | 0.72-0.78 |

---

## Next Steps

See **[RUNNING_GUIDE.md](RUNNING_GUIDE.md)** for:
- Detailed parameter explanations
- Advanced configurations
- Troubleshooting guide
- Performance benchmarks

---

**Total time from scratch: ~26 hours**

**Your improved Radio-Vision system is ready! 🚀**
