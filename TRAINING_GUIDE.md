# Comprehensive Training Guide for Radio-Vision Classifier

This guide provides step-by-step instructions to train all models in the optimal order for maximum accuracy.

## Prerequisites

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm scipy pillow
```

## Training Order and Commands

### Step 1: Train Improved MobileNetV2 (HIGHEST PRIORITY)

This is your current strongest model with the most improvements:
- Batch-level MixUp/CutMix (properly implemented)
- Mixed Precision Training (AMP) for 2x speedup
- Exponential Moving Average (EMA) for +1-2% accuracy
- Test-Time Augmentation (TTA) for +1.5-3.5% accuracy

**Command:**
```bash
python training/train_mobilenet_improved.py \
    --train_dir dataset/train/radio \
    --val_dir dataset/val/radio \
    --output_dir outputs/mobilenet_improved \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --mixup_alpha 0.2 \
    --cutmix_alpha 1.0 \
    --ema_decay 0.9999 \
    --use_tta
```

**Expected improvement:** +2-4% accuracy over your current MobileNetV2
**Training time:** ~2-3 hours on GPU

---

### Step 2: Train ConvNeXt-V2 Optical Classifier

Pure optical model with state-of-the-art architecture.

**Command:**
```bash
python training/train_convnextv2.py \
    --train_dir dataset/train/optical \
    --val_dir dataset/val/optical \
    --output_dir outputs/convnextv2_base \
    --model_size base \
    --batch_size 32 \
    --epochs 100 \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --ema_decay 0.9999 \
    --use_tta
```

**For better results (if you have GPU memory):**
```bash
# Use 'large' model instead of 'base'
python training/train_convnextv2.py \
    --train_dir dataset/train/optical \
    --val_dir dataset/val/optical \
    --output_dir outputs/convnextv2_large \
    --model_size large \
    --batch_size 16 \
    --epochs 100 \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --ema_decay 0.9999 \
    --use_tta
```

**Expected accuracy:** 35-45% (should beat current models)
**Training time:** 3-4 hours (base), 5-6 hours (large)

---

### Step 3: Train Multi-Modal Fusion Model

Combines radio + optical features for maximum performance.

**Command:**
```bash
python training/train_multimodal_fusion.py \
    --train_radio_dir dataset/train/radio \
    --train_optical_dir dataset/train/optical \
    --val_radio_dir dataset/val/radio \
    --val_optical_dir dataset/val/optical \
    --output_dir outputs/multimodal_concat \
    --fusion_type concat \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --ema_decay 0.9999
```

**Alternative fusion strategies:**
```bash
# Attention-based fusion (may perform better)
python training/train_multimodal_fusion.py \
    --train_radio_dir dataset/train/radio \
    --train_optical_dir dataset/train/optical \
    --val_radio_dir dataset/val/radio \
    --val_optical_dir dataset/val/optical \
    --output_dir outputs/multimodal_attention \
    --fusion_type attention \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4
```

**Expected accuracy:** 40-50% (4-8% jump over single branch)
**Training time:** 4-5 hours

---

### Step 4: Create Final Ensemble

Combine all your best models for maximum accuracy.

**Command:**
```bash
python training/train_ensemble.py \
    --val_dir dataset/val/radio \
    --output_dir outputs/final_ensemble \
    --model_paths \
        outputs/mobilenet_improved/best_ema_model.pth \
        outputs/convnextv2_base/best_ema_model.pth \
        outputs/multimodal_concat/best_ema_model.pth \
    --model_types mobilenet convnext multimodal \
    --ensemble_type soft_voting \
    --batch_size 32
```

**Expected accuracy:** 45-55% (best overall result)
**Time:** Few minutes (just evaluation)

---

### Step 5: Comprehensive Evaluation

Run detailed evaluation with all metrics journals care about.

**Command:**
```bash
python evaluation/comprehensive_eval.py \
    --model_path outputs/final_ensemble/best_model.pth \
    --model_type ensemble \
    --data_dir dataset/val \
    --output_dir evaluation_results \
    --batch_size 32
```

**Outputs:**
- Per-class precision/recall/F1
- Normalized confusion matrix
- Reliability diagrams + Expected Calibration Error
- Inference speed benchmarks (CPU + GPU)
- All metrics as JSON

---

## Training Schedule (Recommended)

### Week 1: Core Models
- **Day 1-2**: Train improved MobileNetV2 (Step 1)
- **Day 3-4**: Train ConvNeXt-V2 (Step 2)
- **Day 5-7**: Train multi-modal fusion (Step 3)

### Week 2: Ensemble & Evaluation
- **Day 1**: Create and evaluate ensemble (Step 4)
- **Day 2**: Run comprehensive evaluation (Step 5)
- **Day 3**: Analyze results, tune hyperparameters if needed
- **Day 4-7**: Optional: Train larger models, experiment with fusion types

---

## Expected Final Results

Based on your current ~28% baseline:

| Model | Expected Accuracy | Notes |
|-------|------------------|-------|
| Current MobileNetV2 | ~28% | Baseline |
| Improved MobileNetV2 | 32-36% | +AMP, proper MixUp, EMA, TTA |
| ConvNeXt-V2 Base | 35-40% | Strong optical baseline |
| ConvNeXt-V2 Large | 38-45% | Best single model |
| Multi-Modal Fusion | 40-50% | Combines radio+optical |
| **Final Ensemble** | **45-55%** | **Paper result** |

---

## Quick Start (Run Everything)

If you want to train everything sequentially:

```bash
# 1. MobileNetV2 Improved
python training/train_mobilenet_improved.py \
    --train_dir dataset/train/radio --val_dir dataset/val/radio \
    --output_dir outputs/mobilenet_improved --epochs 100 --use_tta

# 2. ConvNeXt-V2
python training/train_convnextv2.py \
    --train_dir dataset/train/optical --val_dir dataset/val/optical \
    --output_dir outputs/convnextv2_base --model_size base --epochs 100 --use_tta

# 3. Multi-Modal Fusion
python training/train_multimodal_fusion.py \
    --train_radio_dir dataset/train/radio --train_optical_dir dataset/train/optical \
    --val_radio_dir dataset/val/radio --val_optical_dir dataset/val/optical \
    --output_dir outputs/multimodal_concat --fusion_type concat --epochs 100

# 4. Ensemble
python training/train_ensemble.py \
    --val_dir dataset/val/radio \
    --model_paths outputs/mobilenet_improved/best_ema_model.pth \
                  outputs/convnextv2_base/best_ema_model.pth \
                  outputs/multimodal_concat/best_ema_model.pth \
    --model_types mobilenet convnext multimodal \
    --ensemble_type soft_voting \
    --output_dir outputs/final_ensemble

# 5. Evaluation
python evaluation/comprehensive_eval.py \
    --model_path outputs/final_ensemble/best_model.pth \
    --model_type ensemble \
    --data_dir dataset/val \
    --output_dir evaluation_results
```

---

## Tips for Maximum Performance

1. **Use the best checkpoint**: Always use the EMA model (`best_ema_model.pth`) for final evaluation

2. **Enable TTA for final results**: Add `--use_tta` flag when training is done

3. **Adjust batch size**: If you run out of memory:
   - Reduce `--batch_size` (16 or 8)
   - Use gradient accumulation (not implemented but easy to add)

4. **Learning rate tuning**: If accuracy plateaus early:
   - Lower LR: `--lr 5e-5` or `--lr 1e-5`
   - Increase epochs: `--epochs 150` or `--epochs 200`

5. **Data augmentation**: For overfitting:
   - Increase augmentation strength in code
   - Use more aggressive CutMix: `--cutmix_alpha 2.0`

6. **Monitor training**: Check `outputs/*/training_curves.png` regularly

---

## Troubleshooting

**Out of memory:**
```bash
# Reduce batch size
--batch_size 16  # or even 8
```

**Training too slow:**
```bash
# Already using AMP (mixed precision), but you can:
--num_workers 8  # More data loading threads (if you have CPU cores)
```

**Low accuracy:**
- Check data loading (are images correct?)
- Check class balance (use Focal Loss if imbalanced)
- Try longer training: `--epochs 200`
- Use larger models: ConvNeXt-V2 Large

**Model not improving:**
- Lower learning rate by 10x
- Check if validation loss is decreasing
- Enable more augmentation

---

## File Organization

After training, your directory will look like:

```
outputs/
├── mobilenet_improved/
│   ├── best_model.pth
│   ├── best_ema_model.pth
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── metrics.json
├── convnextv2_base/
│   ├── ...
├── multimodal_concat/
│   ├── ...
└── final_ensemble/
    ├── ensemble_results.json
    └── ...

evaluation_results/
├── classification_report.txt
├── confusion_matrix.png
├── reliability_diagram.png
├── metrics.json
└── ...
```

---

## What to Report in Paper

From the evaluation results, report:

1. **Main Result**: Final ensemble accuracy (e.g., "Our ensemble achieves 52.3% accuracy")
2. **Ablation**: Individual model accuracies (Table 1)
3. **Per-Class**: Precision/Recall/F1 for each class (Table 2)
4. **Calibration**: ECE score (e.g., "ECE of 0.045 indicates well-calibrated model")
5. **Speed**: Inference time (e.g., "15ms on GPU, 45ms on CPU")
6. **Confusion Matrix**: As a figure showing where model confuses classes

Good luck! Start with Step 1 (improved MobileNetV2) - it has the highest ROI.
