@echo off
REM Master training script for Windows - Run all models in sequence
REM Adjust paths and parameters as needed

echo ===============================================
echo Radio-Vision Classifier - Complete Training
echo ===============================================
echo.

REM Set your data paths here
set TRAIN_RADIO=dataset/train/radio
set TRAIN_OPTICAL=dataset/train/optical
set VAL_RADIO=dataset/val/radio
set VAL_OPTICAL=dataset/val/optical

REM Create output directory
if not exist outputs mkdir outputs

echo Step 1/5: Training Improved MobileNetV2...
python training/train_mobilenet_improved.py ^
    --train_dir %TRAIN_RADIO% ^
    --val_dir %VAL_RADIO% ^
    --output_dir outputs/mobilenet_improved ^
    --batch_size 32 ^
    --epochs 100 ^
    --lr 1e-4 ^
    --weight_decay 0.05 ^
    --mixup_alpha 0.2 ^
    --cutmix_alpha 1.0 ^
    --ema_decay 0.9999 ^
    --use_tta

echo.
echo Step 2/5: Training ConvNeXt-V2...
python training/train_convnextv2.py ^
    --train_dir %TRAIN_OPTICAL% ^
    --val_dir %VAL_OPTICAL% ^
    --output_dir outputs/convnextv2_base ^
    --model_size base ^
    --batch_size 32 ^
    --epochs 100 ^
    --lr 5e-5 ^
    --weight_decay 0.05 ^
    --ema_decay 0.9999 ^
    --use_tta

echo.
echo Step 3/5: Training Multi-Modal Fusion...
python training/train_multimodal_fusion.py ^
    --train_radio_dir %TRAIN_RADIO% ^
    --train_optical_dir %TRAIN_OPTICAL% ^
    --val_radio_dir %VAL_RADIO% ^
    --val_optical_dir %VAL_OPTICAL% ^
    --output_dir outputs/multimodal_concat ^
    --fusion_type concat ^
    --batch_size 32 ^
    --epochs 100 ^
    --lr 1e-4 ^
    --weight_decay 0.05 ^
    --ema_decay 0.9999

echo.
echo Step 4/5: Creating Ensemble...
python training/train_ensemble.py ^
    --val_dir %VAL_RADIO% ^
    --output_dir outputs/final_ensemble ^
    --model_paths outputs/mobilenet_improved/best_ema_model.pth outputs/convnextv2_base/best_ema_model.pth outputs/multimodal_concat/best_ema_model.pth ^
    --model_types mobilenet convnext multimodal ^
    --ensemble_type soft_voting ^
    --batch_size 32

echo.
echo Step 5/5: Running Comprehensive Evaluation...
echo See evaluation/comprehensive_eval.py for detailed evaluation

echo.
echo ===============================================
echo Training Complete!
echo ===============================================
echo Results saved in outputs/
echo Check TRAINING_GUIDE.md for details
pause
