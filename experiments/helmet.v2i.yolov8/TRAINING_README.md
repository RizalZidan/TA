# Helmet and Vest Detection Training

## Overview
This training script configures YOLOv8 for detecting helmets and vests with 50 epochs and comprehensive augmentations.

## Dataset Information
- **Classes**: helmet, vest (2 classes)
- **Dataset**: helmet.v2i.yolov8
- **Training Images**: 2,043
- **Validation Images**: 195
- **Test Images**: 97

## Training Configuration
- **Epochs**: 50
- **Model**: YOLOv8 Nano (yolov8n.pt)
- **Batch Size**: 16
- **Image Size**: 640x640
- **Learning Rate**: 0.01 (initial)

## Augmentations Applied
1. **Geometric Augmentations**:
   - Horizontal flip (50% probability)
   - Vertical flip (50% probability)
   - 90-degree rotation (±90° range)
   - Translation (±10%)
   - Scaling (±50%)
   - Random crop (40% probability)

2. **Color Augmentations**:
   - Brightness variation (±40%)
   - Contrast adjustment (±30%)
   - Hue shift (±1.5%)
   - Saturation variation (±70%)

3. **Noise and Blur**:
   - Gaussian blur (30% probability)
   - Motion blur (20% probability)
   - Gaussian noise (30% probability)

4. **Advanced Augmentations**:
   - Mosaic (100% probability)
   - Random erasing (40% probability)
   - Auto augment (RandAugment)

## Files Created
1. **train_helmet_vest.py** - Main training script with custom augmentations
2. **training_config.yaml** - Configuration file with all training parameters
3. **run_training.bat** - Windows batch file to run training with one click

## How to Run Training

### Option 1: Run the batch file (Recommended)
```bash
run_training.bat
```

### Option 2: Run the Python script directly
```bash
pip install ultralytics albumentations opencv-python torch torchvision
python train_helmet_vest.py
```

### Option 3: Use Ultralytics CLI
```bash
pip install ultralytics
yolo train data=data.yaml epochs=50 imgsz=640 batch=16
```

## Expected Output
- Training will run for 50 epochs
- Best model will be saved in `helmet_vest_detection/yolov8n_50epochs_augmented/weights/best.pt`
- Training logs and metrics will be saved in the same directory
- Validation will be performed automatically after training

## Monitoring Training
- Training progress will be displayed in real-time
- Metrics include mAP@50, mAP@50-95, precision, recall, and loss values
- TensorBoard logs are automatically generated for visualization

## Post-Training
After training completes, you can:
1. Test the model on your test dataset
2. Use the best model for inference on new images
3. Fine-tune further if needed

## Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with 4GB+ VRAM for faster training
- **Storage**: ~2GB free space for models and logs
