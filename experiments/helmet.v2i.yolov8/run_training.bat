@echo off
echo Starting YOLOv8 Training for Helmet and Vest Detection
echo ========================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python first.
    pause
    exit /b 1
)

REM Install required packages
echo Installing required packages...
pip install ultralytics albumentations opencv-python torch torchvision

REM Run training
echo.
echo Starting training with 50 epochs...
echo Classes: helmet, vest
echo Augmentations: flip, 90-degree rotation, brightness, blur, noise, crop
echo.

python train.py

echo.
echo Training completed! Check the 'helmet_vest_detection' directory for results.
pause
