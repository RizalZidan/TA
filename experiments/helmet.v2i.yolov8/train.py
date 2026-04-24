#!/usr/bin/env python3
"""
Consolidated YOLOv8 Training Script
Loads configuration from training_config.yaml
"""

import os
import yaml
from ultralytics import YOLO
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model():
    # Base directory of the script
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / 'training_config.yaml'
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return

    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Resolve data path
    data_path = config.get('data_path', 'data.yaml')
    if not os.path.isabs(data_path):
        data_path = str(base_dir / data_path)
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset yaml not found at {data_path}")
        return

    # Initialize YOLO model
    model_name = config.get('model', 'yolov8n.pt')
    print(f"Initializing YOLO model: {model_name}")
    model = YOLO(model_name)
    
    # Prepare training arguments
    # Merge basic params and augmentations
    train_args = {
        'data': data_path,
        'epochs': config.get('epochs', 50),
        'imgsz': config.get('image_size', 640),
        'batch': config.get('batch_size', 16),
        'workers': config.get('workers', 4),
        'device': config.get('device', 0),
        'project': config.get('project', 'helmet_vest_detection'),
        'name': config.get('name', 'experiment'),
        'exist_ok': config.get('exist_ok', True),
        'pretrained': config.get('pretrained', True),
        'lr0': config.get('lr0', 0.01),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'warmup_epochs': config.get('warmup_epochs', 3),
        'box': config.get('box_weight', 7.5),
        'cls': config.get('cls_weight', 0.5),
        'dfl': config.get('dfl_weight', 1.5),
        'save_period': config.get('save_period', -1),
    }
    
    # Add augmentations from config
    if 'augmentations' in config:
        train_args.update(config['augmentations'])
    
    print("\nTraining Parameters:")
    for k, v in train_args.items():
        if k != 'data': # skip printing data path full for brevity if needed
            print(f"  {k}: {v}")
    
    print("-" * 60)
    print("Starting Training...")
    
    # Start training
    results = model.train(**train_args)
    
    print("\nTraining completed!")
    print(f"Best model saved at: {results.save_dir}")
    
    # Validation
    print("\nRunning final validation...")
    model.val()

if __name__ == "__main__":
    # Ensure dependencies are installed
    try:
        import ultralytics
        import yaml
    except ImportError:
        print("Installing required dependencies...")
        os.system("pip install ultralytics pyyaml")
        
    train_model()
