from ultralytics import YOLO
import os

def train_model_v2():
    # Path to the data.yaml
    dataset_path = r'C:\Users\RizalZidan\Downloads\APDNYELL\experiments\PPE_Combined_Dataset'
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    
    # Load a model
    model = YOLO('yolov8n.pt')
    
    # Train the model with 50 epochs and enhanced augmentations
    # Optimized for complex site environments (glare, distant objects)
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        name='ppe_retraining_50_epochs_v2',
        project=os.path.join(dataset_path, 'runs'),
        device=0,
        # --- AUGMENTATION SETTINGS ---
        hsv_h=0.015, # color jitter
        hsv_s=0.7,   # saturation jitter
        hsv_v=0.4,   # brightness jitter (HANDLES GLARE)
        degrees=0.0, # rotation
        translate=0.1,
        scale=0.5,   # scale jitter (HANDLES DISTANT OBJECTS)
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,  # mix images to handle crowded scenes
        mixup=0.0,
        copy_paste=0.0
    )
    
    print("Training Complete!")
    print(f"Results saved to: {results.save_dir}")

if __name__ == '__main__':
    train_model_v2()
