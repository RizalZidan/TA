from ultralytics import YOLO
import os

def train_model():
    # Path to the data.yaml
    dataset_path = r'C:\Users\RizalZidan\Downloads\APDNYELL\experiments\PPE_Combined_Dataset'
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    
    # Load a model
    # We use yolov8n.pt for speed and efficiency
    model = YOLO('yolov8n.pt')
    
    # Train the model
    # User requested 30 epochs
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        name='ppe_retraining_30_epochs',
        project=os.path.join(dataset_path, 'runs'),
        device=0 # Use CUDA GPU
    )
    
    print("Training Complete!")
    print(f"Results saved to: {results.save_dir}")

if __name__ == '__main__':
    train_model()
