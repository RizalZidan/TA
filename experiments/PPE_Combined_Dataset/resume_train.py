from ultralytics import YOLO
import os

def resume_training():
    # Path to the last.pt checkpoint
    checkpoint_path = r'C:\Users\RizalZidan\Downloads\APDNYELL\experiments\PPE_Combined_Dataset\runs\ppe_retraining_50_epochs_v2\weights\last.pt'
    
    if os.path.exists(checkpoint_path):
        # Load the model from checkpoint
        model = YOLO(checkpoint_path)
        
        # Resume training
        print(f"🚀 Resuming training from: {checkpoint_path}")
        results = model.train(resume=True)
        
        print("Training Continued and Completed!")
    else:
        print(f"❌ Checkpoint not found at: {checkpoint_path}")

if __name__ == '__main__':
    resume_training()
