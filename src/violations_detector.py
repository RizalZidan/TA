"""
Simple Violations Detector
Using helmet.v2i.yolov8 dataset directly
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
try:
    from scaling_config import scaling_config
except ImportError:
    scaling_config = None

try:
    from src.face_recognition import FaceRecognitionSystem
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.face_recognition import FaceRecognitionSystem


class ViolationsDetector:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize Violations Detector with simple setup
        
        Args:
            confidence_threshold: Confidence threshold for detection
        """
        self.confidence_threshold = confidence_threshold
        
        # Try to use APD combined model first (has 4 classes)
        apd_model_path = os.path.join(os.path.dirname(__file__), "..", "apd_detection_combined3/best.pt")
        helmet_model_path = os.path.join(os.path.dirname(__file__), "..", "helmet.v2i.yolov8/helmet_vest_detection/yolov8n_50epochs_augmented/weights/best.pt")
        
        if os.path.exists(apd_model_path):
            print(f"🎯 Loading APD Combined model from {apd_model_path}")
            self.model = YOLO(apd_model_path)
            self.class_names = {0: 'Helmet', 1: 'No_Helmet', 2: 'Vest', 3: 'No_Vest'}
            self.use_apd_model = True
            print("📊 Using APD Combined model - Focusing on violations: No_Helmet, No_Vest")
        # Fallback to helmet.v2i.yolov8 dataset model
        elif os.path.exists(helmet_model_path):
            print(f"🎯 Loading helmet detection model from {helmet_model_path}")
            self.model = YOLO(helmet_model_path)
            self.class_names = {0: 'helmet', 1: 'vest'}
            self.use_apd_model = False
            print("📊 Using helmet.v2i.yolov8 dataset model (2 classes)")
        else:
            print("⚠️  No APD model found, using YOLOv8n for person detection")
            self.model = YOLO('yolov8n.pt')
            self.class_names = {0: 'person'}
            self.use_apd_model = False
        
        # Optimize for performance
        self.model.fuse()  # Fuse Conv and BatchNorm for faster inference
        
        # Initialize Face Recognition
        self.recognition_cooldowns = {} # (location) -> last_time
        try:
            self.face_recognizer = FaceRecognitionSystem()
            self.use_face_recognition = len(self.face_recognizer.face_encodings) > 0
            print(f"👤 Face Recognition: {'Active' if self.use_face_recognition else 'Inactive (No registered workers)'}")
        except Exception as e:
            print(f"⚠️ Failed to initialize Face Recognition: {e}")
            self.face_recognizer = None
            self.use_face_recognition = False
        
        print("✅ Optimized Violations Detector initialized")
        print(f"📊 Classes: {list(self.class_names.values())}")
        print(f"🎯 Confidence threshold: {self.confidence_threshold}")
        print("⚡ Performance optimizations enabled")
    
    def detect_violations(self, frame):
        """
        Detect APD violations only (No_Helmet, No_Vest) - Optimized with smart scaling
        
        Args:
            frame: Input image frame
            
        Returns:
            List of violation detections only
        """
        # Use optimized inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        violations = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names.get(class_id, 'unknown')
                    
                    # Simple size filtering - optimized
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    if bbox_width < 8 or bbox_height < 8:  # Reduced from 15 completely to catch distant Helmets
                        continue
                    
                    # Only process violations (No_Helmet, No_Vest)
                    if class_name in ['No_Helmet', 'No_Vest']:
                        # Apply smart scaling for better coverage
                        if scaling_config and scaling_config.use_smart_scaling:
                            scaled_bbox = scaling_config.apply_custom_scaling(
                                [int(x1), int(y1), int(x2), int(y2)], 
                                class_name
                            )
                        else:
                            # Fallback to original smart scaling
                            scaled_bbox = self._apply_smart_scaling(
                                [int(x1), int(y1), int(x2), int(y2)], 
                                class_name
                            )
                        
                        # Try to recognize face if active
                        worker_id = "Unknown"
                        
                        # PERFORMANCE FIX: Only try to recognize every 2 seconds for this location
                        current_time = datetime.now().timestamp()
                        # Use grid-based key for location to account for movement
                        loc_key = f"{camera_id}_{int(center_x/20)}_{int(center_y/20)}"
                        last_try = self.recognition_cooldowns.get(loc_key, 0)
                        
                        if self.use_face_recognition and (current_time - last_try > 2.0):
                            self.recognition_cooldowns[loc_key] = current_time
                            try:
                                # Crop the scaled bbox region
                                sx1, sy1, sx2, sy2 = scaled_bbox
                                crop_region = frame[max(0, sy1):min(frame.shape[0], sy2), 
                                                 max(0, sx1):min(frame.shape[1], sx2)]
                                
                                if crop_region.size > 0:
                                    recognized = None
                                    
                                    # Fallback: if 'nohelmet', use scaled_bbox directly
                                    if class_name == 'No_Helmet':
                                        recognized = self.face_recognizer.recognize_face(frame, scaled_bbox)
                                    
                                    # Still try Haar Cascade if needed
                                    if not recognized:
                                        faces = self.face_recognizer.detect_faces(crop_region)
                                        if faces:
                                            fx1, fy1, fx2, fy2 = faces[0]['bbox']
                                            global_face_bbox = [sx1 + fx1, sy1 + fy1, sx1 + fx2, sy1 + fy2]
                                            recognized = self.face_recognizer.recognize_face(frame, global_face_bbox)
                                            
                                    if recognized:
                                        metadata = self.face_recognizer.face_metadata.get(recognized, {})
                                        worker_id = metadata.get('name', recognized)
                            except Exception as e:
                                pass
                        
                        # Add violation detection
                        violations.append({
                            'bbox': scaled_bbox,
                            'class': class_name.lower().replace('_', ''),
                            'confidence': float(confidence),
                            'worker_id': worker_id,
                            'violation_severity': 'high',
                            'violation_info': {
                                'has_helmet': False,
                                'has_vest': False,
                                'is_violation': True,
                                'violation_type': class_name.lower().replace('_', '')
                            }
                        })
        
        return violations
    
    def _apply_smart_scaling(self, bbox, class_name):
        """
        Apply smart scaling to bounding box for better violation coverage
        
        Args:
            bbox: [x1, y1, x2, y2] original bounding box
            class_name: 'No_Helmet' or 'No_Vest'
            
        Returns:
            Scaled bounding box with better coverage
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        if class_name == 'No_Helmet':
            # For no helmet - expand to cover head area better
            # Head area typically needs more coverage
            expand_factor = 1.3  # Expand by 30%
            new_width = int(width * expand_factor)
            new_height = int(height * expand_factor)
            
            # Position to cover upper head area
            new_y1 = center_y - new_height // 2
            new_y2 = center_y + new_height // 2
            new_x1 = center_x - new_width // 2
            new_x2 = center_x + new_width // 2
            
        elif class_name == 'No_Vest':
            # For no vest - expand to cover torso area better
            # Vest area needs wider coverage
            expand_factor = 1.4  # Expand by 40%
            new_width = int(width * expand_factor)
            new_height = int(height * expand_factor)
            
            # Position to cover upper torso area
            new_y1 = center_y - new_height // 2
            new_y2 = center_y + new_height // 2
            new_x1 = center_x - new_width // 2
            new_x2 = center_x + new_width // 2
        
        else:
            # Return original if unknown class
            return [x1, y1, x2, y2]
        
        # Ensure bounds are within frame
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def detect_all_apd(self, frame):
        """
        Detect all APD items (same as detect_violations for simple setup)
        
        Args:
            frame: Input image frame
            
        Returns:
            List of all APD detections
        """
        return self.detect_violations(frame)
    
    def draw_violations(self, frame, detections):
        """
        Draw APD violations on frame - Optimized with scaling info
        
        Args:
            frame: Input frame
            detections: List of violation detections
            
        Returns:
            Frame with drawn violations
        """
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Color based on violation type
            worker_id = detection.get('worker_id', 'Unknown')
            if class_name == 'nohelmet':
                color = (0, 0, 255)  # Red for no helmet violation
                label = f"No Helmet {confidence:.2f} [{worker_id}]"
            elif class_name == 'novest':
                color = (0, 165, 255)  # Orange for no vest violation
                label = f"No Vest {confidence:.2f} [{worker_id}]"
            else:
                continue  # Skip unknown classes
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background - optimized
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0]  # Reduced font size
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)  # Thinner text
        
        return frame
    
    def draw_all_apd(self, frame, detections):
        """
        Draw all APD detections (same as draw_violations for simple setup)
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        return self.draw_violations(frame, detections)
