"""
Face Recognition Module using Cosine Similarity
Implements face detection, feature extraction, and recognition
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

class FaceRecognitionSystem:
    def __init__(self, similarity_threshold=0.45):
        """
        Initialize Face Recognition System
        
        Args:
            similarity_threshold: Threshold for face recognition (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.face_encodings = {}
        self.face_metadata = {}
        self.database_path = "data/face_database.pkl"
        
        # Create data directory if not exists
        os.makedirs("data", exist_ok=True)
        
        # Load existing face database
        self.load_face_database()
        
        print("✅ Face Recognition System initialized")
        print(f"🎯 Similarity threshold: {self.similarity_threshold}")
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using OpenCV Haar Cascade
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face detections with bounding boxes
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_detections = []
        for (x, y, w, h) in faces:
            face_info = {
                'bbox': [x, y, x + w, y + h],
                'confidence': 1.0  # Haar Cascade doesn't provide confidence
            }
            face_detections.append(face_info)
        
        return face_detections
    
    def extract_face_features(self, frame, bbox):
        """
        Extract face features using simple histogram features
        
        Args:
            frame: Input image frame
            bbox: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            Face feature vector or None if face cannot be processed
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # Ensure positive region (x2 > x1, y2 > y1)
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Extract face region
        face_region = frame[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return None
        
        try:
            # Resize to standard size (64x64 for Histograms)
            face_resized = cv2.resize(face_region, (64, 64))
            
            # Convert to different color spaces
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Standardize face image (CLAHE for much better lighting robustness)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            face_gray = clahe.apply(face_gray)
            
            # Re-convert to color after equalization if needed or use gray for pixels
            face_hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
            
            # Extract features
            # 1. Color Histograms
            hist_b = cv2.calcHist([face_resized], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([face_resized], [1], None, [16], [0, 256])
            hist_r = cv2.calcHist([face_resized], [2], None, [16], [0, 256])
            hist_gray = cv2.calcHist([face_gray], [0], None, [16], [0, 256])
            hist_h = cv2.calcHist([face_hsv], [0], None, [8], [0, 180])
            hist_s = cv2.calcHist([face_hsv], [1], None, [8], [0, 256])
            
            # Normalize histograms
            hist_b = cv2.normalize(hist_b, hist_b).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            
            # 2. EDGE & SHAPE FEATURES (HOG Lite)
            # Use larger strides for much higher FPS
            try:
                win_size = (64, 64)
                block_size = (16, 16)
                block_stride = (16, 16) 
                cell_size = (16, 16)
                nbins = 9
                hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
                hog_features = hog.compute(face_gray).flatten()
            except:
                hog_features = np.zeros(324, dtype=np.float32)
            
            # 3. LBP (Local Binary Patterns) - Lightweight & Robust Texture
            # Hand-coded simple LBP for speed and accuracy
            lbp = np.zeros_like(face_gray)
            # Optimized simple inner loop
            f_g = face_gray.astype(np.int32)
            for i in range(1, 63, 2): # Stepped for speed
                for j in range(1, 63, 2):
                    c = f_g[i, j]
                    code = 0
                    if f_g[i-1, j-1] >= c: code |= 128
                    if f_g[i-1, j] >= c:   code |= 64
                    if f_g[i-1, j+1] >= c: code |= 32
                    if f_g[i, j+1] >= c:   code |= 16
                    if f_g[i+1, j+1] >= c: code |= 8
                    if f_g[i+1, j] >= c:   code |= 4
                    if f_g[i+1, j-1] >= c: code |= 2
                    if f_g[i, j-1] >= c:   code |= 1
                    lbp[i, j] = code
            lbp_hist = cv2.calcHist([lbp], [0], None, [32], [1, 256]) # Skip 0
            lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
            
            # Combine all features (Histograms + HOG + LBP)
            features = np.concatenate([
                hist_b, hist_g, hist_r, hist_gray, hist_h, hist_s, hog_features, lbp_hist
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting face features: {e}")
            return None
    
    def recognize_face(self, frame, bbox):
        """
        Recognize face using cosine similarity
        
        Args:
            frame: Input image frame
            bbox: Face bounding box
            
        Returns:
            Worker ID if recognized, None otherwise
        """
        # Extract face features
        face_encoding = self.extract_face_features(frame, bbox)
        
        if face_encoding is None:
            return None
        
        # Prepare current encoding for vector math
        face_encoding = face_encoding.reshape(1, -1)
        
        # Compare with known faces
        best_match_id = None
        best_similarity = 0
        overall_best_sim = 0
        
        for worker_id, encodings in self.face_encodings.items():
            if not encodings: continue
            
            # Conver list to numpy for batch similarity
            all_stored = np.array(encodings)
            
            # Calculate similarity scores using batch dot product (faster for FPS)
            # Since vectors are normalized (due to hist norm + manual), dot product is approx cosine similarity
            # Actually, proper cosine sim is preferred
            try:
                similarities = cosine_similarity(face_encoding, all_stored)[0]
                worker_max_sim = np.max(similarities)
            except:
                worker_max_sim = 0
                
            overall_best_sim = max(overall_best_sim, worker_max_sim)
                
            if worker_max_sim > best_similarity:
                best_similarity = worker_max_sim
                if worker_max_sim >= self.similarity_threshold:
                    best_match_id = worker_id
        
        # DEBUG LOGGING (Reduced to only found or significant)
        if best_match_id:
            print(f"👤 Face recognized: {best_match_id} (Similarity: {best_similarity:.4f})")
        elif overall_best_sim > 0.4:
            print(f"👤 Face unknown (Best: {overall_best_sim:.4f}, Threshold: {self.similarity_threshold})")
            
        if best_match_id:
            return best_match_id
        else:
            return None
    
    def register_worker(self, worker_id, worker_name, face_images_path):
        """
        Register new worker with face images
        
        Args:
            worker_id: Unique worker identifier
            worker_name: Worker name
            face_images_path: Path to folder containing face images
            
        Returns:
            True if registration successful, False otherwise
        """
        if not os.path.exists(face_images_path):
            print(f"❌ Face images path not found: {face_images_path}")
            return False
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(face_images_path).glob(f"*{ext}"))
            image_files.extend(Path(face_images_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No face images found in {face_images_path}")
            return False
        
        # Process each face image
        face_encodings = []
        successful_images = 0
        
        for image_path in image_files:
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                # Detect faces
                faces = self.detect_faces(image)
                
                if len(faces) == 0:
                    print(f"⚠️  No face detected in {image_path.name}")
                    continue
                
                if len(faces) > 1:
                    print(f"⚠️  Multiple faces detected in {image_path.name}, using first face")
                
                # Extract features from first face
                face_encoding = self.extract_face_features(image, faces[0]['bbox'])
                
                if face_encoding is not None:
                    face_encodings.append(face_encoding)
                    successful_images += 1
                    print(f"✅ Processed {image_path.name}")
                else:
                    print(f"❌ Failed to extract features from {image_path.name}")
                    
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
        
        if len(face_encodings) == 0:
            print(f"❌ No valid face encodings extracted for worker {worker_id}")
            return False
        
        # Store face encodings
        self.face_encodings[worker_id] = face_encodings
        self.face_metadata[worker_id] = {
            'name': worker_name,
            'registration_date': datetime.now().isoformat(),
            'num_images': successful_images
        }
        
        # Save database
        self.save_face_database()
        
        print(f"✅ Worker {worker_id} ({worker_name}) registered with {successful_images} face images")
        return True
    
    def save_face_database(self):
        """Save face database to file"""
        database = {
            'encodings': self.face_encodings,
            'metadata': self.face_metadata,
            'similarity_threshold': self.similarity_threshold
        }
        
        with open(self.database_path, 'wb') as f:
            pickle.dump(database, f)
        
        print(f"💾 Face database saved to {self.database_path}")
    
    def load_face_database(self):
        """Load face database from file"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    database = pickle.load(f)
                
                self.face_encodings = database.get('encodings', {})
                self.face_metadata = database.get('metadata', {})
                self.similarity_threshold = database.get('similarity_threshold', 0.45)
                
                print(f"📂 Face database loaded from {self.database_path}")
                print(f"👥 Registered workers: {len(self.face_encodings)}")
                
            except Exception as e:
                print(f"❌ Error loading face database: {e}")
                self.face_encodings = {}
                self.face_metadata = {}
        else:
            print("📂 No existing face database found, starting fresh")
    
    def get_registered_workers(self):
        """Get list of registered workers"""
        workers = []
        for worker_id, metadata in self.face_metadata.items():
            workers.append({
                'worker_id': worker_id,
                'name': metadata['name'],
                'registration_date': metadata['registration_date'],
                'num_images': metadata['num_images']
            })
        return workers
    
    def remove_worker(self, worker_id):
        """Remove worker from database"""
        if worker_id in self.face_encodings:
            del self.face_encodings[worker_id]
        if worker_id in self.face_metadata:
            del self.face_metadata[worker_id]
        
        self.save_face_database()
        print(f"🗑️  Worker {worker_id} removed from database")
        return True
    
    def set_similarity_threshold(self, threshold):
        """Set similarity threshold for face recognition"""
        self.similarity_threshold = max(0.1, min(1.0, threshold))
        self.save_face_database()
        print(f"🎯 Similarity threshold set to {self.similarity_threshold}")
    
    def verify_face(self, frame, bbox, claimed_worker_id):
        """
        Verify if face matches claimed worker ID
        
        Args:
            frame: Input image frame
            bbox: Face bounding box
            claimed_worker_id: Worker ID to verify against
            
        Returns:
            Tuple of (is_verified, similarity_score)
        """
        if claimed_worker_id not in self.face_encodings:
            return False, 0.0
        
        face_encoding = self.extract_face_features(frame, bbox)
        
        if face_encoding is None:
            return False, 0.0
        
        # Calculate similarity with claimed worker's encodings
        max_similarity = 0
        for stored_encoding in self.face_encodings[claimed_worker_id]:
            similarity = cosine_similarity(
                [face_encoding], 
                [stored_encoding]
            )[0][0]
            max_similarity = max(max_similarity, similarity)
        
        is_verified = max_similarity >= self.similarity_threshold
        return is_verified, max_similarity
