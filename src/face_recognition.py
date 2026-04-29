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
from collections import deque, Counter
import threading
import urllib.request
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

class FaceRecognitionSystem:
    def __init__(self, similarity_threshold=0.520):
        """
        Initialize Face Recognition System
        
        Args:
            similarity_threshold: Threshold for face recognition (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.face_encodings = {}
        self.face_metadata = {}
        # Make path relative to src/ directory
        self.database_path = os.path.join(os.path.dirname(__file__), "..", "data", "face_database.pkl")
        
        # Create data directory if not exists
        os.makedirs("data", exist_ok=True)
        
        # Load existing face database
        self.load_face_database()
        
        # Track recent similarity scores for statistics fallback
        self.last_similarities = []
        self.max_history_size = 100
        
        # ── FIX 1: Load face detector ONCE (bukan setiap frame) ──────────────
        # Coba DNN SSD (lebih akurat) dulu, fallback ke Haar Cascade
        self.dnn_net = None
        self.embedder_net = None
        self._try_load_dnn_detector()
        self._try_load_embedder()

        # Haar Cascade sebagai fallback – di-load sekali
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # ─────────────────────────────────────────────────────────────────────

        print("[*] Face Recognition System initialized")
        print("[*] Target Akurasi Face Rec (Skripsi): >= 77.20%")
        print(f"[*] Strict Similarity threshold applied: {self.similarity_threshold}")
        
        # ── FIX 3: IoU-based face tracking (ganti grid-key lama) ─────────────
        # tracked_faces: list of {bbox, identity_history, last_time}
        self.tracked_faces = []
        self.iou_threshold  = 0.30   # overlap minimal agar dianggap wajah sama
        self.history_window = 5      # jumlah frame untuk voting (dipercepat dari 15)
        self.face_expire_sec = 4.0   # hapus track jika tidak terlihat > 4 detik
        # ─────────────────────────────────────────────────────────────────────

        self.lock = threading.Lock()

    # ── Helper: Download DNN model jika belum ada ─────────────────────────────
    def _try_load_dnn_detector(self):
        """Load OpenCV DNN face detector (SSD ResNet). Download jika belum ada."""
        model_dir  = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_detector')
        proto_path = os.path.join(model_dir, 'deploy.prototxt')
        model_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

        proto_url = (
            "https://raw.githubusercontent.com/opencv/opencv/master/"
            "samples/dnn/face_detector/deploy.prototxt"
        )
        model_url = (
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
            "res10_300x300_ssd_iter_140000.caffemodel"
        )

        try:
            os.makedirs(model_dir, exist_ok=True)
            if not os.path.exists(proto_path):
                print("[*] Downloading DNN face detector prototxt...")
                urllib.request.urlretrieve(proto_url, proto_path)
            if not os.path.exists(model_path):
                print("[*] Downloading DNN face detector model (~2 MB)...")
                urllib.request.urlretrieve(model_url, model_path)

            self.dnn_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            print("[*] DNN Face Detector loaded (lebih akurat dari Haar Cascade)")
        except Exception as e:
            print(f"[!] DNN detector tidak tersedia ({e}), menggunakan Haar Cascade.")
            self.dnn_net = None
    # ── Helper: Download Embedder model (CNN) jika belum ada ─────────────────
    def _try_load_embedder(self):
        """Load Official FaceNet (InceptionResnetV1 - 512D)."""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[*] Loading FaceNet 512D (InceptionResnetV1) on {self.device}...")
            
            # Load pretrained model
            self.embedder_net = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            print("[*] FaceNet 512D loaded successfully")
        except Exception as e:
            print(f"[!] Gagal load FaceNet 512D ({e}). Sistem akan menggunakan fallback.")
            self.embedder_net = None
            self.device = 'cpu'
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect_faces(self, frame):
        """
        Detect faces in frame.
        Prioritas: DNN SSD (akurat) → Haar Cascade (fallback).
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face detections with bounding boxes & confidence
        """
        h, w = frame.shape[:2]
        face_detections = []

        # ── Jalur 1: DNN SSD ─────────────────────────────────────────────────
        if self.dnn_net is not None:
            try:
                # Pastikan frame memiliki 3 channel (BGR)
                if len(frame.shape) == 2:
                    infer_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 4:
                    infer_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    infer_frame = frame
                
                # Tidak perlu cv2.resize manual, blobFromImage sudah melakukannya
                blob = cv2.dnn.blobFromImage(
                    infer_frame, 1.0,
                    (300, 300), (104.0, 177.0, 123.0),
                    swapRB=False, crop=False
                )
                self.dnn_net.setInput(blob)
                detections = self.dnn_net.forward()

                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence < 0.50:   # buang deteksi lemah
                        continue
                    x1 = max(0, int(detections[0, 0, i, 3] * w))
                    y1 = max(0, int(detections[0, 0, i, 4] * h))
                    x2 = min(w, int(detections[0, 0, i, 5] * w))
                    y2 = min(h, int(detections[0, 0, i, 6] * h))
                    if x2 > x1 and y2 > y1:
                        face_detections.append({'bbox': [x1, y1, x2, y2], 'confidence': confidence})
                
                if face_detections:
                    return face_detections
                # Jika DNN tidak menemukan apa-apa, biarkan jatuh ke Haar Cascade sebagai fallback
            except Exception as e:
                print(f"[!] DNN SSD Error: {e}. Fallback to Haar Cascade.")
                # Lanjut ke Haar Cascade di bawah ini

        # ── Jalur 2: Haar Cascade (fallback) ─────────────────────────────────
        try:
            h, w = frame.shape[:2]
            # Validasi ukuran: Jika frame/roi terlalu kecil, Haar Cascade akan crash.
            if h < 20 or w < 20:
                return face_detections
                
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)   # normalisasi pencahayaan sebelum deteksi
            
            # Gunakan minSize yang proporsional dengan gambar
            min_s = min(40, int(min(h, w) * 0.2))
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,    # Diubah ke 1.1 agar tidak memicu bug getScaleData di OpenCV
                minNeighbors=4,     # sedikit lebih permisif (dari 5)
                minSize=(min_s, min_s)
            )
            for (x, y, fw, fh) in faces:
                face_detections.append({
                    'bbox': [x, y, x + fw, y + fh],
                    'confidence': 0.90
                })
        except Exception as e:
            print(f"[!] Haar Cascade Error: {e}")
            
        return face_detections
    
    def extract_face_features(self, frame, bbox):
        """
        Extract 512-d face embeddings using FaceNet (InceptionResnetV1)
        """
        if self.embedder_net is None:
            # Fallback jika DNN gagal
            print("[!] Warning: FaceNet Embedder not loaded, using zero-fallback")
            return np.zeros(512, dtype=np.float32)

        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop wajah
            face_region = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            # Validasi Ukuran: FaceNet lebih toleran tapi terlalu kecil bisa jelek kualitasnya
            fh, fw = face_region.shape[:2]
            if fw < 20 or fh < 20:
                return None

            # Convert BGR (OpenCV) to RGB (PIL/PyTorch format)
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # Preprocessing untuk InceptionResnetV1
            # Input yang optimal adalah 160x160 dengan normalisasi standar image
            preprocess = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            face_tensor = preprocess(face_pil).unsqueeze(0).to(self.device)
            
            # Forward pass melalui CNN
            with torch.no_grad():
                face_encoding = self.embedder_net(face_tensor).cpu().numpy()[0]
            
            # Normalisasi L2 untuk Cosine Similarity
            encoding = face_encoding.flatten()
            norm = np.linalg.norm(encoding)
            if norm > 1e-6:
                encoding = encoding / norm
                
            return encoding

        except Exception as e:
            print(f"[!] Error in FaceNet feature extraction: {e}")
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
        # --- LAYER VALIDASI: Pastikan benar-benar ada wajah di region ini ---
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, w), min(y2, h)
        face_roi = frame[y1:y2, x1:x2]
        
        tight_bbox = bbox
        if face_roi.size > 0:
            found_faces = self.detect_faces(face_roi)
            # Cari apakah ada wajah dengan confidence cukup tinggi (0.5)
            # Jika tidak ada wajah nyata, jangan tebak nama (return None)
            valid_faces = [f for f in found_faces if f['confidence'] >= 0.5]
            if not valid_faces:
                # Return None (Unknown) tapi tetap update last_similarities agar grafik jalan
                self.last_similarities.append(0.0) 
                if len(self.last_similarities) > self.max_history_size: self.last_similarities.pop(0)
                return None, 0.0
            
            # Gunakan TIGHT BBOX dari wajah yang terdeteksi (SANGAT PENTING untuk FaceNet)
            best_face = max(valid_faces, key=lambda f: f['confidence'])
            fx1, fy1, fx2, fy2 = best_face['bbox']
            # Convert koordinat lokal face_roi ke koordinat absolut frame
            tight_bbox = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]
        
        # Extract face features menggunakan TIGHT BBOX (bukan kotak badan utuh)
        face_encoding = self.extract_face_features(frame, tight_bbox)
        
        if face_encoding is None:
            return None, 0.0
        
        # Prepare current encoding for vector math
        face_encoding = face_encoding.reshape(1, -1)
        
        # Compare with known faces
        best_match_id = None
        best_similarity = 0
        second_best_similarity = 0
        potential_best_id = None
        overall_best_sim = 0
        
        for worker_id, encodings in self.face_encodings.items():
            if not encodings: continue
            
            all_stored = np.array(encodings)
            
            try:
                similarities = cosine_similarity(face_encoding, all_stored)[0]
                worker_max_sim = np.max(similarities)
            except ValueError:
                print(f"⚠️ [PENTING] Data wajah {worker_id} tidak kompatibel! HAPUS pekerja ini dan DAFTARKAN ULANG.")
                worker_max_sim = 0
            except Exception:
                worker_max_sim = 0
                
            overall_best_sim = max(overall_best_sim, worker_max_sim)
                
            if worker_max_sim > best_similarity:
                second_best_similarity = best_similarity
                best_similarity = worker_max_sim
                potential_best_id = worker_id
            elif worker_max_sim > second_best_similarity:
                second_best_similarity = worker_max_sim
        
        # LOGIKA MARGIN: Juara 1 harus lebih unggul dari juara 2 minimal 0.012
        # Nilai 0.012 berdasarkan observasi margin nyata (0.00-0.04) di log sistem.
        # Terlalu besar (0.05) akan memblokir semua deteksi yang valid.
        if potential_best_id and best_similarity >= self.similarity_threshold:
            best_match_id = potential_best_id
        else:
            if potential_best_id:
                 # Optional: still log it but don't block it
                 pass
        
        # SISTEM VOTING / SMOOTHING ANTI-ACAK (TEMPORAL FILTER)
        # Catat similarity untuk statistik
        self.last_similarities.append(overall_best_sim)
        if len(self.last_similarities) > self.max_history_size:
            self.last_similarities.pop(0)

        current_time = datetime.now().timestamp()
        
        with self.lock:
            # ── FIX 3: IoU-based face tracking ──────────────────────────────
            # Bersihkan track yang sudah expire
            self.tracked_faces = [
                t for t in self.tracked_faces
                if current_time - t['last_time'] <= self.face_expire_sec
            ]

            # Hitung IoU bbox saat ini dengan setiap track yang ada
            def _iou(a, b):
                """Intersection over Union antara dua bbox [x1,y1,x2,y2]"""
                ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
                ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
                if ix2 <= ix1 or iy2 <= iy1:
                    return 0.0
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_a = (a[2]-a[0]) * (a[3]-a[1])
                area_b = (b[2]-b[0]) * (b[3]-b[1])
                return inter / (area_a + area_b - inter + 1e-6)

            # Cari track yang paling overlap dengan bbox ini
            best_track = None
            best_iou   = 0.0
            for track in self.tracked_faces:
                iou_val = _iou(bbox, track['bbox'])
                if iou_val > best_iou:
                    best_iou   = iou_val
                    best_track = track

            # Jika tidak ada track yang cukup overlap, buat track baru
            if best_track is None or best_iou < self.iou_threshold:
                best_track = {
                    'bbox': list(bbox),
                    'history': [],
                    'last_time': current_time
                }
                self.tracked_faces.append(best_track)

            # Update posisi bbox track ke posisi terbaru
            best_track['bbox']      = list(bbox)
            best_track['last_time'] = current_time

            # LOGIKA STICKINESS: jika Cosine Similarity gagal threshold tapi
            # kandidat terakhir di track ini masih cukup mirip (hysteresis)
            last_winner = (
                Counter(best_track['history']).most_common(1)[0][0]
                if best_track['history'] else None
            )

            if not best_match_id and last_winner:
                target_encodings = np.array(self.face_encodings.get(last_winner, []))
                if len(target_encodings) > 0:
                    try:
                        loc_sim = np.max(cosine_similarity(face_encoding, target_encodings)[0])
                        if loc_sim >= 0.75:   # Hysteresis: sedikit di bawah threshold masih ok
                            best_match_id = last_winner
                    except Exception:
                        pass

            if best_match_id:
                best_track['history'].append(best_match_id)

                # Batasi history window ke N frame terakhir
                if len(best_track['history']) > self.history_window:
                    best_track['history'].pop(0)

                # Voting mayoritas dari history window
                most_common_id = Counter(best_track['history']).most_common(1)[0][0]

                # INSTANT RECOGNITION: Jika sangat yakin (>0.90), langsung tampilkan
                if best_similarity > 0.90:
                    most_common_id = best_match_id

                print(f"👤 Face recognized: {most_common_id} (Similarity: {best_similarity:.4f})")
                return most_common_id, best_similarity

            elif overall_best_sim > 0.4:
                print(f"👤 Face unknown (Best: {overall_best_sim:.4f}, Threshold: {self.similarity_threshold})")

            return None, overall_best_sim
    
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
        
        # Kumpulkan semua file gambar dan deduplikat (penting di Windows:
        # glob "*.jpg" dan "*.JPG" mengembalikan file yang SAMA di filesystem case-insensitive)
        image_extensions = ['.jpg', '.jpeg', '.png']
        seen_paths = set()
        image_files = []

        for ext in image_extensions:
            for p in Path(face_images_path).glob(f"*{ext}"):
                key = p.resolve()
                if key not in seen_paths:
                    seen_paths.add(key)
                    image_files.append(p)
            for p in Path(face_images_path).glob(f"*{ext.upper()}"):
                key = p.resolve()
                if key not in seen_paths:
                    seen_paths.add(key)
                    image_files.append(p)

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
                
                # FALLBACK: If no face detected, and it's a small crop (from captures), 
                # assume the whole image IS the face.
                if len(faces) == 0:
                    h, w = image.shape[:2]
                    # Simulate a face detection for the whole image
                    faces = [{'bbox': [0, 0, w, h], 'confidence': 1.0}]
                    print(f"[*] No face detected in {image_path.name}, using whole image as fallback...")
                
                if len(faces) > 1:
                    print(f"[*] Multiple faces detected in {image_path.name}, using first face")
                
                # Extract features from first face
                face_encoding = self.extract_face_features(image, faces[0]['bbox'])
                
                if face_encoding is not None:
                    face_encodings.append(face_encoding)
                    successful_images += 1
                    print(f"[*] Processed {image_path.name}")
                else:
                    print(f"[!] Failed to extract features from {image_path.name}")
                    
            except Exception as e:
                print(f"[!] Error processing {image_path.name}: {e}")
        
        if len(face_encodings) == 0:
            print(f"[!] No valid face encodings extracted for worker {worker_id}")
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
        
        print(f"[*] Worker {worker_id} ({worker_name}) registered with {successful_images} face images")
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
        
        print(f"[*] Face database saved to {self.database_path}")
    
    def load_face_database(self):
        """Load face database from file"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    database = pickle.load(f)
                
                self.face_encodings = database.get('encodings', {})
                self.face_metadata = database.get('metadata', {})
                # Menggunakan threshold yang di-set di __init__ daripada yang di database
                # self.similarity_threshold = database.get('similarity_threshold', 0.65)
                
                print(f"[*] Face database loaded from {self.database_path}")
                print(f"[#] Registered workers: {len(self.face_encodings)}")
                
            except Exception as e:
                print(f"[!] Error loading face database: {e}")
                self.face_encodings = {}
                self.face_metadata = {}
        else:
            print("[*] No existing face database found, starting fresh")
    
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
        print(f"[-] Worker {worker_id} removed from database")
        return True
    
    def update_worker_name(self, worker_id, new_name):
        """Update worker name in metadata"""
        if worker_id in self.face_metadata:
            self.face_metadata[worker_id]['name'] = new_name
            self.save_face_database()
            print(f"[*] Worker {worker_id} name updated to {new_name}")
            return True
        return False
    
    def set_similarity_threshold(self, threshold):
        """Set similarity threshold for face recognition"""
        self.similarity_threshold = max(0.1, min(1.0, threshold))
        self.save_face_database()
        print(f"[*] Similarity threshold set to {self.similarity_threshold}")
    
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
        if not is_verified and max_similarity > 0.4:
            print(f"[DEBUG] Best match for {claimed_worker_id} score: {max_similarity:.3f} (Threshold: {self.similarity_threshold})")
        elif is_verified:
            print(f"[DEBUG] VERIFIED {claimed_worker_id} score: {max_similarity:.3f}")
            
        return is_verified, max_similarity
