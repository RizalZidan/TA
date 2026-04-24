"""
Simple Violations Detector
Using helmet.v2i.yolov8 dataset directly
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
try:
    from scaling_config import scaling_config
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
    def __init__(self, confidence_threshold=0.40, model_path=None):
        """
        Initialize Violations Detector with simple setup
        
        Args:
            confidence_threshold: Confidence threshold for detection
        """
        self.confidence_threshold = confidence_threshold
        
        # Determine target path
        helmet_v2_path = os.path.join(os.path.dirname(__file__), "..", "experiments/helmet.v2i.yolov8/helmet_vest_detection/yolov8n_50epochs_augmented/weights/best.pt")
        new_retrained_model = os.path.join(os.path.dirname(__file__), "..", "experiments/PPE_Combined_Dataset/runs/ppe_retraining_30_epochs/weights/best.pt")
        premier_model_path = os.path.join(os.path.dirname(__file__), "..", "models/yolov8_50_epoch/best.pt")
        
        # Priority: 1. Helmet v2 (User requested), 2. Retrained, 3. Premier
        if os.path.exists(helmet_v2_path):
            target_path = helmet_v2_path
        elif os.path.exists(new_retrained_model):
            target_path = new_retrained_model
        else:
            target_path = premier_model_path
        
        # Override with manual path if provided
        if model_path:
            target_path = model_path
            
        if target_path and os.path.exists(target_path):
            print(f"🎯 Loading Model: {os.path.basename(os.path.dirname(os.path.dirname(target_path)))}")
            self.model = YOLO(target_path)
            
            # ── MODE DETECTOR ──────────────────────────────────────────────
            if "helmet.v2i" in target_path.lower():
                self.class_names = {0: 'helmet', 1: 'vest'}
                self.use_apd_model = False
                self.model_label = "Helmet v2 (2-Class)"
            elif "ppe_retraining" in target_path.lower():
                self.class_names = {0: 'helmet', 1: 'no helmet', 2: 'person', 3: 'vest', 4: 'no vest'}
                self.use_apd_model = True
                self.model_label = "Retrained v2 (30 Ep)"
            else:
                self.class_names = {0: 'helmet', 1: 'no helmet', 2: 'person', 3: 'vest', 4: 'no vest'}
                self.use_apd_model = True
                self.model_label = "Combined Model"
                
            print(f"📊 Running in {self.model_label} mode")
        else:
            raise FileNotFoundError(f"❌ No valid model found at {target_path}")

        # Load person model for 2-class presence checking
        if not self.use_apd_model:
            print("👤 Loading Person Detector for logical check...")
            self.person_model = YOLO('yolov8n.pt')
        else:
            self.person_model = None
        
        # Optimize for performance
        self.model.fuse()  # Fuse Conv and BatchNorm for faster inference
        
        # Initialize Face Recognition
        self.recognition_cooldowns = {} # (location) -> last_time
        # Load person model for tracking and logical checks
        self.person_model = None
        person_model_path = os.path.join(os.path.dirname(__file__), "..", "models/yolov8n.pt")
        if os.path.exists(person_model_path):
            self.person_model = YOLO(person_model_path)
            print("👤 Person Detector loaded for tracking support")
        else:
            self.person_model = YOLO("yolov8n.pt")
            print("👤 Loading Default YOLOv8n for Person Detection...")
            
        self.unknown_face_memory = {} # {temp_id: {'encoding': encoding, 'last_seen': timestamp}}
        self.identity_cache = {}    # {(loc_key): {'id': name, 'sim': sim, 'time': timestamp}}
        self.identity_votes = {}    # {(loc_key): [list of recent IDs]}
        self.captures_path = "data/captures"
        os.makedirs(self.captures_path, exist_ok=True)
        
        try:
            from src.face_recognition import FaceRecognitionSystem
            self.face_recognizer = FaceRecognitionSystem()
            self.use_face_recognition = True
            print(f"👤 Face Recognition System Loaded")
        except Exception as e:
            print(f"⚠️ Failed to initialize Face Recognition: {e}")
            self.face_recognizer = None
            self.use_face_recognition = False
        
        print("✅ Optimized Violations Detector initialized")
        print(f"📊 Classes: {list(self.class_names.values())}")
        print("🎯 Target Akurasi APD (Skripsi): >= 79.60%")
        print(f"🎯 Confidence threshold: {float(self.confidence_threshold):.3f}")
        print("⚡ Performance optimizations enabled")
    
    def detect_violations(self, frame):
        """
        Detect APD violations only (No_Helmet, No_Vest) - Optimized with smart scaling
        
        Args:
            frame: Input image frame
            
        Returns:
            List of violation detections only
        """
        # --- PERFORMANCE FIX: Resize frame for AI to 640x480 once ---
        h, w = frame.shape[:2]
        if w > 640:
            frame_infer = cv2.resize(frame, (640, 480))
        else:
            frame_infer = frame
            
        # Enhance lighting if image is too dark or washed out (to help YOLO in bad lighting)
        try:
            gray = cv2.cvtColor(frame_infer, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # If the frame is too dark (< 90) or too bright/washed out, apply CLAHE
            if mean_brightness < 90 or mean_brightness > 180:
                lab = cv2.cvtColor(frame_infer, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl,a,b))
                frame_infer = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except:
            pass
            
        # --- RULE 6: Confidence threshold 0.30 (Lowered for better responsiveness) ---
        CONF_THRESHOLD = 0.30
        
        # 1. Get raw detections
        raw_results = self.model(frame_infer, verbose=False)
        
        # 2. Extract PPE items and Persons
        persons = []
        apd_items = []
        
        # Process primary model detections
        for r in raw_results:
            for b in r.boxes:
                conf = float(b.conf[0].cpu().numpy())
                
                cid = int(b.cls[0].cpu().numpy())
                cname = self.class_names.get(cid, 'unknown').lower().replace('_', ' ')
                bbox = b.xyxy[0].cpu().numpy().tolist()
                
                # if conf > 0.10: # Silenced for performance
                #    print(f"🔍 AI RAW: [{cname}] conf: {conf:.3f} | class_id: {cid}")

                if conf < CONF_THRESHOLD: continue
                
                if cname == 'person':
                    persons.append({'bbox': bbox, 'conf': conf})
                else:
                    apd_items.append({'class': cname, 'bbox': bbox, 'conf': conf})
                    
        # If model doesn't include 'person', use dedicated person_model
        if not persons and self.person_model:
            p_results = self.person_model(frame_infer, conf=0.30, classes=[0], verbose=False)
            for pr in p_results:
                for b in pr.boxes:
                    persons.append({'bbox': b.xyxy[0].cpu().numpy().tolist(), 'conf': float(b.conf[0].cpu().numpy())})
        
        violations = []
        
        # --- RULE 1: Group detections by person ---
        for p in persons:
            pb = p['bbox']
            
            # Temporary state for this person
            has_helmet = False
            has_no_helmet_box = None
            has_vest = False
            has_no_vest_box = None
            
            # Find all APD items associated with this person
            for item in apd_items:
                if self._is_inside(item['bbox'], pb):
                    # --- RULE 5: Priority Rule ---
                    if item['class'] == 'helmet':
                        has_helmet = True
                    elif item['class'] == 'no helmet':
                        if has_no_helmet_box is None or item['conf'] > has_no_helmet_box['conf']:
                            has_no_helmet_box = item
                    elif item['class'] == 'vest':
                        has_vest = True
                    elif item['class'] == 'no_vest' or item['class'] == 'no vest':
                        if has_no_vest_box is None or item['conf'] > has_no_vest_box['conf']:
                            has_no_vest_box = item
            
            # --- RULE 2 & 3: PPE Evaluation & Violation Criteria ---
            # Rule: If "helmet" exists -> ignore "no helmet"
            if has_helmet:
                violations.append(self._create_violation_entry(frame, pb, 'helmet_ok', 1.0, is_proxy=True))
            else:
                # If using 4-class model, we have explicit 'no helmet' detection
                if has_no_helmet_box:
                    violations.append(self._create_violation_entry(frame, pb, 'nohelmet', has_no_helmet_box['conf'], is_proxy=True))
                # If using 2-class model, absence of 'helmet' IS the violation
                elif not self.use_apd_model:
                    violations.append(self._create_violation_entry(frame, pb, 'nohelmet', p['conf'], is_proxy=True))

            # Rule: If "vest" exists -> ignore "no vest"
            if has_vest:
                violations.append(self._create_violation_entry(frame, pb, 'vest_ok', 1.0, is_proxy=True))
            else:
                if has_no_vest_box:
                    violations.append(self._create_violation_entry(frame, pb, 'novest', has_no_vest_box['conf'], is_proxy=True))
                elif not self.use_apd_model:
                    violations.append(self._create_violation_entry(frame, pb, 'novest', p['conf'], is_proxy=True))
            
            # --- RULE 4: Compliant workers -> DO NOT return anything ---
            # (Handled: we only append to violations list if PPE is missing)
            
            # --- RULE 7: FACE RECOGNITION INTEGRATION ---
            # If ANY violation exists for this person, their 'person' bbox is already part of the violation entry.
            # If no violations, we don't return anything, but tracking in app_advanced still runs.

        return violations

    def _is_inside(self, apd_bbox, person_bbox, type='head'):
        """Check if an APD bbox is reasonably within a person area"""
        ax1, ay1, ax2, ay2 = apd_bbox
        px1, py1, px2, py2 = person_bbox
        
        # Center of APD
        acx = (ax1 + ax2) / 2
        acy = (ay1 + ay2) / 2
        
        # Broadened horizontal check: Allow APD to be within 60% margin outside person width 
        # (EXTREMELY flexible for perspective/distortion)
        w = px2 - px1
        margin_x = w * 0.60
        if acx < (px1 - margin_x) or acx > (px2 + margin_x):
            return False
        
        # For helmet, it should be in upper part of person
        if type == 'head':
            person_height = py2 - py1
            # UNLIMITED Top Margin (up to 40% of height) to catch high-mounted cameras
            if not (py1 - person_height * 0.40 <= acy <= py1 + person_height * 0.80): return False
            
        return (py1 - (py2-py1)*0.2 <= acy <= py2) if type != 'head' else True

    def _create_violation_entry(self, frame, bbox, v_type, conf, is_proxy=False):
        """Helper to create a standard violation entry with face recognition"""
        # If it's a proxy from a person detection, we create a specialized bbox for better UI
        if is_proxy:
            px1, py1, px2, py2 = bbox
            p_w = px2 - px1
            p_h = py2 - py1
            
            if v_type in ['nohelmet', 'helmet_ok']:
                # Head Area Scaling
                scaled_bbox = [
                    int(px1), 
                    int(py1 - p_h * 0.08),
                    int(px2), 
                    int(py1 + p_h * 0.22)
                ]
            elif v_type in ['novest', 'vest_ok']:
                # Torso Area Scaling
                scaled_bbox = [
                    int(px1), 
                    int(py1 + p_h * 0.25),
                    int(px2), 
                    int(py1 + p_h * 0.75)
                ]
            else:
                scaled_bbox = bbox
        else:
            # Direct detection: Apply smart scaling
            scaled_bbox = self._apply_smart_scaling(bbox, 'No_Helmet' if v_type == 'nohelmet' else 'No_Vest')
        
        worker_id = "Unknown"
        face_sim = 0.0
        
        # Face recognition logic
        current_time = datetime.now().timestamp()
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        # Tighter grid for identity cooldown (5px instead of 20px) to distinguish close persons
        loc_key = f"cam_{int(cx/5)}_{int(cy/5)}"
        
        # Identity smoothing logic: Check if we have a recent valid identification for this location
        cached = self.identity_cache.get(loc_key)
        if cached and (current_time - cached['time'] < 4.0):
            # Use cached identity to avoid flickering
            worker_id = cached['id']
            face_sim = cached['sim']
        
        # If no cache or cache expired (or we want to re-verify every 1.5s for faster response)
        if self.use_face_recognition and (current_time - self.recognition_cooldowns.get(loc_key, 0) > 1.5):
            self.recognition_cooldowns[loc_key] = current_time
            
            # Specialized Face Crop Area: Top 40% of person, centered
            if is_proxy:
                px1, py1, px2, py2 = bbox
                h = py2 - py1
                w = px2 - px1
                # Focus on a square-ish head area for better FaceNet performance
                recon_bbox = [
                    int(px1 + w * 0.15), 
                    int(py1 - h * 0.02), 
                    int(px2 - w * 0.15), 
                    int(py1 + h * 0.35)
                ]
            else:
                recon_bbox = scaled_bbox
                
            recognized, face_sim = self.face_recognizer.recognize_face(frame, recon_bbox)
            
            # --- VOTING SYSTEM ---
            if loc_key not in self.identity_votes: self.identity_votes[loc_key] = []
            current_vote = recognized if recognized else "Unknown"
            self.identity_votes[loc_key].append(current_vote)
            if len(self.identity_votes[loc_key]) > 8: self.identity_votes[loc_key].pop(0)
            
            # Count majority vote
            from collections import Counter
            vote_counts = Counter(self.identity_votes[loc_key])
            winner, count = vote_counts.most_common(1)[0]
            
            # Only switch ID if it has significant majority (at least 3 consistent votes)
            final_id = winner if (count >= 3 and winner != "Unknown") else (recognized if recognized else "Unknown")
            
            if final_id != "Unknown":
                metadata = self.face_recognizer.face_metadata.get(final_id, {})
                worker_id = metadata.get('name', final_id)
                face_sim = float(face_sim)
                
                # Update Cache for stability
                self.identity_cache[loc_key] = {
                    'id': worker_id,
                    'sim': face_sim,
                    'time': current_time
                }
            else:
                # --- LOGIKA CAPTURE MUKA UNTUK UNKNOWN ---
                try:
                    # Ambil encoding muka ini
                    current_encoding = self.face_recognizer.extract_face_features(frame, recon_bbox)
                    if current_encoding is not None:
                        # Cari di memory unknown yang mirip
                        matched_temp_id = None
                        from sklearn.metrics.pairwise import cosine_similarity
                        
                        now = datetime.now().timestamp()
                        for tid, data in list(self.unknown_face_memory.items()):
                            if now - data['last_seen'] > 600: # 10 menit
                                del self.unknown_face_memory[tid]
                                continue
                            
                            sim_unknown = cosine_similarity([current_encoding], [data['encoding']])[0][0]
                            if sim_unknown > 0.85: # Threshold ketat untuk unknown grouping
                                matched_temp_id = tid
                                break
                        
                        if not matched_temp_id:
                            # Buat temp ID baru
                            import secrets
                            matched_temp_id = f"Unknown_{secrets.token_hex(4)}"
                            self.unknown_face_memory[matched_temp_id] = {
                                'encoding': current_encoding,
                                'last_seen': now
                            }
                        else:
                            self.unknown_face_memory[matched_temp_id]['last_seen'] = now
                        
                        worker_id = matched_temp_id # Gunakan Temp ID sebagai worker_id
                        
                        # Save image to capture folder
                        temp_id_dir = os.path.join(self.captures_path, matched_temp_id)
                        os.makedirs(temp_id_dir, exist_ok=True)
                        
                        # Count existing images to avoid too many
                        if len(os.listdir(temp_id_dir)) < 10:
                            img_filename = f"face_{datetime.now().strftime('%H%M%S_%f')}.jpg"
                            rx1, ry1, rx2, ry2 = map(int, recon_bbox)
                            fh, fw = frame.shape[:2]
                            face_crop = frame[max(0, ry1):min(fh, ry2), max(0, rx1):min(fw, rx2)]
                            if face_crop.size > 0:
                                cv2.imwrite(os.path.join(temp_id_dir, img_filename), face_crop)
                except Exception as e:
                    print(f"⚠️ Error in unknown capture logic: {e}")

        is_violation = v_type in ['nohelmet', 'novest']
        return {
            'bbox': list(map(int, scaled_bbox)), 
            'class': v_type,
            'confidence': float(conf),
            'worker_id': worker_id,
            'is_violation': is_violation, # Flag untuk Dashboard
            'violation_severity': 'high' if is_violation else 'none',
            'violation_info': {
                'is_violation': is_violation,
                'violation_type': v_type,
                'face_similarity': face_sim
            }
        }
    
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
            
            # --- RULE 4: Filter out SAFE statuses from Dashboard visualization ---
            # But we still keep them in the detection list for backend stabilizer logic
            if not detection.get('is_violation', True):
                continue

            # Color based on violation type
            worker_id = detection.get('worker_id', 'Unknown')
            if class_name == 'nohelmet':
                color = (0, 0, 255)  # Red for no helmet violation
                label = f"No Helmet {confidence:.2f} [{worker_id}]"
            elif class_name == 'novest':
                color = (0, 165, 255)  # Orange for no vest violation
                label = f"No Vest {confidence:.2f} [{worker_id}]"
            elif class_name == 'helmet_ok':
                color = (0, 255, 0)  # Green for SAFE
                label = f"Helmet OK [{worker_id}]"
            elif class_name == 'vest_ok':
                color = (0, 255, 0)  # Green for SAFE
                label = f"Vest OK [{worker_id}]"
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
