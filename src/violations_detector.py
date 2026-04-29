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
        ppe_combined_path = os.path.join(os.path.dirname(__file__), "..", "experiments/apd_finetuned/best.pt")
        helmet_v2_path = os.path.join(os.path.dirname(__file__), "..", "experiments/helmet_v2_training_50ep/weights/best.pt")
        premier_model_path = os.path.join(os.path.dirname(__file__), "..", "models/yolov8_50_epoch/best.pt")
        
        # Priority: 1. Helmet v2 (Requested), 2. PPE Combined, 3. Premier
        if os.path.exists(helmet_v2_path):
            target_path = helmet_v2_path
        elif os.path.exists(ppe_combined_path):
            target_path = ppe_combined_path
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
            elif "combined" in target_path.lower() or "ppe_retraining" in target_path.lower() or "finetuned" in target_path.lower():
                self.class_names = {0: 'helmet', 1: 'no helmet', 2: 'vest', 3: 'no vest'}
                self.use_apd_model = True
                self.model_label = "PPE Combined (4-Class)"
            else:
                self.class_names = {0: 'helmet', 1: 'no helmet', 2: 'person', 3: 'vest', 4: 'no vest'}
                self.use_apd_model = True
                self.model_label = "Default Model (5-Class)"
                
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
        
        # --- Temporal Detection Cache ---
        # Short TTL (5 frames) to prevent false positives from persisting too long
        self._det_cache = []      # cached violation entries
        self._det_age   = 0       # frames since last real detection
        self.CACHE_TTL  = 5       # ~0.15s at 30fps - short enough to avoid locking false positives

        
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
    
    def set_roi(self, polygon_ratios):
        """
        Set Region of Interest polygon.
        polygon_ratios: list of (x_ratio, y_ratio) from 0.0 to 1.0
        e.g. [(0.4, 0.0), (1.0, 0.0), (1.0, 1.0), (0.4, 1.0)] = right 60% of frame
        Set to None to disable ROI masking.
        """
        self.roi_polygon = polygon_ratios
        if polygon_ratios:
            print(f"🗺️ ROI set with {len(polygon_ratios)} points")
        else:
            print("🗺️ ROI cleared (full frame)")

    def _point_in_roi(self, cx_ratio, cy_ratio):
        """Check if a point (as ratio 0-1) is inside the ROI polygon using ray casting."""
        if not self.roi_polygon:
            return True  # No ROI = accept everything
        import math
        poly = self.roi_polygon
        n  = len(poly)
        inside = False
        px, py = cx_ratio, cy_ratio
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def detect_violations(self, frame, enable_face_anchor=False):
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
            
        # --- Dual threshold: lower for positive APD, higher for violation detection ---
        APD_CONF       = 0.38  # For helm/vest positive detection (raised to reduce false positives)
        VIOLATION_CONF = 0.45  # For no_helmet/no_vest (higher = less false alarm)
        
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

                # Apply dual threshold: violations need higher confidence
                if cname == 'person':
                    # Person needs extremely high confidence (0.88) to avoid ghost detections
                    if conf < 0.88:
                        continue
                    persons.append({'bbox': bbox, 'conf': conf})
                else:
                    is_violation_class = 'no' in cname  # 'no helmet' or 'no vest'
                    threshold = VIOLATION_CONF if is_violation_class else APD_CONF
                    if conf < threshold:
                        continue
                    apd_items.append({'class': cname, 'bbox': bbox, 'conf': conf})
                    
        # If model doesn't include 'person', use dedicated person_model
        # ALWAYS run person_model to catch ALL people
        if self.person_model:
            # Threshold ditingkatkan ke 0.90: ekstrim tinggi untuk menolak halusinasi benda mati (Prioritas 3)
            p_results = self.person_model(frame_infer, conf=0.90, classes=[0], verbose=False)
            for pr in p_results:
                for b in pr.boxes:
                    new_bbox = b.xyxy[0].cpu().numpy().tolist()
                    new_conf = float(b.conf[0].cpu().numpy())
                    
                    bw = new_bbox[2] - new_bbox[0]
                    bh = new_bbox[3] - new_bbox[1]
                    
                    # Filter 1: Too small
                    if bw < 25 or bh < 50:
                        continue
                    
                    # Filter 2: Minimum area (3000 px²) — rejects very thin/small detections
                    if bw * bh < 3000:
                        continue
                    
                    # Filter 3: Aspect ratio — real people are significantly taller than wide
                    # h/w < 1.35 akan memblokir struktur lebar horizontal
                    # h/w > 4.0 akan memblokir struktur kurus vertikal (seperti tiang/jeruji turnstile)
                    if bw > 0:
                        ratio = bh / bw
                        if ratio < 1.35 or ratio > 4.0:
                            continue
                    
                    # Deduplicate via IoU
                    is_duplicate = False
                    for existing in persons:
                        if self._iou(new_bbox, existing['bbox']) > 0.4:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        persons.append({'bbox': new_bbox, 'conf': new_conf})
        
        # --- RESCUE WEBCAM PERSON: Gunakan Face Detector sebagai Anchor Person ---
        # Karena YOLO person_model sering ngawur (menangkap bayangan/sisi wajah) jika wajah terlalu dekat ke webcam
        if self.use_face_recognition and enable_face_anchor:
            try:
                found_faces = self.face_recognizer.detect_faces(frame_infer)
                for f in found_faces:
                    fx1, fy1, fx2, fy2 = f['bbox']
                    fw = fx2 - fx1
                    fh = fy2 - fy1
                    # Validasi ukuran minimum absolut
                    if fw < 25 or fh < 25: continue
                    
                    # Buat virtual person bbox berdasarkan proporsi manusia sungguhan
                    virtual_pb = [
                        max(0, fx1 - fw * 0.9),
                        max(0, fy1 - fh * 0.6),
                        min(frame_infer.shape[1], fx2 + fw * 0.9),
                        min(frame_infer.shape[0], fy2 + fh * 3.5)
                    ]
                    
                    is_dup = False
                    for existing in persons:
                        # Jika YOLO menangkap objek di lokasi wajah ini, timpa dengan bentuk virtual yang rapi
                        if self._iou(virtual_pb, existing['bbox']) > 0.05:
                            existing['bbox'] = virtual_pb
                            existing['conf'] = max(existing['conf'], f['confidence'])
                            is_dup = True
                            break
                    
                    if not is_dup:
                        # CCTV FIX: Mencegah detektor wajah berhalusinasi melihat muka di pagar/turnstile CCTV.
                        # Face Anchor HANYA boleh aktif untuk CLOSE-UP Webcam (wajah sangat besar).
                        # Jika wajah di bawah 150 pixel, biarkan YOLO yang mendeteksi badannya!
                        if fw > 150 and f['confidence'] > 0.80:
                            persons.append({'bbox': virtual_pb, 'conf': f['confidence']})
            except Exception as e:
                print(f"[DEBUG] Error di Face Anchor: {e}")
        
        violations = []
        # Track which APD items got matched to a person
        for item in apd_items:
            item['grouped'] = False
        
        # --- RULE 1: Group detections by person ---
        for p in persons:
            pb = p['bbox']
            
            # Temporary state for this person
            has_helmet = False
            has_no_helmet_box = None
            has_vest = False
            has_no_vest_box = None
            matched_any_apd = False
            
            # Find all APD items associated with this person
            for item in apd_items:
                itype = 'head' if 'helmet' in item['class'] else 'torso'
                if self._is_inside(item['bbox'], pb, type=itype):
                    item['grouped'] = True
                    matched_any_apd = True
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
            # Only report violations when there is EXPLICIT APD model evidence.
            if has_helmet:
                violations.append(self._create_violation_entry(frame, pb, 'helmet_ok', 1.0, is_proxy=True))
            elif has_no_helmet_box:
                violations.append(self._create_violation_entry(frame, pb, 'nohelmet', has_no_helmet_box['conf'], is_proxy=True))
            elif not self.use_apd_model:
                # 2-class model only: Guard Langkah 3 (Jangan vonis jika tidak yakin itu orang!)
                if matched_any_apd or p['conf'] > 0.90:
                    violations.append(self._create_violation_entry(frame, pb, 'nohelmet', p['conf'], is_proxy=True))
            elif not matched_any_apd and p['conf'] > 0.90:
                # Person detected with very high confidence but ZERO APD items found
                # Only trigger when extremely sure it's a real person (conf > 0.90)
                violations.append(self._create_violation_entry(frame, pb, 'nohelmet', p['conf'] * 0.65, is_proxy=True))
                violations.append(self._create_violation_entry(frame, pb, 'novest', p['conf'] * 0.65, is_proxy=True))

            if has_vest:
                violations.append(self._create_violation_entry(frame, pb, 'vest_ok', 1.0, is_proxy=True))
            elif has_no_vest_box:
                violations.append(self._create_violation_entry(frame, pb, 'novest', has_no_vest_box['conf'], is_proxy=True))
            elif not self.use_apd_model:
                # Guard Langkah 3
                if matched_any_apd or p['conf'] > 0.90:
                    violations.append(self._create_violation_entry(frame, pb, 'novest', p['conf'], is_proxy=True))

        # --- RESCUE LOGIC: Handle APD items with no matched person ---
        # CRITICAL: Only rescue POSITIVE classes (helmet/vest detected).
        # NEVER rescue negative classes (no helmet/no vest) without a real person anchor -
        # this is the main cause of false positives on turnstiles/barriers/background.
        for item in apd_items:
            if item.get('grouped', False):
                continue  # Already handled by a real person
            
            item_class = item['class']
            
            # Skip violation classes (no helmet / no vest) without a confirmed person anchor
            # These generate too many false positives from background objects
            if 'no' in item_class:
                continue
            
            # Only rescue positive detections (helmet, vest) with high confidence
            if item['conf'] < 0.65:
                continue
            
            ix1, iy1, ix2, iy2 = item['bbox']
            iw, ih = ix2 - ix1, iy2 - iy1
            if iw < 20 or ih < 20:
                continue
            
            virtual_pb = [
                ix1 - iw * 0.2, 
                iy1 - (ih * 1.5 if 'helmet' in item_class else ih * 0.5), 
                ix2 + iw * 0.2, 
                iy2 + (ih * 1.5 if 'vest' in item_class else ih * 0.5)
            ]
            v_type = item_class.replace(' ', '') + '_ok'
            violations.append(self._create_violation_entry(frame, virtual_pb, v_type, item['conf'], is_proxy=True))

        # --- Temporal Cache: prevent flickering ---
        if violations:
            # Got real detections - update cache and reset age
            self._det_cache = violations
            self._det_age = 0
        elif self._det_age < self.CACHE_TTL:
            # No detections this frame, but cache is still fresh - use it
            self._det_age += 1
            return self._det_cache
        else:
            # Cache expired - truly no detections
            self._det_cache = []
            self._det_age = 0

        return violations

    def _iou(self, box1, box2):
        """Calculate Intersection over Union between two bboxes [x1,y1,x2,y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / (area1 + area2 - inter)

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
        if is_proxy:
            px1, py1, px2, py2 = bbox
            p_w = px2 - px1
            p_h = py2 - py1
            
            if v_type in ['nohelmet', 'helmet_ok']:
                # Head Area Scaling (Kotak tepat di area kepala)
                scaled_bbox = [
                    int(px1 + p_w * 0.15), 
                    int(py1 - p_h * 0.05),
                    int(px2 - p_w * 0.15), 
                    int(py1 + p_h * 0.25)
                ]
            elif v_type in ['novest', 'vest_ok']:
                # Torso Area Scaling (Kotak tepat di area bahu ke perut)
                scaled_bbox = [
                    int(px1 + p_w * 0.10), 
                    int(py1 + p_h * 0.25),
                    int(px2 - p_w * 0.10), 
                    int(py1 + p_h * 0.70)
                ]
            else:
                scaled_bbox = bbox
        else:
            # Direct detection: Apply smart scaling
            scaled_bbox = self._apply_smart_scaling(bbox, 'No_Helmet' if v_type == 'nohelmet' else 'No_Vest')
        
        worker_id = "Unknown"
        face_sim = 0.0
        
        # --- PERBAIKAN LOGIKA TRACKING IDENTITAS ---
        current_time = datetime.now().timestamp()
        
        # Cari di cache menggunakan Intersection over Union (IoU) dari kotak person, bukan grid kaku.
        matched_cache_key = None
        for k, v in list(self.identity_cache.items()):
            # Hapus cache lama (> 4 detik)
            if current_time - v['time'] > 4.0:
                del self.identity_cache[k]
                continue
            
            # Cek overlap
            if 'person_bbox' in v and self._iou(bbox, v['person_bbox']) > 0.4:
                matched_cache_key = k
                break
                
        if matched_cache_key:
            worker_id = self.identity_cache[matched_cache_key]['id']
            face_sim = self.identity_cache[matched_cache_key]['sim']
            # Update person bbox as they move
            self.identity_cache[matched_cache_key]['person_bbox'] = bbox
        
        # Panggil Face Recognition tiap 1.5 detik per orang untuk update
        if self.use_face_recognition and (not matched_cache_key or current_time - self.identity_cache[matched_cache_key].get('last_scan', 0) > 1.5):
            
            # Kotak pencarian muka: Selalu gunakan area kepala dari bbox person
            if is_proxy:
                px1, py1, px2, py2 = bbox
                h = py2 - py1; w = px2 - px1
                recon_bbox = [
                    int(px1), 
                    int(max(0, py1 - h * 0.05)), 
                    int(px2), 
                    int(py1 + h * 0.35)
                ]
            else:
                recon_bbox = scaled_bbox
                
            # FaceRecognitionSystem kita sekarang sudah stabil (sudah punya tracking internal)
            recognized, new_sim = self.face_recognizer.recognize_face(frame, recon_bbox)
            
            if recognized:
                metadata = self.face_recognizer.face_metadata.get(recognized, {})
                worker_id = metadata.get('name', recognized)
                face_sim = float(new_sim)
                
                # Simpan/update ke cache
                if not matched_cache_key:
                    import uuid
                    matched_cache_key = str(uuid.uuid4())
                    
                self.identity_cache[matched_cache_key] = {
                    'id': worker_id,
                    'sim': face_sim,
                    'time': current_time,
                    'last_scan': current_time,
                    'person_bbox': bbox
                }
            elif matched_cache_key:
                # Update waktu scan supaya tidak terus-terusan di scan tiap frame
                self.identity_cache[matched_cache_key]['last_scan'] = current_time

        is_violation = v_type in ['nohelmet', 'novest']
        return {
            'bbox': list(map(int, scaled_bbox)), 
            'class': v_type,
            'confidence': float(conf),
            'worker_id': worker_id,
            'is_violation': is_violation,
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
            x1, y1, x2, y2 = map(int, bbox)
            class_name = detection['class']
            confidence = detection['confidence']

            # Format worker ID label
            raw_id = detection.get('worker_id', 'Unknown')
            if raw_id and raw_id != 'Unknown' and not raw_id.startswith('Unknown_') and '#' not in raw_id:
                id_label = raw_id
            else:
                id_label = '?'

            is_violation = class_name in ('nohelmet', 'novest')

            # Only draw VIOLATIONS — compliant workers show nothing (system is silent = safe)
            if not is_violation:
                continue

            # --- VIOLATION: Red/Orange, thick border, prominent label ---
            color = (0, 0, 255) if class_name == 'nohelmet' else (0, 140, 255)
            label = 'No Helm' if class_name == 'nohelmet' else 'No Vest'
            label += f" {confidence:.2f}"
            if id_label != '?':
                label += f" [{id_label}]"
            else:
                label += " [Unknown]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
