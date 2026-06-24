"""
Simple Violations Detector
Using helmet.v2i.yolov8 dataset directly
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import threading
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
    def __init__(self, confidence_threshold=0.60, model_path=None):
        """
        Initialize Violations Detector with simple setup
        
        Args:
            confidence_threshold: Confidence threshold for detection
        """
        self.confidence_threshold = confidence_threshold
        
        # Determine target path
        # [NEW] Prioritaskan model helmet.v2i (30 Epochs Latest) sesuai permintaan
        helmet_v2_path = os.path.join(os.path.dirname(__file__), "..", "experiments/helmet.v2i.yolov8/apd_training_tests/yolov8n_30epochs_latest/weights/best.pt")
        helmet_v2_alt = os.path.join(os.path.dirname(__file__), "..", "experiments/helmet.v2i.yolov8/helmet_vest_detection/yolov8n_50epochs_augmented/weights/best.pt")
        new_retrained_model = os.path.join(os.path.dirname(__file__), "..", "experiments/apd_detection_combined3/best.pt")
        ppe_combined_path = os.path.join(os.path.dirname(__file__), "..", "experiments/apd_finetuned/best.pt")
        premier_model_path = os.path.join(os.path.dirname(__file__), "..", "models/yolov8_50_epoch/best.pt")
        
        # Priority: 1. Helmet v2 (30 ep), 2. Helmet v2 (50 ep), 3. Combined, 4. PPE Combined, 5. Premier
        if os.path.exists(helmet_v2_path):
            target_path = helmet_v2_path
        elif os.path.exists(helmet_v2_alt):
            target_path = helmet_v2_alt
        elif os.path.exists(ppe_combined_path):
            target_path = ppe_combined_path
        elif os.path.exists(premier_model_path):
            target_path = premier_model_path
        # Override target_path to explicitly load the 4-class model we want!
        target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments', 'PPE_Combined_Dataset', 'runs', 'ppe_retraining_50_epochs_v2', 'weights', 'best.pt')
        
        # If none of the predefined models exist, use the default fallback
        if not target_path or not os.path.exists(target_path):
            target_path = os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt")
            
        if target_path and os.path.exists(target_path):
            print(f"🎯 Loading Model: {os.path.basename(os.path.dirname(os.path.dirname(target_path)))}")
            self.model = YOLO(target_path)
            
            # ── MODE DETECTOR ──────────────────────────────────────────────
            if "helmet.v2i" in target_path.lower():
                self.class_names = {0: 'helmet', 1: 'vest'}
                self.use_apd_model = False
                self.model_label = "Helmet v2 (2-Class)"
            elif "combined" in target_path.lower() or "ppe_retraining" in target_path.lower() or "finetuned" in target_path.lower():
                # BUG FIX: ppe_retraining_50_epochs_v2 is actually a 5-class model!
                self.class_names = {0: 'helmet', 1: 'no helmet', 2: 'person', 3: 'vest', 4: 'no vest'}
                self.use_apd_model = True
                self.model_label = "PPE Combined (5-Class)"
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
        self.apd_track_cache = {}   # {cam_key: [{'bbox': [x,y,x,y], 'helmet': bool, 'vest': bool, 'time': float}]}
        self.lock = threading.Lock()
        # Paksa path ke root project agar Dashboard bisa baca
        self.captures_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "captures"))
        os.makedirs(self.captures_path, exist_ok=True)
        
        # --- Temporal Detection Cache (per-camera) ---
        # Di-pisahkan per camera_id agar deteksi satu kamera tidak bocor ke kamera lain.
        # Short TTL (5 frames) to prevent false positives from persisting too long.
        self._det_cache = {}      # {camera_id: [violation entries]}
        self._det_age   = {}      # {camera_id: int}
        self.CACHE_TTL  = 5       # ~0.15s at 30fps

        
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

    def detect_violations(self, frame, enable_face_anchor=False, cctv_mode=False, camera_id=None):
        """
        Detect APD violations only (No_Helmet, No_Vest) - Optimized with smart scaling
        
        Args:
            frame: Input image frame
            cctv_mode: Jika True, gunakan threshold rendah untuk CCTV jarak jauh
            camera_id: ID kamera (string/int). Wajib diisi untuk multi-kamera agar
                       state (identity cache, temporal cache) tidak bocor antar kamera.
            
        Returns:
            List of violation detections only
        """
        # Normalise camera_id: pastikan selalu string agar bisa jadi dict key
        cam_key = str(camera_id) if camera_id is not None else "default"
        with self.lock:
            return self._detect_violations_impl(frame, enable_face_anchor, cctv_mode, cam_key)

    def _detect_violations_impl(self, frame, enable_face_anchor=False, cctv_mode=False, cam_key="default"):
        # --- PERFORMANCE FIX: Biarkan YOLO melakukan auto-scaling ---
        frame_infer = frame
            
        # Enhance lighting if image is too dark or washed out (to help YOLO in bad lighting)
        # BUG FIX: CLAHE is causing ppe_retraining_50_epochs_v2.pt to completely fail and hallucinate!
        # DO NOT modify the image before feeding to YOLO!
        pass
            
        # --- Threshold controlled by User Settings ---
        # base_conf diatur dari dashboard via self.confidence_threshold
        base_conf = float(self.confidence_threshold)
        
        # APD_CONF sedikit lebih rendah dari VIOLATION_CONF agar APD lebih mudah dideteksi
        APD_CONF       = max(0.10, base_conf - 0.05)
        VIOLATION_CONF = base_conf
        
        # 1. Get raw detections
        min_apd_conf = min(APD_CONF, VIOLATION_CONF)
        
        # FIX: Untuk CCTV, gunakan imgsz yang lebih besar (1280) agar objek kecil tidak hilang saat di-downscale ke 640
        infer_imgsz = 1280 if cctv_mode else 640
        raw_results = self.model(frame_infer, conf=min_apd_conf, imgsz=infer_imgsz, verbose=False)
        
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
                    # Person needs extremely high confidence (0.88) in main model to avoid ghost detections
                    # (we rely on the dedicated person_model below for actual person tracking)
                    p_conf_min = 0.88
                    if conf < p_conf_min:
                        continue
                    persons.append({'bbox': bbox, 'conf': conf})
                else:
                    is_violation_class = 'no' in cname  # 'no helmet' or 'no vest'
                    
                    # Vests have very low confidence in this model, so we lower the threshold aggressively
                    if cname == 'vest':
                        threshold = 0.05
                    elif cname == 'helmet':
                        threshold = 0.15
                    else:
                        threshold = VIOLATION_CONF if is_violation_class else APD_CONF
                        
                    if conf < threshold:
                        continue
                    
                    # Anti-Hallucination: Helmets should be roughly square. 
                    # If aspect ratio is extreme, it's probably a hallucination (e.g., door frame)
                    if cname == 'helmet':
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        if h > 0:
                            ar = w / h
                            if ar < 0.4 or ar > 2.5:
                                continue # Skip this hallucinated helmet
                                
                    apd_items.append({'class': cname, 'bbox': bbox, 'conf': conf})
                    
        # If model doesn't include 'person', use dedicated person_model
        # ALWAYS run person_model to catch ALL people
        if self.person_model:
            # Threshold orang di-link langsung dengan pengaturan dashboard!
            # Karena model YOLOv8n-pose/person butuh confidence sedikit lebih rendah
            # untuk mendeteksi orang yang jauh, kita kurangi 0.15 dari base_conf.
            # Jadi kalau di dashboard set 0.40, person_conf = 0.25.
            # Kalau di dashboard set 0.15, person_conf = 0.10.
            person_conf_min = max(0.10, float(self.confidence_threshold) - 0.15)
            # Selalu pakai imgsz 1280 agar orang kecil tidak hilang saat downscale
            p_imgsz = 1280
            p_results = self.person_model(frame_infer, conf=person_conf_min, classes=[0], imgsz=p_imgsz, verbose=False)
            for pr in p_results:
                for b in pr.boxes:
                    new_bbox = b.xyxy[0].cpu().numpy().tolist()
                    new_conf = float(b.conf[0].cpu().numpy())
                    
                    bw = new_bbox[2] - new_bbox[0]
                    bh = new_bbox[3] - new_bbox[1]
                    
                    # Filter 1: Ukuran minimum yang sangat longgar
                    # (15x20 px agar orang di kejauhan tetap lolos)
                    if bw < 15 or bh < 20:
                        continue
                    
                    # Filter 2: Minimum area 800 px²
                    # Orang kecil di kejauhan bisa seukuran 30x30 = 900 px²
                    if bw * bh < 800:
                        continue
                    
                    # Filter 3: Aspect ratio longgar
                    # Min 0.20: orang duduk/sebagian tubuh (close-up webcam) bisa sangat lebar
                    # Max 5.0: still blocks very thin vertical structures
                    if bw > 0:
                        ratio = bh / bw
                        if ratio < 0.05 or ratio > 5.0:
                            continue
                    
                    # Deduplicate via IoU
                    is_duplicate = False
                    for existing in persons:
                        if self._iou(new_bbox, existing['bbox']) > 0.4:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        persons.append({'bbox': new_bbox, 'conf': new_conf, 'face_bbox': None})
        
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
                    if fw < 15 or fh < 15: continue
                    
                    # Buat virtual person bbox berdasarkan proporsi manusia sungguhan
                    virtual_pb = [
                        max(0, fx1 - fw * 0.9),
                        max(0, fy1 - fh * 0.6),
                        min(frame_infer.shape[1], fx2 + fw * 0.9),
                        min(frame_infer.shape[0], fy2 + fh * 3.5)
                    ]
                    
                    is_dup = False
                    for existing in persons:
                        if self._iou(virtual_pb, existing['bbox']) > 0.05:
                            # Wajah terdeteksi = PASTI MANUSIA. Set confidence ke 0.99 agar proxy logic selalu jalan!
                            # [PERBAIKAN]: JANGAN timpa existing['bbox'] dengan virtual_pb! Bbox dari YOLO jauh lebih akurat.
                            existing['conf'] = 0.99
                            existing['face_bbox'] = f['bbox']
                            is_dup = True
                            break
                    
                    if not is_dup:
                        # CCTV FIX: Mencegah detektor wajah berhalusinasi melihat muka di pagar/turnstile CCTV.
                        # Jika wajah terlalu kecil (< 20 pixel pada frame 640x640), biarkan YOLO yang mendeteksi badannya!
                        if fw > 20 and f['confidence'] > 0.40:
                            # Wajah terdeteksi = PASTI MANUSIA. Set confidence ke 0.99 agar proxy logic selalu jalan!
                            persons.append({'bbox': virtual_pb, 'conf': 0.99, 'face_bbox': f['bbox']})
            except Exception as e:
                print(f"[DEBUG] Error di Face Anchor: {e}")
        
        violations = []
        print(f"[DEBUG] Persons found: {len(persons)}. Mode: cctv={cctv_mode}, enable_anchor={enable_face_anchor}")
        for dp in persons:
            print(f"   - Person Conf: {dp['conf']}")
        
        # Track which APD items got matched to a person
        for item in apd_items:
            item['grouped'] = False
            
        current_time = datetime.now().timestamp()
        if cam_key not in getattr(self, 'apd_track_cache', {}):
            if not hasattr(self, 'apd_track_cache'): self.apd_track_cache = {}
            self.apd_track_cache[cam_key] = []
        # Clean up old tracks (> 2 seconds)
        self.apd_track_cache[cam_key] = [t for t in self.apd_track_cache[cam_key] if current_time - t['time'] < 2.0]
        
        # --- RULE 1: Group detections by person ---
        for p in persons:
            pb = p['bbox']
            
            # Temporary state for this person
            has_helmet = False
            has_no_helmet_box = None
            has_vest = False
            has_no_vest_box = None
            matched_any_apd = False
            
            # 1. Read from Temporal Tracker
            tracked_helmet = False
            tracked_vest = False
            best_track_idx = -1
            best_iou = 0.0
            
            for idx, t in enumerate(self.apd_track_cache[cam_key]):
                iou = self._iou(pb, t['bbox'])
                if iou > 0.4 and iou > best_iou:
                    best_iou = iou
                    best_track_idx = idx
                    
            if best_track_idx >= 0:
                tracked_helmet = self.apd_track_cache[cam_key][best_track_idx]['helmet']
                tracked_vest = self.apd_track_cache[cam_key][best_track_idx]['vest']
            
            # 2. Check APD items in current frame
            for item in apd_items:
                if item.get('grouped', False): continue
                
                overlap = self._iou(item['bbox'], p['bbox'])
                
                # Cek khusus untuk CCTV: deteksi helm mungkin sedikit bergeser dari bounding box orang
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
            # Strategi: Cek SPATIAL PROXIMITY — apakah ada deteksi APD positif (helmet/vest)
            # yang secara fisik DEKAT dengan person ini?
            #
            # Ini membedakan dua kasus:
            #   A. Ghost detection jauh dari person (background noise) → person bisa jadi melanggar
            #   B. APD terdeteksi dekat person tapi gagal match via _is_inside() → person patuh
            #
            # Cara hitung: jika jarak pusat APD dari pusat person < 70% diagonal person bbox
            #              → APD dianggap "milik" person ini

            pcx = (pb[0] + pb[2]) / 2
            pcy = (pb[1] + pb[3]) / 2
            person_diag = ((pb[2] - pb[0])**2 + (pb[3] - pb[1])**2) ** 0.5
            proximity_threshold = person_diag * 0.70  # 70% diagonal sebagai radius kedekatan

            nearby_helmet = False
            nearby_vest = False
            for item in apd_items:
                acx = (item['bbox'][0] + item['bbox'][2]) / 2
                acy = (item['bbox'][1] + item['bbox'][3]) / 2
                dist = ((pcx - acx)**2 + (pcy - acy)**2) ** 0.5
                if dist <= proximity_threshold:
                    if item['class'] == 'helmet':
                        nearby_helmet = True
                    elif item['class'] in ('vest',):
                        nearby_vest = True

            # Helm
            # Hanya jalankan color heuristic helm di mode CCTV (bbox full-body vertikal).
            # Di mode Webcam (bbox lebar bahu), ini dinonaktifkan untuk mencegah tembok putih memicu false positive.
            if not has_helmet and not has_no_helmet_box:
                has_helmet = self._check_helmet_color(frame, pb, cctv_mode)
                
            if has_helmet:
                violations.append(self._create_violation_entry(frame, pb, 'helmet_ok', 1.0, is_proxy=True, cam_key=cam_key, face_bbox=p.get('face_bbox')))
            elif has_no_helmet_box:
                # Bukti eksplisit 'no helmet' dari model → laporkan dengan box aslinya
                violations.append(self._create_violation_entry(frame, has_no_helmet_box['bbox'], 'nohelmet', has_no_helmet_box['conf'], is_proxy=False, cam_key=cam_key, face_bbox=p.get('face_bbox')))
            elif not self.use_apd_model:
                # 2-class model only
                if matched_any_apd or p['conf'] > 0.70:
                    violations.append(self._create_violation_entry(frame, pb, 'nohelmet', p['conf'], is_proxy=True, cam_key=cam_key, face_bbox=p.get('face_bbox')))
            elif not nearby_helmet and p['conf'] > (0.40 if cctv_mode else 0.70):
                # Tidak ada helmet dekat person → asumsikan pelanggaran
                # (helmet yang terdeteksi jauh dari person dianggap ghost/background noise)
                violations.append(self._create_violation_entry(frame, pb, 'nohelmet', p['conf'], is_proxy=True, cam_key=cam_key, face_bbox=p.get('face_bbox')))
                print(f"[DEBUG] Added nohelmet proxy! conf={p['conf']}")
            else:
                print(f"[DEBUG] Skipped nohelmet proxy! nearby={nearby_helmet}, conf={p['conf']}")
            # Jika nearby_helmet=True tapi tidak matched → person kemungkinan berhelm, jangan vonis

            # Rompi
            # Jalankan color heuristic dengan ambang batas (threshold) dinamis berdasarkan cctv_mode
            if not has_vest and not has_no_vest_box:
                has_vest = self._check_vest_color(frame, pb, cctv_mode)
                
            # --- Temporal Rescue ---
            if not has_helmet and tracked_helmet:
                has_helmet = True
                has_no_helmet_box = None
            if not has_vest and tracked_vest:
                has_vest = True
                has_no_vest_box = None
                
            # Update Tracker Cache
            if best_track_idx >= 0:
                self.apd_track_cache[cam_key][best_track_idx]['bbox'] = pb
                self.apd_track_cache[cam_key][best_track_idx]['helmet'] = has_helmet
                self.apd_track_cache[cam_key][best_track_idx]['vest'] = has_vest
                self.apd_track_cache[cam_key][best_track_idx]['time'] = current_time
            else:
                self.apd_track_cache[cam_key].append({
                    'bbox': pb, 'helmet': has_helmet, 'vest': has_vest, 'time': current_time
                })
            
            if has_vest:
                violations.append(self._create_violation_entry(frame, pb, 'vest_ok', 1.0, is_proxy=True, cam_key=cam_key, face_bbox=p.get('face_bbox')))
            elif has_no_vest_box:
                # Bukti eksplisit 'no vest' dari model → laporkan dengan box aslinya
                violations.append(self._create_violation_entry(frame, has_no_vest_box['bbox'], 'novest', has_no_vest_box['conf'], is_proxy=False, cam_key=cam_key, face_bbox=p.get('face_bbox')))
            elif not self.use_apd_model:
                # 2-class model only
                if matched_any_apd or p['conf'] > (0.40 if cctv_mode else 0.70):
                    if not cctv_mode:
                        violations.append(self._create_violation_entry(frame, pb, 'novest', p['conf'], is_proxy=True, cam_key=cam_key, face_bbox=p.get('face_bbox')))
            elif not nearby_vest and p['conf'] > (0.40 if cctv_mode else 0.70):
                # Tidak ada rompi dekat person → laporkan
                violations.append(self._create_violation_entry(frame, pb, 'novest', p['conf'], is_proxy=True, cam_key=cam_key, face_bbox=p.get('face_bbox')))
                print(f"[DEBUG] Added novest proxy! conf={p['conf']}")
            else:
                print(f"[DEBUG] Skipped novest proxy! nearby={nearby_vest}, conf={p['conf']}")
            # Jika nearby_vest=True tapi tidak matched → person kemungkinan berrompi, jangan vonis

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
                iy2 + (ih * 1.5 if 'helmet' in item_class else ih * 0.5)
            ]
            v_type = item_class.replace(' ', '') + '_ok'
            violations.append(self._create_violation_entry(frame, virtual_pb, v_type, item['conf'], is_proxy=True, cam_key=cam_key))

        # --- Temporal Cache (per-camera): prevent flickering ---
        if violations:
            # Got real detections - update cache and reset age for this camera
            self._det_cache[cam_key] = violations
            self._det_age[cam_key] = 0
        elif self._det_age.get(cam_key, 0) < self.CACHE_TTL:
            # No detections this frame, but cache is still fresh - use it
            self._det_age[cam_key] = self._det_age.get(cam_key, 0) + 1
            return self._det_cache.get(cam_key, [])
        else:
            # Cache expired - truly no detections
            self._det_cache[cam_key] = []
            self._det_age[cam_key] = 0

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
        """Check if an APD bbox is reasonably within a person area.
        
        Margin diperlebar untuk CCTV agar helm/rompi yang sedikit bergeser
        (akibat perspektif kamera jauh) tetap ter-match ke person.
        """
        ax1, ay1, ax2, ay2 = apd_bbox
        px1, py1, px2, py2 = person_bbox
        
        # Center of APD
        acx = (ax1 + ax2) / 2
        acy = (ay1 + ay2) / 2
        
        # Horizontal check: izinkan APD di luar max 80% lebar person
        # (lebih lebar dari sebelumnya 60%, untuk CCTV sudut jauh)
        w = px2 - px1
        margin_x = w * 0.80
        if acx < (px1 - margin_x) or acx > (px2 + margin_x):
            return False
        
        # Cek tumpang tindih (overlap) bbox APD vs bbox person secara langsung
        # Ini lebih robust daripada hanya cek titik tengah
        overlap_x = min(ax2, px2) - max(ax1, px1)
        overlap_y = min(ay2, py2) - max(ay1, py1)
        if overlap_x > 0 and overlap_y > 0:
            # Ada overlap langsung: pasti match
            return True
        
        # Jika tidak ada overlap, cek berdasarkan tipe APD
        if type == 'head':
            person_height = py2 - py1
            # Helm bisa di atas bbox person (sudut kamera rendah), toleransi 50% tinggi
            if not (py1 - person_height * 0.50 <= acy <= py1 + person_height * 0.85):
                return False
            return True
            
        # Torso (vest): harus berada di rentang vertikal person
        return py1 - (py2 - py1) * 0.25 <= acy <= py2

    def _create_violation_entry(self, frame, bbox, v_type, conf, is_proxy=False, cam_key="default", face_bbox=None):
        """Helper to create a standard violation entry with face recognition"""
        if is_proxy:
            px1, py1, px2, py2 = bbox
            p_w = px2 - px1
            p_h = py2 - py1
            
            if v_type in ['nohelmet', 'helmet_ok']:
                if face_bbox is not None:
                    # Precise Head Anchoring using Face BBox
                    fx1, fy1, fx2, fy2 = face_bbox
                    f_h = fy2 - fy1
                    scaled_bbox = [
                        int(fx1),
                        int(fy1 - f_h * 0.5),
                        int(fx2),
                        int(fy1 + f_h * 0.2)
                    ]
                else:
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
        
        # Cari di cache menggunakan IoU dari kotak person, HANYA dalam kamera yang sama.
        # Ini mencegah identitas dari kamera A bocor ke kamera B.
        matched_cache_key = None
        for k, v in list(self.identity_cache.items()):
            # Hapus cache lama (> 4 detik)
            if current_time - v['time'] > 4.0:
                del self.identity_cache[k]
                continue

            # PERBAIKAN MULTI-CAM: skip entry dari kamera lain
            if v.get('cam_key') != cam_key:
                continue
            
            # Cek overlap dengan threshold yang sangat ketat (0.7) untuk mencegah ID melompat ke orang lain di sebelahnya
            if 'person_bbox' in v and self._iou(bbox, v['person_bbox']) > 0.7:
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
                # FIX: Buat kotak pencarian berwujud PERSEGI (SQUARE) di bagian atas badan.
                # Ini MENCEGAH distorsi saat Face Detector (DNN SSD) meresize gambar ke 300x300.
                px1, py1, px2, py2 = bbox
                w = px2 - px1
                
                # Buat kotak persegi (lebar = tinggi) dari posisi paling atas
                sq_y2 = min(frame.shape[0], py1 + w)
                recon_bbox = [int(px1), int(py1), int(px2), int(sq_y2)]
            else:
                recon_bbox = scaled_bbox
                
            # FaceRecognitionSystem kita sekarang sudah stabil (sudah punya tracking internal)
            recognized, new_sim = self.face_recognizer.recognize_face(frame, recon_bbox)
            
            if recognized:
                metadata = self.face_recognizer.face_metadata.get(recognized, {})
                worker_id = metadata.get('name', recognized)
                face_sim = float(new_sim)
                
                # Simpan/update ke cache (dengan cam_key agar tidak bocor ke kamera lain)
                if not matched_cache_key:
                    import uuid
                    matched_cache_key = str(uuid.uuid4())
                    
                self.identity_cache[matched_cache_key] = {
                    'id': worker_id,
                    'sim': face_sim,
                    'time': current_time,
                    'last_scan': current_time,
                    'person_bbox': bbox,
                    'cam_key': cam_key  # tag kamera agar tidak bocor ke kamera lain
                }
            elif matched_cache_key:
                # Update waktu scan supaya tidak terus-terusan di scan tiap frame
                self.identity_cache[matched_cache_key]['last_scan'] = current_time
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
                        # PERBAIKAN MULTI-CAM: unknown face memory di-namespace per kamera
                        # Prefix key dengan cam_key agar unknown di cam A tidak match ke cam B
                        # Ganti ':' menjadi '_' karena ID ini akan dipakai untuk nama folder di OS
                        cam_prefix = f"{cam_key}_"
                        for tid, data in list(self.unknown_face_memory.items()):
                            if now - data['last_seen'] > 600: # 10 menit
                                del self.unknown_face_memory[tid]
                                continue
                            # Skip entries dari kamera lain
                            if not tid.startswith(cam_prefix):
                                continue
                            sim_unknown = cosine_similarity([current_encoding], [data['encoding']])[0][0]
                            if sim_unknown > 0.40: # Threshold untuk unknown grouping
                                matched_temp_id = tid
                                break
                        
                        if not matched_temp_id:
                            # Buat temp ID baru dengan prefix kamera
                            import secrets
                            matched_temp_id = f"{cam_prefix}Unknown_{secrets.token_hex(4)}"
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
                                print(f"📸 [Capture] Wajah baru terdeteksi! Disimpan ke: {matched_temp_id}")
                except Exception as e:
                    print(f"⚠️ Error in unknown capture logic: {e}")

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

            # If both violations apply to the same person, shift the label so they don't overlap
            y_offset = y1 - 10
            if class_name == 'novest':
                y_offset = y1 + 15
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Text background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y_offset - th - 5), (x1 + tw + 10, y_offset + 5), color, -1)
            cv2.putText(frame, label, (x1 + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _check_vest_color(self, frame, pb, cctv_mode=False):
        """
        Color-based heuristic for detecting highly reflective safety vests (neon orange/yellow).
        Used as a robust fallback when YOLO fails due to cropping or poor visibility.
        """
        try:
            x1, y1, x2, y2 = map(int, pb)
            h = y2 - y1
            if h < 10: return False
            
            # Analyze the lower 80% of the bounding box (from 20% to 100% height)
            # This covers the torso/chest area in full-body shots AND the bottom edge in close-up webcam shots.
            y_start = int(y1 + h * 0.2)
            y_end = y2
            roi = frame[max(0, y_start):min(frame.shape[0], y_end), max(0, x1):min(frame.shape[1], x2)]
            if roi.size == 0: return False
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Neon Orange (H: 0-25, S: 150-255, V: 100-255)
            # S > 150 mencegah kulit (shirtless) terdeteksi sebagai rompi
            lower_orange = np.array([0, 150, 100])
            upper_orange = np.array([25, 255, 255])
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            
            # Neon Red-Orange Wrap-around (H: 165-180, S: 150-255, V: 100-255)
            lower_red = np.array([165, 150, 100])
            upper_red = np.array([180, 255, 255])
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            
            # Neon Yellow/Green (H: 20-85, S: 120-255, V: 100-255)
            lower_yellow = np.array([20, 120, 100])
            upper_yellow = np.array([85, 255, 255])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            mask = cv2.bitwise_or(mask_orange, mask_red)
            mask = cv2.bitwise_or(mask, mask_yellow)
            
            # If > 5% of the torso region is neon-colored, assume vest is worn
            # We increased from 1% to 5% to prevent tiny reflections from triggering it
            ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
            return ratio > 0.05
        except Exception as e:
            return False
            
    def _check_helmet_color(self, frame, pb, cctv_mode=False):
        """
        Color-based heuristic for detecting white/yellow helmets.
        Used as a fallback when YOLO fails due to brightness or angle.
        Hanya aktif di mode CCTV!
        """
        if not cctv_mode:
            return False
            
        try:
            x1, y1, x2, y2 = map(int, pb)
            w = x2 - x1
            h = y2 - y1
            if h < 10 or w < 10: return False
            
            # Analyze top 15% and middle 50% of the bounding box (exactly where a helmet sits)
            x_start = int(x1 + w * 0.25)
            x_end = int(x1 + w * 0.75)
            y_end = int(y1 + h * 0.15)
            
            roi = frame[max(0, y1):min(frame.shape[0], y_end), max(0, x_start):min(frame.shape[1], x_end)]
            if roi.size == 0: return False
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # White hard hat (shaded or bright, low saturation)
            lower_white = np.array([0, 0, 130])
            upper_white = np.array([180, 60, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            # Yellow hard hat
            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([45, 255, 255])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            mask = cv2.bitwise_or(mask_white, mask_yellow)
            
            # If > 15% of this tight head region is strictly white/yellow, assume helmet.
            # We use 15% to prevent false positives from ceiling lamps, white walls, etc.
            ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
            return ratio > 0.15
        except Exception as e:
            return False
    
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
