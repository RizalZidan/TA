#!/usr/bin/env python3
"""
APD Monitoring System - Advanced Version
Features: Login, Multi-Camera, Data Recap
"""

import os
import sys
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, Response, request, redirect, url_for, session, flash
import cv2
import threading
import time
import base64
import json

# Suppress OpenCV warnings untuk RTSP timeout (warning tidak berbahaya, hanya info)
# Warning ini muncul karena FFmpeg menggunakan timeout 30s di level C++
# Tapi kita sudah handle timeout di level Python dengan thread timeout
try:
    # Try to set OpenCV log level (if available in this version)
    cv2.setLogLevel(1)  # 0=Silent, 1=Error, 2=Warn, 3=Info, 4=Debug
except:
    # If not available, try environment variable
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.violations_detector import ViolationsDetector

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Project Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'apd_monitoring.db')
FACE_DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'face_database.pkl')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'operator',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Violations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            violation_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            processed BOOLEAN DEFAULT FALSE
        )
    ''')
    
    # Cameras table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            status TEXT DEFAULT 'inactive',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Workers table (Source of Truth for Names)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # NEW: Face recognition log table for accuracy metrics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_recognition_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            worker_id TEXT,
            similarity REAL,
            camera_id INTEGER
        )
    ''')
    
    # Create default admin user
    admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, password, role) 
        VALUES (?, ?, ?)
    ''', ('admin', admin_password, 'admin'))
    
    # Create default petugas user
    petugas_password = hashlib.sha256('petugas123'.encode()).hexdigest()
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, password, role) 
        VALUES (?, ?, ?)
    ''', ('petugas', petugas_password, 'petugas'))
    
    conn.commit()
    conn.close()

    # --- Migration: tambah kolom worker_id jika belum ada ---
    conn2 = sqlite3.connect(DB_PATH)
    cur2 = conn2.cursor()
    try:
        cur2.execute('ALTER TABLE violations ADD COLUMN worker_id TEXT DEFAULT "Unknown"')
    except Exception:
        pass
        
    try:
        cur2.execute('ALTER TABLE violations ADD COLUMN processed BOOLEAN DEFAULT FALSE')
    except Exception:
        pass
        
    conn2.commit()
    conn2.close()

# Initialize database on startup
init_db()

# Use the high-accuracy PREMIER 50-Epoch model (86.03%)
# Now safe to use because coordinate shifting has been fixed.
detector = ViolationsDetector(confidence_threshold=0.40)

# Migration: Sync workers from pickle to SQLite if needed
def sync_workers_to_db():
    if not hasattr(detector, 'face_recognizer') or not detector.face_recognizer:
        return
    
    fr = detector.face_recognizer
    if not fr.face_metadata:
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if workers table is empty
    cursor.execute('SELECT COUNT(*) FROM workers')
    if cursor.fetchone()[0] == 0:
        print("[*] Syncing workers from face database to SQLite...")
        for worker_id, metadata in fr.face_metadata.items():
            name = metadata.get('name', worker_id)
            cursor.execute('INSERT OR IGNORE INTO workers (worker_id, name) VALUES (?, ?)', (worker_id, name))
        conn.commit()
    conn.close()

sync_workers_to_db()

cameras = {}
camera_threads = {}
camera_stats = {}
global_stats = {
    'total_violations': 0,
    'no_helmet_count': 0,
    'no_vest_count': 0,
    'active_cameras': 0
}
tracked_persons = {}  # {camera_id: {person_id: {last_seen_time, violations}}}
# Disabled (0) to allow temporal stabilizer to process every frame correctly
detection_cooldown = 0 

# Global state for frame sharing (Thread-safe)
camera_frames = {}  # {camera_id: {'frame': frame, 'detections': [], 'timestamp': time.time()}}
frame_lock = threading.Lock()
worker_cooldowns = {} # {worker_id: last_violation_time}
COOLDOWN_SECONDS = 5 # Default global cooldown per ID

# Diagnostic function untuk test RTSP dengan detail
def diagnose_rtsp_connection(url, timeout_sec=8):
    """Diagnose RTSP connection dengan informasi detail"""
    import socket
    from urllib.parse import urlparse
    
    results = {
        'url': url,
        'host': None,
        'port': None,
        'network_reachable': False,
        'rtsp_port_open': False,
        'connection_tests': [],
        'recommendations': []
    }
    
    try:
        # Parse URL
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or 554  # Default RTSP port
        
        results['host'] = host
        results['port'] = port
        
        # Test 1: Network connectivity (ping test via socket)
        print(f"🔍 Testing network connectivity to {host}:{port}...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                results['network_reachable'] = True
                results['rtsp_port_open'] = True
                print(f"✅ Network reachable: {host}:{port} is open")
            else:
                results['network_reachable'] = False
                print(f"❌ Network unreachable: {host}:{port} is closed or filtered")
                results['recommendations'].append(f"Check if camera IP {host} is accessible from this machine")
                results['recommendations'].append(f"Check firewall rules for port {port}")
        except Exception as e:
            print(f"❌ Network test failed: {e}")
            results['recommendations'].append(f"Network test error: {str(e)}")
        
        # Test 2: Try different RTSP URL formats
        print(f"🔍 Testing different RTSP URL formats...")
        test_urls = []
        
        # Original URL
        test_urls.append(('Original', url))
        
        # Try with different paths
        base_url = f"{parsed.scheme}://"
        if parsed.username and parsed.password:
            base_url += f"{parsed.username}:{parsed.password}@"
        base_url += f"{host}"
        if port != 554:
            base_url += f":{port}"
        
        # Common RTSP paths
        common_paths = [
            '/live.sdp',
            '/live',
            '/stream',
            '/cam',
            '/camera1',
            '/media',
            '/video',
            '/h264',
            '/main',
            '/sub'
        ]
        
        for path in common_paths:
            if path not in url:
                test_urls.append((f'Path: {path}', base_url + path))
        
        # Try with transport options
        if '?' not in url:
            test_urls.append(('TCP Transport', url + '?transport=tcp'))
            test_urls.append(('UDP Transport', url + '?transport=udp'))
        
        # Test each URL
        for test_name, test_url in test_urls[:10]:  # Limit to 10 tests
            print(f"📡 Testing: {test_name} - {test_url}")
            test_result = test_rtsp_connection(test_url, timeout_sec=timeout_sec)
            
            results['connection_tests'].append({
                'name': test_name,
                'url': test_url,
                'success': test_result['success'],
                'error': test_result['error']
            })
            
            if test_result['success']:
                print(f"✅ SUCCESS with {test_name}!")
                results['recommendations'].append(f"✅ Use this URL: {test_url}")
                if test_result['cap']:
                    test_result['cap'].release()
                break
            else:
                print(f"❌ Failed: {test_result['error']}")
        
        # Generate recommendations
        if not any(t['success'] for t in results['connection_tests']):
            results['recommendations'].append("❌ All RTSP connection attempts failed")
            results['recommendations'].append("💡 Try testing the URL in VLC player first")
            results['recommendations'].append("💡 Verify camera credentials and IP address")
            results['recommendations'].append("💡 Check if camera supports RTSP protocol")
            if not results['network_reachable']:
                results['recommendations'].append("⚠️ Network connectivity issue detected - fix this first")
        
    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
        results['recommendations'].append(f"Diagnostic error: {str(e)}")
    
    return results

# Helper function untuk test RTSP connection dengan timeout (cross-platform)
def test_rtsp_connection(url, timeout_sec=5):
    """Test RTSP connection dengan timeout yang lebih pendek (menggunakan threading)"""
    result = {'success': False, 'cap': None, 'error': None}
    cap_container = {'cap': None}
    cap_ref = {'cap': None}  # Reference untuk force release saat timeout
    exception_container = {'exception': None}
    stop_flag = threading.Event()
    
    def connect_rtsp():
        """Function yang dijalankan di thread terpisah"""
        cap = None
        try:
            # Buat VideoCapture dengan timeout yang lebih pendek
            # Gunakan CAP_FFMPEG dengan opsi timeout
            timeout_ms = timeout_sec * 1000
            
            # Coba set environment variable untuk FFmpeg (jika didukung)
            # Note: Ini mungkin tidak bekerja di semua sistem, tapi tidak akan error
            try:
                os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '1'
                os.environ['OPENCV_FFMPEG_READ_ATTEMPT_MSEC'] = str(timeout_ms)
                
                # Parse transport from url and set environment variable correctly for OpenCV FFmpeg
                if '?transport=tcp' in url:
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
                    url = url.replace('?transport=tcp', '')
                elif '?transport=udp' in url:
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
                    url = url.replace('?transport=udp', '')
                else:
                    # TCP Transport often more reliable for wifi/lan
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            except:
                pass
            
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap_ref['cap'] = cap  # Simpan reference untuk force release jika timeout
            
            # Set properties dengan timeout lebih pendek
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Check if opened dengan polling (lebih cepat detect)
            start_check = time.time()
            is_opened = False
            max_check_time = timeout_sec * 0.8  # 80% dari timeout untuk check
            
            # Polling isOpened dengan interval pendek
            while not stop_flag.is_set() and (time.time() - start_check) < max_check_time:
                try:
                    is_opened = cap.isOpened()
                    if is_opened:
                        break
                except:
                    pass
                time.sleep(0.05)  # Check setiap 50ms untuk lebih responsif
            
            check_elapsed = time.time() - start_check
            
            if stop_flag.is_set():
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
                cap_ref['cap'] = None
                exception_container['exception'] = 'Cancelled by timeout'
                return
            
            if not is_opened:
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
                cap_ref['cap'] = None
                exception_container['exception'] = f'Failed to open ({check_elapsed:.1f}s)'
                return
            
            # Try read frame dengan timeout yang tersisa
            remaining_time = timeout_sec - check_elapsed
            if remaining_time < 0.5:
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
                cap_ref['cap'] = None
                exception_container['exception'] = 'Not enough time for read'
                return
            
            start_time = time.time()
            ret = False
            frame = None
            
            # Try read dengan timeout check
            while not stop_flag.is_set() and (time.time() - start_time) < remaining_time:
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                except:
                    pass
                time.sleep(0.05)  # Check setiap 50ms
            
            read_elapsed = time.time() - start_time
            
            if stop_flag.is_set():
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
                cap_ref['cap'] = None
                exception_container['exception'] = 'Cancelled by timeout'
                return
            
            if not ret or frame is None or frame.size == 0:
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
                cap_ref['cap'] = None
                exception_container['exception'] = f'No valid frame ({read_elapsed:.1f}s)'
                return
            
            # Success!
            cap_container['cap'] = cap
            cap_ref['cap'] = None  # Clear reference karena sudah di container
            cap = None  # Prevent release
                
        except Exception as e:
            exception_container['exception'] = f'Exception: {str(e)}'
        finally:
            # Pastikan cap di-release jika tidak berhasil
            if cap:
                try:
                    cap.release()
                except:
                    pass
            # Clear reference
            if cap_ref['cap'] == cap:
                cap_ref['cap'] = None
    
    # Jalankan di thread terpisah dengan timeout
    thread = threading.Thread(target=connect_rtsp, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec + 0.5)  # Beri sedikit extra time
    
    if thread.is_alive():
        # Thread masih berjalan = timeout, set stop flag untuk cancel
        stop_flag.set()
        
        # Force release VideoCapture jika masih ada (dari cap_ref atau cap_container)
        # Ini penting untuk mencegah FFmpeg terus mencoba koneksi di background
        if cap_ref['cap']:
            try:
                cap_ref['cap'].release()
            except:
                pass
            cap_ref['cap'] = None
        
        if cap_container['cap']:
            try:
                cap_container['cap'].release()
            except:
                pass
            cap_container['cap'] = None
        
        # Tunggu sebentar untuk thread selesai (non-blocking)
        thread.join(timeout=0.3)
        
        # Note: Warning FFmpeg 30s mungkin masih muncul di log setelah ini
        # Ini NORMAL dan tidak berbahaya - itu hanya info dari FFmpeg di level C++
        # Yang penting: koneksi sudah di-cancel di level Python setelah 5s
        # dan VideoCapture sudah di-release, jadi tidak ada resource leak
        result['error'] = f'Connection timeout after {timeout_sec}s'
        print(f"ℹ️  Note: FFmpeg warning (30s) may appear but connection already cancelled at {timeout_sec}s")
        return result
    
    if exception_container['exception']:
        result['error'] = exception_container['exception']
        return result
    
    if cap_container['cap']:
        result['success'] = True
        result['cap'] = cap_container['cap']
    else:
        result['error'] = 'Unknown error'
    
    return result

# Camera monitoring functions
def start_camera_monitoring(camera_id, camera_source):
    """Start monitoring a specific camera with enhanced RTSP support"""
    global cameras, camera_threads, tracked_persons
    
    # Normalize source string to avoid leading/trailing whitespace issues
    if isinstance(camera_source, str):
        camera_source = camera_source.strip()

    if camera_id in cameras:
        return False
    
    print(f"🎯 Starting camera {camera_id} with source: {camera_source}")
    
    # Enhanced RTSP connection dengan timeout lebih pendek
    if camera_source.startswith('rtsp://'):
        cap = None
        connection_timeout = 5  # 5 detik per method
        
        # Method 1: Direct connection
        print(f"📡 Method 1: Direct RTSP connection (timeout: {connection_timeout}s)...")
        result = test_rtsp_connection(camera_source, connection_timeout)
        if result['success']:
            print(f"✅ Direct RTSP connection successful!")
            cap = result['cap']
            cameras[camera_id] = cap
        else:
            print(f"❌ Direct RTSP failed: {result['error']}")
        
        # Method 2: TCP transport
        if cap is None:
            print(f"📡 Method 2: RTSP with TCP transport (timeout: {connection_timeout}s)...")
            tcp_url = f"{camera_source}?transport=tcp"
            result = test_rtsp_connection(tcp_url, connection_timeout)
            if result['success']:
                print(f"✅ TCP RTSP connection successful!")
                cap = result['cap']
                cameras[camera_id] = cap
            else:
                print(f"❌ TCP RTSP failed: {result['error']}")
        
        # Method 3: UDP transport
        if cap is None:
            print(f"📡 Method 3: RTSP with UDP transport (timeout: {connection_timeout}s)...")
            udp_url = f"{camera_source}?transport=udp"
            result = test_rtsp_connection(udp_url, connection_timeout)
            if result['success']:
                print(f"✅ UDP RTSP connection successful!")
                cap = result['cap']
                cameras[camera_id] = cap
            else:
                print(f"❌ UDP RTSP failed: {result['error']}")
        
        # Method 4: Alternative paths (hanya jika URL mengandung /live)
        if cap is None and '/live' in camera_source:
            print(f"📡 Method 4: Trying alternative RTSP paths (timeout: {connection_timeout}s)...")
            alternatives = [
                camera_source.replace('/live', '/stream'),
                camera_source.replace('/live', '/camera1'),
                camera_source.replace('/live', '/media'),
            ]
            
            for alt_url in alternatives:
                if alt_url != camera_source:
                    print(f"📡 Trying: {alt_url}")
                    result = test_rtsp_connection(alt_url, connection_timeout)
                    if result['success']:
                        print(f"✅ Alternative RTSP connection successful: {alt_url}")
                        cap = result['cap']
                        cameras[camera_id] = cap
                        break
                    else:
                        print(f"❌ Failed: {result['error']}")
        
        if cap is None:
            print(f"\n❌ All RTSP connection methods failed for {camera_source}")
            print(f"\n🔍 Running diagnostic to identify the issue...")
            print(f"{'='*60}")
            
            # Run diagnostic untuk mengetahui masalahnya
            diagnostic = diagnose_rtsp_connection(camera_source, timeout_sec=6)
            
            print(f"\n{'='*60}")
            print(f"📊 DIAGNOSTIC SUMMARY:")
            print(f"{'='*60}")
            print(f"Network Reachable: {'✅ YES' if diagnostic['network_reachable'] else '❌ NO'}")
            print(f"RTSP Port Open: {'✅ YES' if diagnostic['rtsp_port_open'] else '❌ NO'}")
            
            if diagnostic['network_reachable'] and not diagnostic['rtsp_port_open']:
                print(f"\n⚠️  ISSUE IDENTIFIED: Network reachable but RTSP port is closed/filtered")
                print(f"   → This suggests a firewall or port blocking issue")
                print(f"   → The RTSP URL might be correct, but port {diagnostic.get('port', 554)} is blocked")
            elif not diagnostic['network_reachable']:
                print(f"\n⚠️  ISSUE IDENTIFIED: Network connectivity problem")
                print(f"   → Camera IP is not reachable from this machine")
                print(f"   → Check network configuration, VPN, or firewall rules")
            else:
                print(f"\n⚠️  ISSUE IDENTIFIED: RTSP URL or authentication problem")
                print(f"   → Network is OK, but RTSP connection failed")
                print(f"   → Check RTSP URL format, username/password, or camera settings")
            
            # Show recommendations
            if diagnostic.get('recommendations'):
                print(f"\n💡 RECOMMENDATIONS:")
                for rec in diagnostic['recommendations'][:5]:  # Show top 5
                    print(f"   {rec}")
            
            # Check if any test was successful
            successful_test = None
            for test in diagnostic.get('connection_tests', []):
                if test.get('success'):
                    successful_test = test
                    break
            
            if successful_test:
                print(f"\n✅ FOUND WORKING URL: {successful_test['url']}")
                print(f"   Try using this URL instead!")
            
            print(f"{'='*60}\n")
            return False
            
    else:
        # Regular webcam/file connection
        source = int(camera_source) if camera_source.isdigit() else camera_source
        
        # On Windows, open webcam with silent multi-backend fallback.
        # CAP_ANY lets OpenCV choose the best available backend automatically.
        if isinstance(source, int) and os.name == 'nt':
            backends = [
                (cv2.CAP_ANY,  "Auto"),
                (cv2.CAP_MSMF, "MSMF"),
                (-1,           "Default"),   # -1 = no backend hint
            ]
            cap = None
            used_backend = "None"
            for backend_id, backend_name in backends:
                try:
                    if backend_id == -1:
                        _cap = cv2.VideoCapture(source)
                    else:
                        _cap = cv2.VideoCapture(source, backend_id)
                    ret, _ = _cap.read()
                    if ret and _cap.isOpened():
                        cap = _cap
                        used_backend = backend_name
                        break
                    else:
                        _cap.release()
                except Exception:
                    pass
            
            if cap is None:
                print(f"❌ All webcam backends failed for source {source}")
                return False
            print(f"✅ Webcam {source} opened via {used_backend} backend")
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Failed to open camera source: {camera_source}")
            return False
            
        # Try reading a test frame to ensure it actually works
        ret, _ = cap.read()
        if not ret:
            print(f"❌ Camera source opened but failed to read frames: {camera_source}")
            cap.release()
            return False
            
        cameras[camera_id] = cap
    
    # Safety: make sure camera handle really exists before starting thread
    if camera_id not in cameras:
        print(f"❌ Camera {camera_id} not initialized correctly, aborting start")
        return False
    
    tracked_persons[camera_id] = {}
    # camera_stats[camera_id] = {'fps': 0.0}
    
    # Start monitoring thread
    thread = threading.Thread(target=monitor_camera, args=(camera_id, cameras[camera_id]), daemon=True)
    camera_threads[camera_id] = thread
    thread.start()
    
    # Update camera status in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE cameras SET status = ? WHERE id = ?', ('active', camera_id))
    conn.commit()
    conn.close()
    
    print(f"✅ Camera {camera_id} monitoring started successfully!")
    return True

def stop_camera_monitoring(camera_id):
    """Stop monitoring a specific camera"""
    global cameras, camera_threads
    
    if camera_id in cameras:
        cameras[camera_id].release()
        del cameras[camera_id]
    
    if camera_id in camera_threads:
        del camera_threads[camera_id]
    
    if camera_id in tracked_persons:
        del tracked_persons[camera_id]
    
    if camera_id in camera_stats:
        del camera_stats[camera_id]
    
    # Update camera status in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE cameras SET status = ? WHERE id = ?', ('inactive', camera_id))
    conn.commit()
    conn.close()
    
    return True

def monitor_camera(camera_id, cap):
    """Monitor a single camera for violations"""
    global tracked_persons, global_stats
    
    frame_count = 0
    start_time = time.time()
    consecutive_failures = 0
    max_failures = 10
    
    while camera_id in cameras and cameras[camera_id].isOpened():
        # Add timeout check untuk read operation
        read_start = time.time()
        ret, frame = cameras[camera_id].read()
        read_elapsed = time.time() - read_start
        
        # Jika read terlalu lama (>2 detik), skip frame ini
        if read_elapsed > 2.0:
            print(f"⚠️ Camera {camera_id} read took {read_elapsed:.1f}s, skipping frame")
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(f"❌ Camera {camera_id} too many slow reads, stopping monitoring")
                break
            time.sleep(0.1)
            continue
        
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(f"❌ Camera {camera_id} too many read failures, stopping monitoring")
                break
            time.sleep(0.1)
            continue
            
        # --- OPTIMIZATION: BUFFER FLUSHING ---
        # Grab only the most recent frame if there are many waiting in the buffer
        # This prevents the stream from lagging behind real-time
        if frame_count % 5 == 0:
            for _ in range(5):
                if camera_id not in cameras:
                    break
                temp_ret, temp_frame = cameras[camera_id].read()
                if temp_ret:
                    frame = temp_frame
                else:
                    break
        
        consecutive_failures = 0  # Reset on success
        
        frame_count += 1
        
        # --- OPTIMIZATION: FRAME SKIPPING ---
        # Only run AI every 5 frames to keep the feed smooth (Optimized from 3)
        if frame_count % 5 == 0:
            try:
                is_cctv = str(camera_id) != '0'
                detections = detector.detect_violations(frame, enable_face_anchor=(not is_cctv))
            except Exception as e:
                import traceback
                with open('error_log.txt', 'a') as f:
                    f.write(f"Error on camera {camera_id}: {str(e)}\n{traceback.format_exc()}\n")
                print(f"⚠️ Error detecting violations on camera {camera_id}: {str(e)}")
                detections = []
        else:
            # Re-use last detections to avoid flickering while skipping AI
            with frame_lock:
                if camera_id in camera_frames:
                    detections = camera_frames[camera_id].get('detections', [])
                else:
                    detections = []
            
        # Update shared frame for video feed
        with frame_lock:
            camera_frames[camera_id] = {
                'frame': frame.copy(),
                'detections': detections,
                'timestamp': time.time()
            }
        
        # Process detections with person tracking
        current_time = time.time()
        if camera_id not in tracked_persons:
            tracked_persons[camera_id] = {}
        
        for detection in detections:
            if camera_id not in tracked_persons:
                break
            class_name = detection['class']
            
            # Create person ID based on bounding box position
            # IMPROVEMENT: Use a much finer grid (8px) to distinguish between people close together
            bbox = detection.get('bbox', [0, 0, 0, 0])
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            person_id = f"person_{center_x // 8}_{center_y // 8}"
            
            # Check if this person exists
            if person_id not in tracked_persons[camera_id]:
                tracked_persons[camera_id][person_id] = {
                    'last_seen': current_time,
                    'worker_id': 'Unknown',
                    'violations': {
                        'no_helmet': False,
                        'no_vest': False
                    },
                    'violation_counters': {
                        'nohelmet': 0,
                        'novest': 0
                    }
                }
            
            person_data = tracked_persons[camera_id][person_id]
            
            # --- IMPROVEMENT: Reset counters if APD is found ---
            # Jika kelasnya 'ok', berarti APD terdeteksi. Reset counter pelanggaran.
            if class_name == 'helmet_ok':
                person_data['violation_counters']['nohelmet'] = 0
                person_data['violations']['no_helmet'] = False
                # Important: update last_seen even for safe frames to keep tracking alive
                person_data['last_seen'] = current_time 
                continue 
            elif class_name == 'vest_ok':
                person_data['violation_counters']['novest'] = 0
                person_data['violations']['no_vest'] = False
                person_data['last_seen'] = current_time
                continue
            
            # USE CACHED Recognition if available
            if detection['worker_id'] != 'Unknown':
                person_data['worker_id'] = detection['worker_id']
            else:
                detection['worker_id'] = person_data['worker_id']

            # --- STABILIZER LOGIC: Reset on SAFE signal ---
            if class_name == 'helmet_ok':
                person_data['violation_counters']['nohelmet'] = 0
                person_data['violations']['no_helmet'] = False
                person_data['last_seen'] = current_time
                continue
            elif class_name == 'vest_ok':
                person_data['violation_counters']['novest'] = 0
                person_data['violations']['no_vest'] = False
                person_data['last_seen'] = current_time
                continue
            
            # If it's a violation, proceed to logging logic
            is_no_helmet = (class_name == 'nohelmet')
            is_no_vest = (class_name == 'novest')
            
            if not is_no_helmet and not is_no_vest:
                person_data['last_seen'] = current_time
                continue
            
            # Check if this is a new detection (after cooldown)
            if current_time - person_data['last_seen'] > detection_cooldown:
                
                # Update violation status
                # --- LOGIKA STABILIZER TERAPAN ---
                # Kita hanya mencatat pelanggaran jika terdeteksi secara konsisten selama N frame
                # Jika terdeteksi aman (APD terpakai), counter langsung reset ke 0.
                
                # Cek tipe pelanggaran di data deteksi (ini adalah proxy yang dibuat oleh ViolationsDetector)
                if class_name == 'nohelmet':
                    person_data['violation_counters']['nohelmet'] += 1
                elif class_name == 'novest':
                    person_data['violation_counters']['novest'] += 1
                
                # Logika Logging berdasarkan counter
                STABILIZER_THRESHOLD = 10 # 10 frame berturut-turut (~1 detik)
                
                if class_name == 'nohelmet':
                    if person_data['violation_counters']['nohelmet'] >= STABILIZER_THRESHOLD:
                        if not person_data['violations']['no_helmet']:
                            person_data['violations']['no_helmet'] = True
                            
                            # COOLDOWN CHECK PER WORKER_ID
                            w_id = person_data['worker_id']
                            current_ts = time.time()
                            last_vio = worker_cooldowns.get(w_id, 0)
                            
                            if current_ts - last_vio >= COOLDOWN_SECONDS:
                                save_violation(camera_id, class_name, detection['confidence'], bbox, w_id)
                                worker_cooldowns[w_id] = current_ts # Update cooldown
                                global_stats['no_helmet_count'] += 1
                                global_stats['total_violations'] += 1
                            else:
                                remaining = int(COOLDOWN_SECONDS - (current_ts - last_vio))
                                if remaining > 0:
                                    print(f"⏳ Cooldown active for {w_id} ({remaining}s remaining)")
                
                elif class_name == 'novest':
                    if person_data['violation_counters']['novest'] >= STABILIZER_THRESHOLD:
                        if not person_data['violations']['no_vest']:
                            person_data['violations']['no_vest'] = True
                            
                            # COOLDOWN CHECK PER WORKER_ID
                            w_id = person_data['worker_id']
                            current_ts = time.time()
                            last_vio = worker_cooldowns.get(w_id, 0)
                            
                            if current_ts - last_vio >= COOLDOWN_SECONDS:
                                save_violation(camera_id, class_name, detection['confidence'], bbox, w_id)
                                worker_cooldowns[w_id] = current_ts # Update cooldown
                                global_stats['no_vest_count'] += 1
                                global_stats['total_violations'] += 1
                            else:
                                remaining = int(COOLDOWN_SECONDS - (current_ts - last_vio))
                                if remaining > 0:
                                    print(f"⏳ Cooldown active for {w_id} ({remaining}s remaining)")
                
                # Update last seen time
                person_data['last_seen'] = current_time
        
        # Update FPS for this camera every 30 frames
        if frame_count % 30 == 0 and camera_id in camera_stats:
            elapsed = time.time() - start_time
            if elapsed > 0:
                camera_stats[camera_id]['fps'] = round(frame_count / elapsed, 1)
        
        # Clean up old persons (not seen for 30 seconds)
        if camera_id in tracked_persons:
            cleanup_time = current_time - 30
            persons_to_remove = []
            for pid, pdata in tracked_persons[camera_id].items():
                if pdata['last_seen'] < cleanup_time:
                    persons_to_remove.append(pid)
            
            for pid in persons_to_remove:
                if camera_id in tracked_persons and pid in tracked_persons[camera_id]:
                    del tracked_persons[camera_id][pid]
        
        time.sleep(0.03)

def save_violation(camera_id, violation_type, confidence, bbox, worker_id='Unknown'):
    """Save violation to database including local timestamp and worker_id"""
    from datetime import datetime
    local_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO violations (camera_id, violation_type, confidence, bbox, worker_id, timestamp) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (camera_id, violation_type, confidence, str(bbox), worker_id or 'Unknown', local_time))
    conn.commit()
    conn.close()

def generate_camera_feed(camera_id):
    """Generate video feed for a specific camera with enhanced RTSP support"""
    global cameras
    
    # IMPORTANT: keep it light.
    # Only stream frames for cameras that are actively started (exist in cameras dict).
    if camera_id not in cameras:
        import numpy as np
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Camera {camera_id} Inactive", (150, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(placeholder, "Click START to run", (180, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        _, buffer = cv2.imencode('.jpg', placeholder)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    # Stream frames from shared state
    last_frame_time = 0
    
    while camera_id in cameras:
        # Check if we have a frame for this camera
        current_frame_data = None
        
        with frame_lock:
            if camera_id in camera_frames:
                data = camera_frames[camera_id]
                # Only process if it's a new frame
                if data['timestamp'] > last_frame_time:
                    current_frame_data = data
                    last_frame_time = data['timestamp']
        
        if current_frame_data:
            frame = current_frame_data['frame'].copy()
            detections = current_frame_data['detections']
            
            # Draw detections on frame (using pre-computed detections)
            try:
                frame_with_detections = detector.draw_violations(frame, detections)
            except Exception as e:
                # print(f"⚠️ Draw error: {e}")
                frame_with_detections = frame
            
            # Encode frame
            try:
                _, buffer = cv2.imencode('.jpg', frame_with_detections)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"⚠️ Encode error: {e}")
                
        else:
            # No new frame yet, wait a bit
            time.sleep(0.01)
            continue
            
        # Control FPS of the stream slightly
        time.sleep(0.03)  # ~30 FPS cap for the stream
    
    # Generate placeholder frame
    import numpy as np
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, f"Camera {camera_id} Inactive", (150, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(placeholder, f"Feed Disconnected", (160, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    _, buffer = cv2.imencode('.jpg', placeholder)
    frame_bytes = buffer.tobytes()
    
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# HTML Templates
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APD Monitoring - Login</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Outfit:wght@400;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg-slate-950: #020617;
            --bg-slate-900: #0f172a;
            --text-slate-400: #94a3b8;
            --border-slate-700: #334155;
            --radius: 12px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-slate-950);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            overflow: hidden;
        }

        .login-card {
            background: var(--bg-slate-900);
            border: 1px solid var(--border-slate-700);
            width: 100%;
            max-width: 400px;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            position: relative;
            z-index: 10;
        }

        .login-header {
            text-align: center;
            margin-bottom: 32px;
        }

        .login-header h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .login-header p {
            color: var(--text-slate-400);
            font-size: 14px;
        }

        .form-group { margin-bottom: 24px; }
        .form-group label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-slate-400);
            margin-bottom: 8px;
        }

        .form-group input {
            width: 100%;
            padding: 12px 16px;
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid var(--border-slate-700);
            border-radius: var(--radius);
            color: #fff;
            font-size: 14px;
            transition: all 0.2s ease;
        }

        .form-group input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }

        .login-btn {
            width: 100%;
            padding: 12px;
            background: var(--primary);
            color: #fff;
            border: none;
            border-radius: var(--radius);
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-top: 8px;
        }

        .login-btn:hover {
            background: var(--primary-hover);
            transform: translateY(-1px);
        }

        .error-message {
            background: rgba(239, 68, 68, 0.1);
            color: #f87171;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(239, 68, 68, 0.2);
            margin-bottom: 24px;
            font-size: 13px;
            text-align: center;
        }

        /* Abstract Background */
        .bg-glow {
            position: absolute;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, rgba(99, 102, 241, 0) 70%);
            z-index: 1;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="bg-glow"></div>
    <div class="login-card">
        <div class="login-header">
            <h1>NYAWANG</h1>
            <p>Sistem Monitoring Pelanggaran APD</p>
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="error-message">Error: {{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <form method="POST" action="/login">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" placeholder="admin" required autofocus>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" placeholder="••••••••" required>
            </div>
            <button type="submit" class="login-btn">Sign In to Dashboard</button>
        </form>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APD Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Outfit:wght@400;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg-slate-950: #020617;
            --bg-slate-900: #0f172a;
            --bg-slate-800: #1e293b;
            --text-slate-50: #f8fafc;
            --text-slate-300: #cbd5e1;
            --text-slate-400: #94a3b8;
            --border-slate-700: #334155;
            --sidebar-width: 260px;
            --header-height: 72px;
            --radius: 12px;
            --radius-lg: 16px;
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(255, 255, 255, 0.05);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Inter', sans-serif;
            background: var(--bg-slate-950);
            color: var(--text-slate-50);
            line-height: 1.6;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* Modern Dashboard Layout */
        .app-wrapper {
            display: flex;
            min-height: 100vh;
        }

        /* Sidebar Styling */
        .sidebar {
            width: var(--sidebar-width);
            background: var(--bg-slate-900);
            border-right: 1px solid var(--border-slate-700);
            display: flex;
            flex-direction: column;
            position: fixed;
            height: 100vh;
            z-index: 100;
            transition: all 0.3s ease;
        }

        .sidebar-brand {
            height: var(--header-height);
            display: flex;
            align-items: center;
            padding: 0 24px;
            border-bottom: 1px solid var(--border-slate-700);
        }

        .sidebar-brand h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-menu {
            padding: 24px 16px;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .nav-item {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            border-radius: var(--radius);
            color: var(--text-slate-400);
            text-decoration: none;
            font-weight: 600;
            font-size: 15px;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            gap: 12px;
        }

        .nav-item:hover {
            background: rgba(255, 255, 255, 0.03);
            color: #fff;
        }

        .nav-item.active {
            background: rgba(99, 102, 241, 0.1);
            color: var(--primary);
            box-shadow: inset 0 0 0 1px rgba(99, 102, 241, 0.1);
        }

        .nav-label {
            font-size: 10px;
            font-weight: 800;
            color: var(--text-slate-400);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin: 24px 16px 8px;
            opacity: 0.5;
        }

        .nav-menu {
            padding: 16px;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        /* Main Content Styling */
        .main-container {
            flex-grow: 1;
            margin-left: var(--sidebar-width);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            height: var(--header-height);
            background: var(--bg-slate-900);
            border-bottom: 1px solid var(--border-slate-700);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 32px;
            position: sticky;
            top: 0;
            z-index: 90;
            backdrop-filter: blur(12px);
        }

        .header-title {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-slate-400);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .btn-action {
            padding: 8px 16px;
            border-radius: 8px;
            background: var(--bg-slate-800);
            border: 1px solid var(--border-slate-700);
            color: var(--text-slate-300);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
        }

        .btn-action:hover {
            background: var(--border-slate-700);
            color: #fff;
        }

        .btn-primary {
            background: var(--primary);
            border: none;
            color: #fff;
        }

        .btn-primary:hover {
            background: var(--primary-hover);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        /* Dashboard Grid & Cards */
        .content-body {
            padding: 32px;
            flex-grow: 1;
        }

        .card {
            background: var(--bg-slate-900);
            border: 1px solid var(--border-slate-700);
            border-radius: var(--radius-lg);
            padding: 24px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            overflow: hidden;
        }

        .card-header {
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-title {
            font-size: 18px;
            font-weight: 700;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Camera Grid Specifics */
        .camera-grid {
            display: grid;
            gap: 24px;
            margin-top: 24px;
        }

        .video-container {
            aspect-ratio: 16/9;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }

        .video-feed {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Circular Chart Styles */
        .flex-center-center {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .circular-chart {
            display: block;
            margin: 10px auto;
            max-width: 151px;
            max-height: 151px;
        }

        .circle-bg {
            fill: none;
            stroke: var(--bg-slate-800);
            stroke-width: 2.8;
        }

        .circle {
            fill: none;
            stroke-width: 2.8;
            stroke-linecap: round;
            transition: stroke-dashoffset 1s ease-in-out;
        }

        .chart-container {
            position: relative;
            width: 150px;
            height: 150px;
            margin: 0 auto;
        }

        .percentage {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-family: 'Outfit', sans-serif;
            font-size: 24px;
            font-weight: 800;
            color: var(--text-slate-50);
        }

        .status-badge {
            padding: 4px 10px;
            border-radius: 100px;
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: rgba(16, 185, 129, 0.1);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .status-badge.offline {
            background: rgba(239, 68, 68, 0.1);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        /* Stats Dashboard Styling */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }

        .stat-card {
            background: var(--bg-slate-900);
            border: 1px solid var(--border-slate-700);
            padding: 24px;
            border-radius: var(--radius);
        }

        .stat-label {
            color: var(--text-slate-400);
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .stat-value {
            font-family: 'Outfit', sans-serif;
            font-size: 38px;
            font-weight: 800;
            color: var(--text-slate-50);
        }

        /* Table Modernization */
        .table-container {
            background: var(--bg-slate-900);
            border: 1px solid var(--border-slate-700);
            border-radius: var(--radius-lg);
            overflow: hidden;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }

        th {
            background: rgba(255, 255, 255, 0.02);
            padding: 16px 24px;
            font-size: 13px;
            font-weight: 700;
            color: var(--text-slate-400);
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid var(--border-slate-700);
        }

        td {
            padding: 16px 24px;
            font-size: 14px;
            color: var(--text-slate-300);
            border-bottom: 1px solid var(--border-slate-700);
        }

        tr:last-child td { border-bottom: none; }

        tr:hover td { background: rgba(255, 255, 255, 0.01); }

        /* Tab Content Display Logic */
        .tab-content {
            display: none !important;
            animation: fadeIn 0.3s ease;
        }
        .tab-content.active {
            display: block !important;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Modal Refinement */
        .modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(2, 6, 23, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            z-index: 2000;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .modal.active {
            display: flex;
            opacity: 1;
        }
        .modal-content {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            width: 95%;
            max-width: 550px;
            padding: 32px;
            transform: scale(0.95);
            transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        .modal.active .modal-content {
            transform: scale(1);
        }

        /* Camera Registration Specific UI */
        #registration-video {
            border: 2px solid var(--primary);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
            border-radius: 16px;
        }
        #face-guide-box {
            border: 2px dashed rgba(255, 255, 255, 0.5);
            box-shadow: 0 0 0 5000px rgba(0, 0, 0, 0.4);
        }
        #capture-instruction {
            font-family: 'Outfit';
            font-size: 16px;
            letter-spacing: 0.5px;
            animation: pulse-ui 1.5s infinite;
        }
        @keyframes pulse-ui {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }

        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-slate-400);
            margin-bottom: 8px;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px 14px;
            background: var(--bg-slate-800);
            border: 1px solid var(--border-slate-700);
            border-radius: 8px;
            color: var(--text-slate-50);
            font-size: 14px;
            transition: all 0.2s ease;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        /* Sub-navigation for Workers */
        .sub-section { display: none; }
        .sub-section.active { display: block; }

        /* Modal Sidebar Submenu Styles */
        .nav-group {
            display: flex;
            flex-direction: column;
        }

        /* Modal Popup Styles */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(4px);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .modal-overlay.active {
            display: flex;
            opacity: 1;
        }
        .modal-content {
            background: var(--bg-slate-900);
            border: 1px solid var(--border-slate-700);
            border-radius: 16px;
            width: 90%;
            max-width: 450px;
            padding: 32px;
            transform: scale(0.9);
            transition: transform 0.3s ease;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        .modal-overlay.active .modal-content {
            transform: scale(1);
        }
        .modal-header {
            margin-bottom: 24px;
            text-align: center;
        }
        .modal-title {
            font-size: 20px;
            font-weight: 800;
            color: #fff;
            font-family: 'Outfit';
        }
        .modal-body {
            margin-bottom: 32px;
        }
        .modal-footer {
            display: flex;
            gap: 12px;
            justify-content: flex-end;
        }
        .submenu {
            display: none;
            padding-left: 36px;
            margin-top: 4px;
            flex-direction: column;
            gap: 4px;
        }
        .submenu.active {
            display: flex;
        }
        .submenu-item {
            padding: 8px 16px;
            border-radius: 8px;
            color: var(--text-slate-400);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
        }
        .submenu-item:hover {
            color: #fff;
            background: rgba(255, 255, 255, 0.03);
        }
        .submenu-item.active {
            color: var(--primary);
        }
        .nav-item .arrow {
            margin-left: auto;
            font-size: 10px;
            transition: transform 0.3s ease;
        }
        .nav-item.expanded .arrow {
            transform: rotate(180deg);
        }

        /* Modern Light Mode - Full Refinement */
        body.light-mode {
            --bg-slate-950: #f8fafc;
            --bg-slate-900: #ffffff;
            --bg-slate-800: #f1f5f9;
            --text-slate-50: #0f172a;
            --text-slate-400: #64748b;
            --text-slate-300: #334155;
            --border-slate-700: #e2e8f0;
            --glass-bg: rgba(255, 255, 255, 0.8);
        }
        body.light-mode .sidebar { background: #fff; border-right-color: #e2e8f0; }
        body.light-mode .sidebar-brand h1 { background: var(--primary); -webkit-background-clip: text; }
        body.light-mode .nav-item { color: #64748b; }
        body.light-mode .nav-item:hover { background: #f1f5f9; color: var(--primary); }
        body.light-mode .nav-item.active { background: rgba(99, 102, 241, 0.08); color: var(--primary); }
        body.light-mode .header { background: rgba(255, 255, 255, 0.8); border-bottom-color: #e2e8f0; }
        body.light-mode .card, body.light-mode .stat-card { 
            background: #fff; 
            border-color: #e2e8f0; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); 
        }
        body.light-mode .card-title, body.light-mode h3 { color: #0f172a; }
        body.light-mode .stat-value { color: #0f172a; }
        body.light-mode td { color: #334155; border-bottom-color: #f1f5f9; }
        body.light-mode th { background: #f8fafc; color: #64748b; border-bottom-color: #e2e8f0; }
        body.light-mode input, body.light-mode select { 
            background: #fff; 
            border-color: #cbd5e1; 
            color: #0f172a; 
        }
        body.light-mode .btn-action:not(.btn-primary) { 
            background: #fff; 
            border-color: #e2e8f0; 
            color: #475569; 
        }
        body.light-mode .btn-action:not(.btn-primary):hover { background: #f8fafc; border-color: #cbd5e1; }
        body.light-mode .modal-content { background: #fff; border-color: #e2e8f0; color: #0f172a; }
        body.light-mode .modal-title { color: #0f172a; }
        body.light-mode .circle-bg { stroke: #f1f5f9; }
        body.light-mode .percentage { color: #0f172a; }
        body.light-mode #face-guide-box { border-color: var(--primary); }
        body.light-mode .submenu-item:hover { background: #f1f5f9; color: var(--primary); }
        body.light-mode .submenu-item.active { color: var(--primary); }
    </style>
</head>
<body>
    <div class="app-wrapper">
        <!-- Sidebar Navigation -->
        <aside class="sidebar">
            <div class="sidebar-brand">
                <h1>NYAWANG</h1>
            </div>
            <nav class="nav-menu">
                <div class="nav-label">Monitoring</div>
                <div class="nav-group">
                    <a class="nav-item active" onclick="toggleSubmenu('cameras'); showTab('cameras')" id="nav-cameras">
                        <span class="icon">📹</span> Kamera <span class="arrow">▼</span>
                    </a>
                    <div id="submenu-cameras" class="submenu active">
                        <a class="submenu-item" onclick="changeViewModeSidebar('1', this)">1 Kamera</a>
                        <a class="submenu-item" onclick="changeViewModeSidebar('2', this)">2 Kamera</a>
                        <a class="submenu-item active" onclick="changeViewModeSidebar('4', this)">4 Kamera</a>
                        <a class="submenu-item" onclick="changeViewModeSidebar('8', this)">8 Kamera</a>
                        <a class="submenu-item" onclick="changeViewModeSidebar('16', this)">16 Kamera</a>
                        <a class="submenu-item" onclick="changeViewModeSidebar('all', this)">Semua Kamera</a>
                        {% if session.role == 'admin' %}
                        <div style="border-top: 1px solid var(--border-slate-700); margin-top: 8px; padding-top: 8px;">
                            <a class="submenu-item" onclick="openAddCameraModal()" style="color: var(--primary); font-weight: 700;">+ Tambah Kamera</a>
                        </div>
                        {% endif %}
                    </div>
                </div>
                
                <a class="nav-item" onclick="showTab('violations')" id="nav-violations">
                    <span class="icon">🚨</span> Pelanggaran
                </a>

                {% if session.role == 'admin' %}
                <div class="nav-label">Analisis</div>
                <a class="nav-item" onclick="showTab('statistics')" id="nav-statistics">
                    <span class="icon">📊</span> Statistik
                </a>

                <div class="nav-label">Management</div>
                <div class="nav-group">
                    <a class="nav-item" onclick="toggleSubmenu('workers')" id="nav-workers">
                        <span class="icon">👥</span> Pekerja <span class="arrow">▼</span>
                    </a>
                    <div id="submenu-workers" class="submenu">
                        <a class="submenu-item" onclick="switchToWorkerSection('list')" id="sub-list">Daftar Pekerja</a>
                        <a class="submenu-item" onclick="switchToWorkerSection('register')" id="sub-register">Registrasi Baru</a>
                        <a class="submenu-item" onclick="switchToWorkerSection('captures')" id="sub-captures">Dataset Capture</a>
                    </div>
                </div>
                <a class="nav-item" onclick="showTab('users')" id="nav-users">
                    <span class="icon">👤</span> Manajemen User
                </a>
                <a class="nav-item" onclick="showTab('settings')" id="nav-settings">
                    <span class="icon">⚙️</span> Pengaturan
                </a>
                {% endif %}
            </nav>
            <div style="padding: 24px; border-top: 1px solid var(--border-slate-700);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <div style="width: 32px; height: 32px; background: var(--primary); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 12px;">AD</div>
                    <div style="overflow: hidden;">
                        <p style="font-size: 13px; font-weight: 700; color: #fff; white-space: nowrap; text-overflow: ellipsis;">{{ session.username }}</p>
                        <p style="font-size: 11px; color: var(--text-slate-400);">{{ 'Administrator' if session.role == 'admin' else 'Petugas Lapangan' }}</p>
                    </div>
                </div>
                <a href="/logout" class="btn-action" style="justify-content: center; color: #f87171; border-color: rgba(239, 68, 68, 0.1); background: rgba(239, 68, 68, 0.05); width: 100%;">
                    <span>🚪</span> Sign Out
                </a>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="main-container">
            <header class="header">
                <div>
                    <h2 class="header-title" id="current-tab-title">Monitoring Kamera</h2>
                </div>
                <div class="header-actions">
                    <button class="btn-action" onclick="toggleLightMode()" id="theme-btn">
                        <span>💡</span> Mode Terang
                    </button>
                </div>
            </header>

            <div class="content-body">
                <!-- Cameras Tab -->
                <div id="cameras" class="tab-content active">
                    <div style="margin-bottom: 24px;">
                        <!-- Filter atau Label tambahan jika perlu di sini -->
                    </div>
                    <div class="camera-grid" id="camera-grid">
                        <!-- Camera cards loaded via JS -->
                    </div>
                </div>

                <!-- Statistics Tab -->
                <div id="statistics" class="tab-content">
                    <div style="background: var(--bg-slate-900); border: 1px solid var(--border-slate-700); border-radius: 12px; padding: 12px 20px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center;">
                        <!-- Left: Date Filter -->
                        <div style="display: flex; gap: 12px; align-items: center;">
                            <span style="font-size: 13px; font-weight: 700; color: var(--text-slate-400); text-transform: uppercase; letter-spacing: 0.5px;">📅 Periode Laporan:</span>
                            <input type="date" id="start-date" onchange="refreshStatistics()" class="btn-action" style="padding: 6px 12px; font-size: 13px;">
                            <span style="color: var(--text-slate-400); font-size: 12px;">s/d</span>
                            <input type="date" id="end-date" onchange="refreshStatistics()" class="btn-action" style="padding: 6px 12px; font-size: 13px;">
                        </div>
                        
                        <!-- Right: Danger Action -->
                        <button class="btn-action" onclick="resetData()" style="color: #f87171; border-color: rgba(239,68,68,0.2); font-weight: 700; background: rgba(239,68,68,0.05); padding: 8px 16px; font-size: 13px;">
                            <span>🗑️</span> Reset Data
                        </button>
                    </div>

                    <div class="stats-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 24px;">
                        <div class="stat-card">
                            <p class="stat-label">Total Pelanggaran</p>
                            <p class="stat-value" id="total-violations">0</p>
                        </div>
                        <div class="stat-card">
                            <p class="stat-label">Tanpa Helm</p>
                            <p class="stat-value" id="no-helmet-count" style="color: #f87171;">0</p>
                        </div>
                        <div class="stat-card">
                            <p class="stat-label">Tanpa Rompi</p>
                            <p class="stat-value" id="no-vest-count" style="color: #fbbf24;">0</p>
                        </div>
                    </div>

                    <div class="grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; margin-bottom: 24px;">
                        <div class="card flex-center-center" style="padding: 32px 0;">
                            <div>
                                <h4 class="card-title" style="margin-bottom: 20px; text-align: center;">Akurasi Deteksi APD</h4>
                                <div class="chart-container">
                                    <svg viewBox="0 0 36 36" class="circular-chart">
                                        <path class="circle-bg"
                                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                        />
                                        <path id="apd-circle" class="circle"
                                            stroke="#6366f1"
                                            stroke-dasharray="100, 100"
                                            stroke-dashoffset="100"
                                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                        />
                                    </svg>
                                    <div class="percentage" id="avg-apd-accuracy">0%</div>
                                </div>
                            </div>
                        </div>
                        <div class="card flex-center-center" style="padding: 32px 0;">
                            <div>
                                <h4 class="card-title" style="margin-bottom: 20px; text-align: center;">Akurasi Face Recognition</h4>
                                <div class="chart-container">
                                    <svg viewBox="0 0 36 36" class="circular-chart">
                                        <path class="circle-bg"
                                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                        />
                                        <path id="face-circle" class="circle"
                                            stroke="#10b981"
                                            stroke-dasharray="100, 100"
                                            stroke-dashoffset="100"
                                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                        />
                                    </svg>
                                    <div class="percentage" id="avg-face-accuracy">0%</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <h4 class="card-title">Tren Pelanggaran Harian</h4>
                            <p style="font-size: 12px; color: var(--text-slate-400);" id="last-update"></p>
                        </div>
                        <div style="height: 300px;">
                            <canvas id="daily-chart"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Violations Tab -->
                <div id="violations" class="tab-content">
                    <div class="card" style="padding: 0; overflow: hidden; border: 1px solid var(--border-slate-700);">
                        <div style="padding: 16px 24px; border-bottom: 1px solid var(--border-slate-700); display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.01);">
                            <!-- Left: Date Filter -->
                            <div style="display: flex; gap: 12px; align-items: center;">
                                <div style="display: flex; align-items: center; gap: 8px; color: var(--text-slate-400); font-size: 14px; font-weight: 600;">
                                    <span>📅</span> Filter Tanggal:
                                </div>
                                <input type="date" id="violation-start-date" onchange="loadViolations()" class="btn-action" style="padding: 6px 12px; font-size: 13px;">
                                <span style="color: var(--text-slate-400); font-size: 12px;">sampai</span>
                                <input type="date" id="violation-end-date" onchange="loadViolations()" class="btn-action" style="padding: 6px 12px; font-size: 13px;">
                            </div>
                            
                            <!-- Right: Export Actions -->
                            <div style="display: flex; gap: 10px;">
                                <button class="btn-action" onclick="exportData('pdf')" style="padding: 8px 16px; font-size: 13px; font-weight: 700; border-color: rgba(239, 68, 68, 0.2); color: #f87171;">
                                    <span>📄</span> PDF
                                </button>
                                <button class="btn-action" onclick="exportData('excel')" style="padding: 8px 16px; font-size: 13px; font-weight: 700; border-color: rgba(16, 185, 129, 0.2); color: #10b981;">
                                    <span>📊</span> Excel
                                </button>
                            </div>
                        </div>
                        <div class="table-container" style="border: none; border-radius: 0;">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Jam</th>
                                        <th>Tanggal</th>
                                        <th>Lokasi Kamera</th>
                                        <th>Jenis Pelanggaran</th>
                                        <th>Identitas Pekerja</th>
                                        <th>Skor Akurasi</th>
                                    </tr>
                                </thead>
                                <tbody id="violations-tbody">
                                    <!-- Loaded via JS -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Workers Tab -->
                <div id="workers" class="tab-content">
                    
                    <div id="worker-list-section" class="sub-section active">
                        <div class="table-container">
                            <table>
                                <thead>
                                    <tr>
                                        <th>ID</th>
                                        <th>Nama Lengkap</th>
                                        <th>Sampel Wajah</th>
                                        <th>Tgl Registrasi</th>
                                        <th>Aksi</th>
                                    </tr>
                                </thead>
                                <tbody id="workers-tbody"></tbody>
                            </table>
                        </div>
                    </div>

                    <div id="worker-register-section" class="sub-section">
                        <div class="card" style="max-width: 600px; margin: 0 auto;">
                            <h4 style="margin-bottom: 24px;">Registrasi Wajah Baru</h4>
                            <form id="worker-form" onsubmit="registerWorker(event)">
                                <div class="form-group">
                                    <label>Worker ID</label>
                                    <input type="text" id="worker-id" required placeholder="W-001">
                                </div>
                                <div class="form-group">
                                    <label>Nama Lengkap</label>
                                    <input type="text" id="worker-name" required placeholder="Nama Lengkap">
                                </div>
                                <div class="form-group">
                                    <label>Upload Sampel Wajah (Wajib 10 Foto)</label>
                                    <div style="border: 2px dashed var(--border-slate-700); border-radius: 12px; padding: 32px; text-align: center;">
                                        <input type="file" id="worker-images" multiple accept="image/*" required>
                                        <p style="font-size: 12px; color: var(--text-slate-400); margin-top: 12px;">Klik untuk memilih atau seret file ke sini</p>
                                    </div>
                                </div>
                                <div style="display: flex; gap: 12px; margin-top: 12px;">
                                    <button type="button" class="btn-action" onclick="openCameraRegistration()" style="flex: 1; justify-content: center; padding: 12px; background: var(--bg-slate-800); border: 1px solid var(--primary);">
                                        📸 Daftar via Kamera
                                    </button>
                                    <button type="submit" class="btn-action btn-primary" id="worker-submit-btn" style="flex: 1; justify-content: center; padding: 12px;">
                                        📁 Daftar via Upload
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>

                    <div id="worker-captures-section" class="sub-section">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
                            <p style="color: var(--text-slate-400); font-size: 14px;">Capture wajah pelanggar yang tidak terdaftar. Pilih folder untuk didaftarkan sebagai pekerja.</p>
                            <button class="btn-action" onclick="loadCaptures()">🔄 Refresh</button>
                        </div>
                        <div id="captures-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px;">
                            <!-- Captures loaded via JS -->
                        </div>
                    </div>
                </div>

                {% if session.role == 'admin' %}
                <!-- Users Management Tab -->
                <div id="users" class="tab-content">
                    <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 24px;">
                        <button class="btn-action btn-primary" onclick="openUserModal()">+ Tambah Akun</button>
                    </div>
                    
                    <div class="card" style="padding: 0; overflow: hidden; border: 1px solid var(--border-slate-700);">
                        <div class="table-container" style="border: none; border-radius: 0;">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Username</th>
                                        <th>Role</th>
                                        <th>Tanggal Dibuat</th>
                                        <th>Aksi</th>
                                    </tr>
                                </thead>
                                <tbody id="users-tbody">
                                    <!-- Loaded via JS -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <div id="settings" class="tab-content">
                    <div class="card" style="max-width: 450px; background: var(--bg-slate-900); border: 1px solid var(--border-slate-700);">
                        <div class="form-group" style="margin-bottom: 24px;">
                            <label style="font-weight: 600; color: var(--text-slate-300);">Cooldown Deteksi</label>
                            <div style="display: flex; align-items: center; gap: 10px; margin-top: 8px;">
                                <input type="number" id="cooldown-setting" value="5" style="width: 80px; padding: 8px;">
                                <span style="font-size: 14px; color: var(--text-slate-400);">detik</span>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label style="font-weight: 600; color: var(--text-slate-300); display: block; margin-bottom: 12px;">
                                Treshold Confidence: <span id="confidence-value" style="color: var(--primary);">0.40</span>
                            </label>
                            <input type="range" id="confidence-setting" min="0.40" max="0.90" step="0.01" value="0.40" 
                                   style="width: 100%; cursor: pointer;"
                                   oninput="document.getElementById('confidence-value').textContent = parseFloat(this.value).toFixed(2)">
                            <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-slate-400); margin-top: 8px;">
                                <span>0.40</span>
                                <span>0.90</span>
                            </div>
                        </div>
                        
                        <button class="btn-action btn-primary" onclick="saveSettings()" style="width: 100%; justify-content: center; padding: 10px; margin-top: 20px; border-radius: 8px; font-weight: 600;">
                            Simpan Pengaturan
                        </button>
                    </div>
                </div>
                {% endif %}
            </div>
        </main>
    </div>

    <!-- Modals -->
    <div id="camera-modal" class="modal">
        <div class="modal-content">
            <h3 id="modal-title" style="margin-bottom: 24px; font-family: 'Outfit'; font-weight: 800;">Konfigurasi Kamera</h3>
            <div class="form-group">
                <label>Nama Kamera</label>
                <input type="text" id="camera-name" placeholder="Pintu Utama">
            </div>
            <div class="form-group">
                <label>Sumber Video</label>
                <select id="camera-source" onchange="handleSourceChange()">
                    <option value="">-- Pilih Sumber --</option>
                    <option value="0">Webcam Internal</option>
                    <option value="1">Webcam Eksternal</option>
                    <option value="rtsp">CCTV (RTSP Stream)</option>
                    <option value="file">File Video (MP4)</option>
                </select>
            </div>
            <div id="rtsp-group" style="display:none;" class="form-group">
                <label>RTSP URL</label>
                <input type="text" id="rtsp-url" placeholder="rtsp://admin:pass@ip:port/stream">
            </div>
            <div id="file-group" style="display:none;" class="form-group">
                <label>Path Video</label>
                <input type="text" id="file-path" placeholder="C:/videos/footage.mp4">
            </div>
            <div style="display: flex; gap: 12px; margin-top: 32px;">
                <button class="btn-action" style="flex: 1; justify-content: center;" onclick="closeCameraModal()">Batal</button>
                <button class="btn-action btn-primary" style="flex: 1; justify-content: center;" onclick="addCamera()">Simpan</button>
            </div>
        </div>
    </div>

    </div>

    <!-- Registration Camera Modal -->
    <div id="registration-camera-modal" class="modal">
        <div class="modal-content" style="max-width: 700px; padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h3 style="font-family: 'Outfit'; font-weight: 800;">Pengambilan Sampel Wajah Otomatis</h3>
                <span id="capture-count" class="badge badge-primary">0 / 10 Foto</span>
            </div>
            
            <div style="position: relative; border-radius: 12px; overflow: hidden; background: #000; aspect-ratio: 16/9; margin-bottom: 20px;">
                <video id="registration-video" autoplay playsinline style="width: 100%; height: 100%; object-fit: cover;"></video>
                <canvas id="registration-canvas" style="display: none;"></canvas>
                
                <!-- Overlay Panduan -->
                <div id="capture-overlay" style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(0,0,0,0.4); pointer-events: none;">
                    <div id="face-guide-box" style="width: 250px; height: 300px; border: 2px dashed #fff; border-radius: 100px; transition: all 0.3s;"></div>
                    <p id="capture-instruction" style="margin-top: 20px; color: #fff; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.8); background: var(--primary); padding: 8px 20px; border-radius: 20px;">Mempersiapkan Kamera...</p>
                </div>
            </div>

            <!-- Progress Bar -->
            <div style="height: 6px; width: 100%; background: var(--bg-slate-800); border-radius: 3px; margin-bottom: 24px; overflow: hidden;">
                <div id="capture-progress" style="height: 100%; width: 0%; background: var(--primary); transition: width 0.3s;"></div>
            </div>

            <div style="display: flex; gap: 12px;">
                <button id="stop-capture-btn" class="btn-action" style="flex: 1; justify-content: center;" onclick="closeCameraRegistration()">Batal</button>
                <button id="start-capture-btn" class="btn-action btn-primary" style="flex: 2; justify-content: center;" onclick="startGuidedCapture()">Mulai Pengambilan Foto</button>
            </div>
        </div>
    </div>

    <!-- Export Filter Modal -->
    <div id="export-modal" class="modal-overlay">
        <div class="modal-content">
            <div class="modal-header">
                <h4 class="modal-title">Filter Laporan</h4>
                <p style="font-size: 13px; color: var(--text-slate-400); margin-top: 8px;">Pilih rentang tanggal untuk data yang ingin di-export.</p>
            </div>
            <div class="modal-body">
                <div class="form-group">
                    <label>Dari Tanggal</label>
                    <input type="date" id="export-start-date" class="btn-action" style="width: 100%; padding: 12px; margin-top: 8px;">
                </div>
                <div class="form-group" style="margin-top: 20px;">
                    <label>Sampai Tanggal</label>
                    <input type="date" id="export-end-date" class="btn-action" style="width: 100%; padding: 12px; margin-top: 8px;">
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn-action" style="flex: 1; justify-content: center;" onclick="closeExportModal()">Batal</button>
                <button class="btn-action btn-primary" style="flex: 1; justify-content: center;" onclick="confirmExport()">Download Laporan</button>
            </div>
        </div>
    </div>

    <!-- Add User Modal -->
    <div id="user-modal" class="modal">
        <div class="modal-content" style="max-width: 450px;">
            <div class="modal-header">
                <h3 style="font-family: 'Outfit'; font-weight: 800;">Tambah Akun Baru</h3>
            </div>
            <div class="form-group" style="margin-top: 20px;">
                <label>Username</label>
                <input type="text" id="new-user-username" placeholder="Masukkan username" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" id="new-user-password" placeholder="Masukkan password" required>
            </div>
            <div class="form-group">
                <label>Role</label>
                <select id="new-user-role" style="width: 100%; padding: 12px; border-radius: 8px; background: var(--bg-slate-800); border: 1px solid var(--border-slate-700); color: #fff;">
                    <option value="petugas">Petugas Lapangan</option>
                    <option value="admin">Administrator</option>
                </select>
            </div>
            <div style="display: flex; gap: 12px; margin-top: 32px;">
                <button class="btn-action" style="flex: 1; justify-content: center;" onclick="closeUserModal()">Batal</button>
                <button class="btn-action btn-primary" style="flex: 1; justify-content: center;" onclick="addUser()">Simpan User</button>
            </div>
        </div>
    </div>

    <script>
        // Set tema awal sebelum halaman loading selesai agar tidak berkedip
        if(localStorage.getItem('theme') === 'light') {
            document.body.classList.add('light-mode');
        }

        function toggleLightMode() {
            document.body.classList.toggle('light-mode');
            const btn = document.getElementById('theme-btn');
            const isLight = document.body.classList.contains('light-mode');
            
            if(isLight) {
                btn.innerHTML = '<span>🌙</span> Mode Gelap';
                localStorage.setItem('theme', 'light');
            } else {
                btn.innerHTML = '<span>💡</span> Mode Terang';
                localStorage.setItem('theme', 'dark');
            }
            
            // Re-draw chart to update colors if it exists
            if (dailyChartInstance) {
                updateDailyStats();
            }
        }
        
        // Sesuaikan tulisan tombol saat pertama dimuat
        document.addEventListener('DOMContentLoaded', () => {
            const btn = document.getElementById('theme-btn');
            if(document.body.classList.contains('light-mode') && btn) {
                btn.innerText = '🌙 Gelap';
            }
        });
        let currentTab = 'cameras';
        let editingCameraId = null;
        let currentViewMode = 4;  // 1,2,4,8,16 cams
        
        function changeViewModeSidebar(mode, element) {
            // Update current global state
            currentViewMode = (mode === 'all') ? 'all' : parseInt(mode, 10);
            
            // Update UI Sidebar Active state
            const submenu = document.getElementById('submenu-cameras');
            if (submenu) {
                submenu.querySelectorAll('.submenu-item').forEach(item => item.classList.remove('active'));
            }
            if (element) element.classList.add('active');
            
            // Refresh cameras with new mode
            loadCameras();
            localStorage.setItem('viewMode', mode);
        }

        function changeViewMode() {
            // Keep for compatibility if needed elsewhere
            const val = document.getElementById('view-mode')?.value || '4';
            changeViewModeSidebar(val);
        }

        function showTab(tabName) {
            // Updated for new .nav-item structure
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            
            document.getElementById(tabName).classList.add('active');
            document.getElementById('nav-' + tabName).classList.add('active');
            
            const titles = {
                'cameras': 'Monitoring Kamera',
                'statistics': 'Analisis Statistik',
                'violations': 'Log Pelanggaran',
                'workers': 'Manajemen Pekerja',
                'settings': 'Pengaturan Sistem'
            };
            document.getElementById('current-tab-title').textContent = titles[tabName] || 'Dashboard';
            
            currentTab = tabName;
            
            if (tabName === 'cameras') loadCameras();
            else if (tabName === 'statistics') loadStatistics();
            else if (tabName === 'violations') loadViolations();
            else if (tabName === 'workers') showWorkerSection('list');
            else if (tabName === 'users') loadUsers();
            else if (tabName === 'settings') loadCurrentSettings();
        }

        // --- User Management ---
        function openUserModal() {
            document.getElementById('user-modal').style.display = 'flex';
        }
        function closeUserModal() {
            document.getElementById('user-modal').style.display = 'none';
        }
        function loadUsers() {
            fetch('/api/users')
                .then(r => r.json())
                .then(users => {
                    const tbody = document.getElementById('users-tbody');
                    tbody.innerHTML = users.map(u => `
                        <tr>
                            <td style="font-weight: 700;">${u.username}</td>
                            <td><span class="badge ${u.role === 'admin' ? 'badge-primary' : ''}" style="background: ${u.role === 'admin' ? 'rgba(99,102,241,0.2)' : 'rgba(100,116,139,0.2)'}; color: ${u.role === 'admin' ? '#a5b4fc' : '#94a3b8'};">${u.role.toUpperCase()}</span></td>
                            <td style="color: var(--text-slate-400); font-size: 12px;">${u.created_at}</td>
                            <td>
                                ${u.username !== 'admin' ? `
                                    <button class="btn-action" style="padding: 4px 8px; color: #f87171; border-color: rgba(239,68,68,0.1);" onclick="deleteUser('${u.username}')">Hapus</button>
                                ` : '<span style="font-size: 11px; color: #555;">Sistem</span>'}
                            </td>
                        </tr>
                    `).join('');
                });
        }
        function addUser() {
            const username = document.getElementById('new-user-username').value;
            const password = document.getElementById('new-user-password').value;
            const role = document.getElementById('new-user-role').value;
            
            if (!username || !password) return alert('Lengkapi data!');
            
            fetch('/api/users', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username, password, role})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    closeUserModal();
                    loadUsers();
                } else {
                    alert('Gagal: ' + data.message);
                }
            });
        }
        function deleteUser(username) {
            if (!confirm(`Hapus akun ${username}?`)) return;
            fetch(`/api/users/${username}`, { method: 'DELETE' })
                .then(r => r.json())
                .then(data => {
                    if (data.success) loadUsers();
                    else alert('Gagal: ' + data.message);
                });
        }
        
        function toggleSubmenu(id) {
            const submenu = document.getElementById('submenu-' + id);
            const navItem = document.getElementById('nav-' + id);
            const isExpanding = !submenu.classList.contains('active');
            
            // Close other submenus first (optional, but cleaner)
            document.querySelectorAll('.submenu').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('expanded'));
            
            if (isExpanding) {
                submenu.classList.add('active');
                navItem.classList.add('expanded');
                showTab(id); // Switch to the main tab when expanding
            }
        }

        function switchToWorkerSection(section) {
            showTab('workers');
            showWorkerSection(section);
            // Ensure submenu is expanded
            document.getElementById('submenu-workers').classList.add('active');
            document.getElementById('nav-workers').classList.add('expanded');
        }
        
        function showWorkerSection(section) {
            // Hide all sub-sections and clear submenu items
            document.querySelectorAll('.sub-section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.submenu-item').forEach(s => s.classList.remove('active'));
            
            if (section === 'list') {
                document.getElementById('worker-list-section').classList.add('active');
                document.getElementById('sub-list').classList.add('active');
                loadWorkers();
            } else if (section === 'register') {
                document.getElementById('worker-register-section').classList.add('active');
                document.getElementById('sub-register').classList.add('active');
            } else if (section === 'captures') {
                document.getElementById('worker-captures-section').classList.add('active');
                document.getElementById('sub-captures').classList.add('active');
                loadCaptures();
            }
        }
        
        function loadCameras() {
            const grid = document.getElementById('camera-grid');
            if (!grid) {
                console.error('Camera grid element not found!');
                return;
            }
            
            // Show loading state
            grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: #888;">Loading cameras...</div>';
            
            fetch('/api/cameras')
                .then(r => {
                    if (!r.ok) {
                        throw new Error(`HTTP error! status: ${r.status}`);
                    }
                    return r.json();
                })
                .then(data => {
                    grid.innerHTML = '';

                    // Check if data and cameras exist
                    if (!data || !data.cameras || data.cameras.length === 0) {
                        grid.innerHTML = `
                            <div style="grid-column: 1/-1; text-align: center; padding: 60px; color: #888; background: #111; border: 2px solid #333; border-radius: 8px;">
                                <div style="font-size: 24px; margin-bottom: 20px;">📹</div>
                                <div style="font-size: 18px; margin-bottom: 10px; color: var(--text-slate-50);">No Cameras Added Yet</div>
                                <div style="font-size: 14px; color: #666;">Add a camera using the form above</div>
                            </div>
                        `;
                        return;
                    }

                    // Set grid columns based on view mode
                    let cols = 1;
                    if (currentViewMode === 1) cols = 1;
                    else if (currentViewMode === 2) cols = 2;
                    else if (currentViewMode === 4) cols = 2;  // 2x2 layout
                    else if (currentViewMode === 8) cols = 4;  // 4x2 layout (2 rows x 4 cols)
                    else if (currentViewMode === 16) cols = 4; // 4x4 layout
                    else if (currentViewMode === 'all') {
                        // Adaptive grid: 2 cols for up to 4, 3 cols for up to 9, 4 cols for more
                        const count = data.cameras.length;
                        if (count <= 1) cols = 1;
                        else if (count <= 4) cols = 2;
                        else if (count <= 9) cols = 3;
                        else cols = 4;
                    }
                    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

                    // Add body class for CSS per-mode styling
                    document.body.classList.remove('view-mode-1','view-mode-2','view-mode-4','view-mode-8','view-mode-16','view-mode-all');
                    document.body.classList.add(`view-mode-${currentViewMode}`);

                    // Select cameras to display based on view mode
                    const camerasToShow = (currentViewMode === 'all') ? data.cameras : data.cameras.slice(0, currentViewMode);
                    
                    if (camerasToShow.length === 0) {
                        grid.innerHTML = `
                            <div style="grid-column: 1/-1; text-align: center; padding: 60px; color: #888; background: #111; border: 2px solid #333; border-radius: 8px;">
                                <div style="font-size: 24px; margin-bottom: 20px;">📹</div>
                                <div style="font-size: 18px; margin-bottom: 10px; color: var(--text-slate-50);">No Cameras to Display</div>
                                <div style="font-size: 14px; color: #666;">Add cameras or adjust view mode</div>
                            </div>
                        `;
                        return;
                    }
                    
                    camerasToShow.forEach(camera => {
                        const card = createCameraCard(camera);
                        grid.innerHTML += card;
                    });
                })
                .catch(error => {
                    console.error('Error loading cameras:', error);
                    grid.innerHTML = `
                        <div style="grid-column: 1/-1; text-align: center; padding: 60px; color: #ff0000; background: #111; border: 2px solid #ff0000; border-radius: 8px;">
                            <div style="font-size: 24px; margin-bottom: 20px;">❌</div>
                            <div style="font-size: 18px; margin-bottom: 10px; color: var(--text-slate-50);">Error Loading Cameras</div>
                            <div style="font-size: 14px; color: #666;">${error.message || 'Unknown error'}</div>
                            <button onclick="loadCameras()" style="margin-top: 20px; padding: 10px 20px; background: #000; color: #fff; border: 2px solid #fff; cursor: pointer;">Retry</button>
                        </div>
                    `;
                });
        }
        
        function createCameraCard(camera) {
            const statusClass = camera.status === 'active' ? '' : 'offline';
            const statusLabel = camera.status === 'active' ? 'ONLINE' : 'OFFLINE';
            
            return `
                <div class="card" style="padding: 16px;">
                    <div class="card-header" style="margin-bottom: 12px;">
                        <h4 class="card-title" style="font-size: 14px;">
                            ${camera.name.replace(/</g, '&lt;').replace(/>/g, '&gt;')}
                        </h4>
                        <span class="status-badge ${statusClass}">${statusLabel}</span>
                    </div>
                    <div class="video-container">
                        <img class="video-feed" src="/camera_feed/${camera.id}?t=${new Date().getTime()}" alt="${camera.name}">
                    </div>
                    <div style="margin-top:12px; display:flex; flex-direction:column; gap:8px;">
                        <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-slate-400);">
                            <span>ID: CAM-${camera.id}</span>
                            <span style="max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">Source: ${camera.source}</span>
                        </div>
                        <div style="display:flex; gap:8px; margin-top:4px;">
                            <button class="btn-action ${camera.status === 'active' ? '' : 'btn-primary'}" style="flex:1; justify-content:center; padding:6px;" data-action="start" data-camera-id="${camera.id}">
                                Start
                            </button>
                            <button class="btn-action" style="flex:1; justify-content:center; padding:6px;" data-action="stop" data-camera-id="${camera.id}">
                                Stop
                            </button>
                            <button class="btn-action" style="padding:6px;" data-action="edit" data-camera-id="${camera.id}" data-camera-name="${camera.name.replace(/"/g, '&quot;')}" data-camera-source="${camera.source.replace(/"/g, '&quot;')}">
                                ⚙️
                            </button>
                            <button class="btn-action" style="padding:6px; color:#f87171; border-color:rgba(239,68,68,0.1);" data-action="delete" data-camera-id="${camera.id}">
                                🗑️
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Delegated event listener for camera buttons
        document.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-action]');
            if (!btn) return;
            
            const action = btn.dataset.action;
            const cameraId = parseInt(btn.dataset.cameraId, 10);
            
            if (action === 'start') {
                startCamera(cameraId);
            } else if (action === 'stop') {
                stopCamera(cameraId);
            } else if (action === 'edit') {
                editCamera(cameraId, btn.dataset.cameraName, btn.dataset.cameraSource);
            } else if (action === 'delete') {
                deleteCamera(cameraId);
            }
        });
        
        // Auto-refresh timer (10 seconds)
        let refreshInterval = setInterval(() => {
            if (currentTab === 'statistics') {
                loadStatistics();
            } else if (currentTab === 'violations') {
                loadViolations();
            } else if (currentTab === 'cameras') {
                 // Check if camera feeds need any periodic updates (but they are MJPEG, so no)
            }
        }, 10000);

        function resetData() {
            if (!confirm('AWAS! Apakah Anda yakin ingin menghapus semua data statistik dan log pelanggaran? Tindakan ini tidak dapat dibatalkan.')) return;
            
            fetch('/api/violations/reset', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert('Data berhasil direset.');
                        loadStatistics();
                        if (currentTab === 'violations') loadViolations();
                    } else {
                        alert('Gagal mereset data: ' + data.message);
                    }
                })
                .catch(err => alert('Error mereset data.'));
        }
        
        function refreshStatistics() {
            loadStatistics();
            updateDailyStats();
        }

        function loadStatistics() {
            const start = document.getElementById('start-date').value;
            const end = document.getElementById('end-date').value;
            
            fetch(`/api/statistics?start=${start}&end=${end}`)
                .then(r => r.json())
                .then(data => {
                    document.getElementById('total-violations').textContent = data.total_violations;
                    document.getElementById('no-helmet-count').textContent = data.no_helmet_count;
                    document.getElementById('no-vest-count').textContent = data.no_vest_count;
                    
                    const apdVal  = typeof data.avg_apd_accuracy  === 'number' ? data.avg_apd_accuracy  : 0;
                    const faceVal = typeof data.avg_face_accuracy  === 'number' ? data.avg_face_accuracy : 0;
                    
                    document.getElementById('avg-apd-accuracy').textContent  = apdVal.toFixed(1)  + '%';
                    document.getElementById('avg-face-accuracy').textContent = faceVal.toFixed(1) + '%';
                    
                    // Update Circular Progress
                    const apdOffset = 100 - Math.min(apdVal, 100);
                    const faceOffset = 100 - Math.min(faceVal, 100);
                    
                    document.getElementById('apd-circle').style.strokeDashoffset = apdOffset;
                    document.getElementById('face-circle').style.strokeDashoffset = faceOffset;
                    
                    const now = new Date();
                    document.getElementById('last-update').textContent = 'Terakhir Diupdate: ' + now.toLocaleTimeString();
                });
        }
        
        
        
        function updateDailyStats() {
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            
            fetch(`/api/daily_stats?start=${startDate}&end=${endDate}`)
                .then(r => r.json())
                .then(data => {
                    drawChart(data);
                });
        }
        
        let dailyChartInstance = null;

        function drawChart(data) {
            const ctx = document.getElementById('daily-chart').getContext('2d');
            const dates = Object.keys(data);
            const helmetCounts = dates.map(date => data[date].no_helmet || 0);
            const vestCounts = dates.map(date => data[date].no_vest || 0);
            
            // Re-use instance for smooth animations if possible
            if (dailyChartInstance && JSON.stringify(dailyChartInstance.data.labels) === JSON.stringify(dates)) {
                dailyChartInstance.data.datasets[0].data = helmetCounts;
                dailyChartInstance.data.datasets[1].data = vestCounts;
                dailyChartInstance.update('none'); // Update without full re-animation to be subtle
                return;
            }

            if (dailyChartInstance) dailyChartInstance.destroy();
            
            // Create nice gradients for the area fill
            const cyanGradient = ctx.createLinearGradient(0, 0, 0, 200);
            cyanGradient.addColorStop(0, 'rgba(0, 229, 255, 0.15)');
            cyanGradient.addColorStop(1, 'rgba(0, 229, 255, 0)');
            
            const greyGradient = ctx.createLinearGradient(0, 0, 0, 200);
            greyGradient.addColorStop(0, 'rgba(136, 136, 136, 0.1)');
            greyGradient.addColorStop(1, 'rgba(136, 136, 136, 0)');

            const isLight = document.body.classList.contains('light-mode');
            const gridColor = isLight ? 'rgba(0, 0, 0, 0.05)' : 'rgba(255, 255, 255, 0.05)';
            const textColor = isLight ? '#64748b' : '#94a3b8';

            dailyChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates.map(d => {
                        const date = new Date(d);
                        return date.toLocaleDateString('id-ID', { day: 'numeric', month: 'short' });
                    }),
                    datasets: [
                        {
                            label: 'Tanpa Helm',
                            data: helmetCounts,
                            borderColor: '#6366f1',
                            backgroundColor: cyanGradient,
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 4,
                            pointBackgroundColor: '#6366f1',
                            pointBorderColor: isLight ? '#fff' : '#000',
                            pointBorderWidth: 2,
                            pointHoverRadius: 6
                        },
                        {
                            label: 'Tanpa Rompi',
                            data: vestCounts,
                            borderColor: isLight ? '#94a3b8' : '#888',
                            backgroundColor: greyGradient,
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 4,
                            pointBackgroundColor: isLight ? '#94a3b8' : '#888',
                            pointBorderColor: isLight ? '#fff' : '#000',
                            pointBorderWidth: 2,
                            pointHoverRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { 
                        legend: { 
                            position: 'top',
                            align: 'end',
                            labels: { color: textColor, boxWidth: 12, font: { size: 10, weight: 'bold' }, padding: 20 } 
                        },
                        tooltip: {
                            backgroundColor: isLight ? 'rgba(255,255,255,0.95)' : 'rgba(0,0,0,0.85)',
                            titleColor: isLight ? '#0f172a' : '#fff',
                            bodyColor: isLight ? '#475569' : '#aaa',
                            borderColor: isLight ? '#e2e8f0' : '#333',
                            borderWidth: 1,
                            padding: 12,
                            displayColors: true,
                            cornerRadius: 4,
                            callbacks: {
                                title: (tooltipItems) => {
                                    const d = dates[tooltipItems[0].dataIndex];
                                    return new Date(d).toLocaleDateString('id-ID', { 
                                        weekday: 'long', 
                                        day: 'numeric', 
                                        month: 'long', 
                                        year: 'numeric' 
                                    });
                                }
                            }
                        }
                    },
                    scales: {
                        x: { 
                            grid: { display: false }, 
                            ticks: { 
                                color: textColor, 
                                font: { size: 10, weight: 'bold' },
                                maxRotation: 0,
                                autoSkip: true
                            } 
                        },
                        y: { 
                            beginAtZero: true, 
                            grid: { color: gridColor, drawBorder: false }, 
                            ticks: { 
                                color: textColor, 
                                font: { size: 10, family: 'Inter' },
                                precision: 0
                            } 
                        }
                    },
                    interaction: {
                        mode: 'index',
                        intersect: false
                    }
                }
            });
        }
        
        function loadViolations() {
            const startDate = document.getElementById('violation-start-date').value;
            const endDate = document.getElementById('violation-end-date').value;
            
            let url = '/api/violations';
            if (startDate && endDate) {
                url += `?start=${startDate}&end=${endDate}`;
            }
            
            fetch(url)
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('violations-tbody');
                    tbody.innerHTML = '';
                    
                    data.violations.forEach(violation => {
                        const row = document.createElement('tr');
                        const date = new Date(violation.timestamp);
                        const workerId = violation.worker_id && violation.worker_id !== 'Unknown' 
                            ? violation.worker_id 
                            : 'Unknown ID';
                        
                        row.innerHTML = `
                            <td style="font-weight:600;">${date.toLocaleTimeString()}</td>
                            <td style="color:var(--text-slate-400);">${date.toLocaleDateString()}</td>
                            <td>${violation.camera_name || 'Camera ' + violation.camera_id}</td>
                            <td><span class="violation-badge ${violation.violation_type}">${violation.violation_type.replace('no', 'No ').toUpperCase()}</span></td>
                            <td><span style="font-weight:700; color:${workerId === 'Unknown ID' ? '#ff6d00' : '#00e5ff'};">${workerId}</span></td>
                            <td style="font-family:monospace; font-weight:600;">${(violation.confidence * 100).toFixed(1)}%</td>
                        `;
                        tbody.appendChild(row);
                    });
                })
                .catch(err => console.error('Error loading violations:', err));
        }
        
        let pendingExportFormat = null;

        function exportData(format) {
            pendingExportFormat = format;
            document.getElementById('export-modal').classList.add('active');
            
            // Set default dates to today if empty
            const today = new Date().toISOString().split('T')[0];
            if (!document.getElementById('export-start-date').value) {
                document.getElementById('export-start-date').value = today;
            }
            if (!document.getElementById('export-end-date').value) {
                document.getElementById('export-end-date').value = today;
            }
        }

        function closeExportModal() {
            document.getElementById('export-modal').classList.remove('active');
            pendingExportFormat = null;
        }

        function confirmExport() {
            const start = document.getElementById('export-start-date').value;
            const end = document.getElementById('export-end-date').value;
            
            if (!start || !end) {
                alert('Pilih rentang tanggal terlebih dahulu!');
                return;
            }
            
            let url = `/api/export?format=${pendingExportFormat}&start=${start}&end=${end}`;
            window.open(url);
            closeExportModal();
        }
        
        function loadWorkers() {
            fetch('/api/workers')
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('workers-tbody');
                    if(!tbody) return;
                    tbody.innerHTML = '';
                    
                    data.workers.forEach(w => {
                        const row = document.createElement('tr');
                        const date = new Date(w.registration_date);
                        row.innerHTML = `
                            <td style="font-weight:700;">${w.worker_id}</td>
                            <td>${w.name}</td>
                            <td><span class="status-badge" style="background:rgba(99,102,241,0.1); color:var(--primary); border:none;">${w.num_images} Foto</span></td>
                            <td style="color:var(--text-slate-400); font-size:12px;">${date.toLocaleString()}</td>
                            <td>
                                <div style="display:flex; gap:8px;">
                                    <button class="btn-action" onclick="openEditWorkerModal('${w.worker_id}', '${w.name}')" style="padding:4px 10px; font-size:11px;">Edit</button>
                                    <button class="btn-action" onclick="deleteWorker('${w.worker_id}')" style="padding:4px 10px; font-size:11px; color:#f87171;">Hapus</button>
                                </div>
                            </td>
                        `;
                        tbody.appendChild(row);
                    });
                });
        }

        function registerWorker(e) {
            e.preventDefault();
            const btn = document.getElementById('worker-submit-btn');
            btn.disabled = true;
            btn.textContent = 'Registering...';
            
            const formData = new FormData();
            formData.append('worker_id', document.getElementById('worker-id').value);
            formData.append('name', document.getElementById('worker-name').value);
            
            const files = document.getElementById('worker-images').files;
            if (files.length !== 10) {
                alert('Harus pas 10 foto ya Pak, tidak boleh kurang atau lebih. Saat ini Bapak memilih ' + files.length + ' foto.');
                btn.disabled = false;
                btn.textContent = 'Daftarkan Pekerja';
                return;
            }
            
            for(let i=0; i<files.length; i++){
                formData.append('images', files[i]);
            }
            
            fetch('/api/workers/register', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                if(data.success){
                    alert('Worker registered successfully!');
                    document.getElementById('worker-form').reset();
                    loadWorkers();
                } else {
                    alert('Failed: ' + data.message);
                }
            })
            .catch(err => alert('Error registering worker'))
            .finally(() => {
                btn.disabled = false;
                btn.textContent = 'Register';
            });
        }

        function deleteWorker(id) {
            if(!confirm('Delete worker ' + id + '?')) return;
            fetch('/api/workers/' + id, {method: 'DELETE'})
            .then(r => r.json())
            .then(data => {
                if(data.success){
                    loadWorkers();
                } else {
                    alert('Failed to delete');
                }
            });
        }

        function openEditWorkerModal(id, name) {
            document.getElementById('edit-worker-id').value = id;
            document.getElementById('edit-worker-name').value = name;
            document.getElementById('worker-modal').classList.add('active');
        }

        function closeWorkerModal() {
            document.getElementById('worker-modal').classList.remove('active');
        }

        function saveWorkerEdit() {
            const id = document.getElementById('edit-worker-id').value;
            const name = document.getElementById('edit-worker-name').value;
            
            if(!name) {
                alert('Nama tidak boleh kosong');
                return;
            }
            
            const btn = document.getElementById('worker-edit-submit-btn');
            btn.disabled = true;
            btn.textContent = 'Menyimpan...';
            
            fetch('/api/workers/' + id, {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name: name })
            })
            .then(r => r.json())
            .then(data => {
                if(data.success) {
                    closeWorkerModal();
                    loadWorkers();
                } else {
                    alert('Gagal menyimpan: ' + (data.error || 'Server error'));
                }
            })
            .catch(err => alert('Error updating worker'))
            .finally(() => {
                btn.disabled = false;
                btn.textContent = 'Simpan Perubahan';
            });
        }

        // Guided Camera Registration Logic (RESTORED)
        let registrationStream = null;
        let captureActive = false;
        let currentStep = 0;
        const totalSteps = 10;
        const instructions = [
            "Hadap Depan (Tatap Kamera)",
            "Sedikit ke Kiri",
            "Sedikit ke Kanan",
            "Mendongak Sedikit (Atas)",
            "Menunduk Sedikit (Bawah)",
            "Ekspresi Senyum",
            "Hadap Depan (Normal)",
            "Miring Kiri 45 Derajat",
            "Miring Kanan 45 Derajat",
            "Tatap Kamera (Terakhir!)"
        ];

        async function openCameraRegistration() {
            const workerId = document.getElementById('worker-id').value;
            const workerName = document.getElementById('worker-name').value;
            
            if(!workerId || !workerName) {
                alert('Mohon isi Worker ID dan Nama Lengkap terlebih dahulu!');
                return;
            }

            try {
                registrationStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 1280, height: 720 } 
                });
                const video = document.getElementById('registration-video');
                video.srcObject = registrationStream;
                document.getElementById('registration-camera-modal').classList.add('active');
                
                // Reset UI
                currentStep = 0;
                updateCaptureUI();
                document.getElementById('start-capture-btn').disabled = false;
                document.getElementById('start-capture-btn').textContent = 'Mulai Pengambilan Foto';
            } catch (err) {
                alert('Gagal mengakses kamera: ' + err.message);
            }
        }

        function closeCameraRegistration() {
            if(registrationStream) {
                registrationStream.getTracks().forEach(track => track.stop());
            }
            document.getElementById('registration-camera-modal').classList.remove('active');
            captureActive = false;
        }

        function updateCaptureUI() {
            const progress = (currentStep / totalSteps) * 100;
            document.getElementById('capture-progress').style.width = progress + '%';
            document.getElementById('capture-count').textContent = `${currentStep} / ${totalSteps} Foto`;
            document.getElementById('capture-instruction').textContent = instructions[currentStep] || 'Selesai!';
            
            // Visual feedback box
            const guide = document.getElementById('face-guide-box');
            guide.style.borderColor = captureActive ? 'var(--primary)' : '#fff';
        }

        async function startGuidedCapture() {
            if(captureActive) return;
            captureActive = true;
            document.getElementById('start-capture-btn').disabled = true;
            
            const workerId = document.getElementById('worker-id').value;
            const workerName = document.getElementById('worker-name').value;
            const video = document.getElementById('registration-video');
            const canvas = document.getElementById('registration-canvas');
            const context = canvas.getContext('2d');
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            for(currentStep = 0; currentStep < totalSteps; currentStep++) {
                updateCaptureUI();
                
                // Countdown/Delay sebelum tiap foto (2 detik agar user sempat ganti pose)
                for(let i=3; i>0; i--) {
                    if(!captureActive) return;
                    document.getElementById('capture-instruction').textContent = `${instructions[currentStep]} (${i}...)`;
                    await new Promise(r => setTimeout(r, 700));
                }

                if(!captureActive) return;
                
                // Flash effect
                document.getElementById('capture-overlay').style.background = '#fff';
                setTimeout(() => document.getElementById('capture-overlay').style.background = 'rgba(0,0,0,0.4)', 50);

                // Take Snapshot
                // Resize ke 640x480 agar ringan dikirim (sudah cukup untuk AI)
                const targetWidth = 640;
                const targetHeight = (video.videoHeight / video.videoWidth) * targetWidth;
                canvas.width = targetWidth;
                canvas.height = targetHeight;
                
                context.drawImage(video, 0, 0, targetWidth, targetHeight);
                // Kompresi kualitas ke 0.7 (70%) agar payload kecil (< 100KB)
                const imageData = canvas.toDataURL('image/jpeg', 0.7);

                // Send to backend
                try {
                    const response = await fetch('/api/workers/capture-step', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Accept': 'application/json'
                        },
                        body: JSON.stringify({
                            worker_id: workerId,
                            worker_name: workerName,
                            image: imageData,
                            step: currentStep
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`Server merespon dengan status ${response.status}. Mungkin sesi Anda habis, silakan refresh halaman.`);
                    }

                    const result = await response.json();
                    if(!result.success) {
                        alert('Kualitas Foto Kurang: ' + result.message);
                        currentStep--; // Ulangi step ini
                        continue;
                    }

                    if(result.done) {
                        alert('Selamat! Registrasi Pekerja Berhasil.');
                        closeCameraRegistration();
                        loadWorkers();
                        return;
                    }
                } catch (err) {
                    console.error('Capture Error:', err);
                    alert(`Gagal mengirim foto: ${err.message}\n\nSaran: Coba Refresh (F5) halaman dan login kembali.`);
                    captureActive = false;
                    return;
                }
            }
            
            captureActive = false;
            updateCaptureUI();
        }

        function startCamera(cameraId) {
            fetch(`/api/camera/${cameraId}/start`, {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        loadCameras();
                    }
                });
        }
        
        function stopCamera(cameraId) {
            fetch(`/api/camera/${cameraId}/stop`, {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        loadCameras();
                    }
                });
        }
        
        function handleSourceChange() {
            const source = document.getElementById('camera-source').value;
            const rtspGroup = document.getElementById('rtsp-group');
            const fileGroup = document.getElementById('file-group');
            
            // Hide all optional groups
            rtspGroup.style.display = 'none';
            fileGroup.style.display = 'none';
            
            // Show relevant group based on selection
            if (source === 'rtsp') {
                rtspGroup.style.display = 'block';
            } else if (source === 'file') {
                fileGroup.style.display = 'block';
            }
        }
        
        function testRtsp() {
            const url = document.getElementById('rtsp-url').value;
            if (!url) {
                alert('Masukkan RTSP URL terlebih dahulu');
                return;
            }
            
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = 'Testing...';
            
            fetch('/api/test_rtsp', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url})
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert(data.message || 'RTSP connection OK');
                    } else {
                        alert(data.message || 'Gagal koneksi ke RTSP stream');
                    }
                })
                .catch(() => {
                    alert('Terjadi error saat mengetes RTSP');
                })
                .finally(() => {
                    btn.disabled = false;
                    btn.textContent = 'Test';
                });
        }
        
        function diagnoseRtsp() {
            const url = document.getElementById('rtsp-url').value;
            if (!url) {
                alert('Masukkan RTSP URL terlebih dahulu');
                return;
            }
            
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = 'Diagnosing...';
            
            fetch('/api/diagnose_rtsp', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url})
            })
                .then(r => r.json())
                .then(data => {
                    let message = 'DIAGNOSTIC RESULTS:\\n\\n';
                    
                    if (data.diagnostic) {
                        const reachable = data.diagnostic.network_reachable ? 'YES' : 'NO';
                        message += 'Network Reachable: ' + reachable + '\\n';
                        const portOpen = data.diagnostic.rtsp_port_open ? 'YES' : 'NO';
                        message += 'RTSP Port Open: ' + portOpen + '\\n\\n';
                        
                        if (data.working_url) {
                            message += 'WORKING URL FOUND:\\n' + data.working_url + '\\n\\n';
                        }
                        
                        if (data.diagnostic.recommendations && data.diagnostic.recommendations.length > 0) {
                            message += 'RECOMMENDATIONS:\\n';
                            for (let i = 0; i < Math.min(5, data.diagnostic.recommendations.length); i++) {
                                message += '- ' + data.diagnostic.recommendations[i] + '\\n';
                            }
                        }
                    } else {
                        message += data.message || 'Diagnostic failed';
                    }
                    
                    alert(message);
                })
                .catch(() => {
                    alert('Terjadi error saat diagnose RTSP');
                })
                .finally(() => {
                    btn.disabled = false;
                    btn.textContent = 'Diagnose';
                });
        }
        
        function openAddCameraModal() {
            resetCameraForm();
            document.getElementById('modal-title').innerText = 'Tambah Kamera Baru';
            document.getElementById('camera-modal').classList.add('active');
        }

        function closeCameraModal() {
            document.getElementById('camera-modal').classList.remove('active');
            resetCameraForm();
        }

        function addCamera() {
            const name = document.getElementById('camera-name').value;
            let source = document.getElementById('camera-source').value;
            
            // Handle different source types
            if (source === 'rtsp') {
                source = document.getElementById('rtsp-url').value;
            } else if (source === 'file') {
                source = document.getElementById('file-path').value;
            }
            
            if (!name || !source) {
                alert('Harap isi semua field yang diperlukan');
                return;
            }
            
            const payload = {name, source};
            let url = '/api/cameras';
            let method = 'POST';
            
            if (editingCameraId !== null) {
                url = `/api/camera/${editingCameraId}`;
                method = 'PUT';
            }
            
            fetch(url, {
                method: method,
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        closeCameraModal();
                        loadCameras();
                    } else {
                        alert(data.error || 'Gagal menyimpan kamera');
                    }
                });
        }
        
        function resetCameraForm() {
            editingCameraId = null;
            document.getElementById('camera-name').value = '';
            document.getElementById('camera-source').value = '';
            document.getElementById('rtsp-url').value = '';
            document.getElementById('file-path').value = '';
            handleSourceChange();
        }
        
        function editCamera(id, name, source) {
            editingCameraId = id;
            document.getElementById('modal-title').innerText = 'Edit Kamera - CAM-' + id;
            document.getElementById('camera-name').value = name;
            
            const sourceSelect = document.getElementById('camera-source');
            const options = Array.from(sourceSelect.options).map(o => o.value);
            
            if (options.includes(String(source))) {
                sourceSelect.value = String(source);
                document.getElementById('rtsp-url').value = '';
                document.getElementById('file-path').value = '';
            } else if (String(source).startsWith('rtsp://')) {
                sourceSelect.value = 'rtsp';
                document.getElementById('rtsp-url').value = source;
                document.getElementById('file-path').value = '';
            } else {
                sourceSelect.value = 'file';
                document.getElementById('file-path').value = source;
                document.getElementById('rtsp-url').value = '';
            }
            
            handleSourceChange();
            document.getElementById('camera-modal').classList.add('active');
        }
        
        function deleteCamera(id) {
            if (!confirm('Are you sure you want to delete this camera?')) {
                return;
            }
            
            fetch(`/api/camera/${id}`, {method: 'DELETE'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        if (editingCameraId === id) {
                            resetCameraForm();
                        }
                        loadCameras();
                    } else {
                        alert(data.error || 'Failed to delete camera');
                    }
                });
        }
        
        function exportViolations() {
            window.open('/api/violations/export');
        }
        
        function loadCurrentSettings() {
            fetch('/api/settings')
                .then(r => r.json())
                .then(data => {
                    if (data.confidence !== undefined) {
                        document.getElementById('confidence-setting').value = data.confidence;
                        document.getElementById('confidence-value').textContent = parseFloat(data.confidence).toFixed(2);
                    }
                    if (data.cooldown !== undefined) {
                        document.getElementById('cooldown-setting').value = data.cooldown;
                    }
                });
        }

        function saveSettings() {
            const cooldown = document.getElementById('cooldown-setting').value;
            const confidence = document.getElementById('confidence-setting').value;
            
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({cooldown, confidence})
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        // Success toast/alert can be added here
                        const btn = document.querySelector('button[onclick="saveSettings()"]');
                        const originalText = btn.innerHTML;
                        btn.innerHTML = '<span>✅</span> Tersimpan!';
                        btn.style.background = '#10b981';
                        setTimeout(() => {
                            btn.innerHTML = originalText;
                            btn.style.background = '';
                        }, 2000);
                    }
                });
        }
        
        // Auto-refresh
        setInterval(() => {
            if (currentTab === 'statistics') {
                loadStatistics();
            } else if (currentTab === 'violations') {
                loadViolations();
            }
        }, 5000);
        
        function loadCaptures() {
            const grid = document.getElementById('captures-grid');
            if(!grid) return;
            grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: #888;">Loading captures...</div>';
            
            fetch('/api/captures')
                .then(r => r.json())
                .then(data => {
                    grid.innerHTML = '';
                    if (!data.captures || data.captures.length === 0) {
                        grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: #888;">No unknown captures found yet.</div>';
                        return;
                    }
                    
                    data.captures.forEach(c => {
                        const card = document.createElement('div');
                        card.className = 'card';
                        card.style.padding = '16px';
                        card.innerHTML = `
                            <div style="aspect-ratio: 16/9; background: #000; border-radius: 8px; overflow: hidden; margin-bottom: 12px;">
                                <img src="data:image/jpeg;base64,${c.preview}" style="width: 100%; height: 100%; object-fit: contain;">
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <span style="font-weight: 700; color: var(--text-slate-50); font-size: 14px;">${c.id}</span>
                                <span class="status-badge" style="background: rgba(99, 102, 241, 0.1); color: var(--primary); border: none;">${c.image_count} Foto</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                                <input type="text" id="name-${c.id}" placeholder="Nama Lengkap" class="form-group" style="padding: 8px; font-size: 12px; margin-bottom: 0; grid-column: 1/-1;">
                                <input type="text" id="id-${c.id}" placeholder="Worker ID (W-XXX)" class="form-group" style="padding: 8px; font-size: 12px; margin-bottom: 0; grid-column: 1/-1;">
                                <button class="btn-action btn-primary" data-action="register-capture" data-id="${c.id}" style="justify-content: center; padding: 8px;">Daftarkan</button>
                                <button class="btn-action" data-action="delete-capture" data-id="${c.id}" style="justify-content: center; padding: 8px; color: #ff4d4d; border-color: rgba(255, 77, 77, 0.2); background: rgba(255, 77, 77, 0.05);">Hapus</button>
                            </div>
                        `;
                        grid.appendChild(card);
                    });
                });
        }

        function registerFromCapture(tempId, btn) {
            const name = document.getElementById('name-' + tempId).value;
            const workerId = document.getElementById('id-' + tempId).value;
            
            if (!name || !workerId) {
                alert('Harap isi Nama dan ID');
                return;
            }
            
            if (btn) {
                btn.disabled = true;
                btn.textContent = 'Mendaftarkan...';
            }
            
            fetch('/api/register_from_capture', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    temp_id: tempId,
                    worker_id: workerId,
                    name: name
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('Berhasil mendaftarkan pekerja!');
                    loadCaptures();
                } else {
                    alert('Gagal: ' + data.message);
                }
            })
            .catch(err => {
                console.error('Registration Error:', err);
                alert('Terjadi kesalahan: ' + err.message);
            })
            .finally(() => {
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = 'Daftarkan';
                }
            });
        }

        function deleteCapture(tempId) {
            console.log('🗑️ Mencoba menghapus capture:', tempId);
            if (!confirm('Apakah Anda yakin ingin menghapus data capture ' + tempId + '?')) return;
            
            fetch('/api/captures/' + tempId, {
                method: 'DELETE'
            })
            .then(r => {
                if (!r.ok) throw new Error('Server returned ' + r.status);
                return r.json();
            })
            .then(data => {
                if (data.success) {
                    console.log('✅ Berhasil dihapus');
                    loadCaptures();
                } else {
                    console.error('❌ Gagal hapus:', data.message);
                    alert('Gagal menghapus: ' + data.message);
                }
            })
            .catch(err => {
                console.error('🌐 Network Error:', err);
                alert('Terjadi kesalahan koneksi: ' + err.message);
            });
        }

        // Delegated event listener for captures grid
        document.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-action]');
            if (!btn) return;
            
            const action = btn.dataset.action;
            const id = btn.dataset.id;
            
            if (action === 'register-capture') {
                registerFromCapture(id, btn);
            } else if (action === 'delete-capture') {
                deleteCapture(id);
            }
        });

        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                loadCameras();
                loadStatistics();
            });
        } else {
            // DOM already loaded
            loadCameras();
            loadStatistics();
        }
    </script>
</body>
</html>
"""

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and user[2] == hashlib.sha256(password.encode()).hexdigest():
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[3]
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(DASHBOARD_TEMPLATE)

# API routes
@app.route('/api/workers/capture-step', methods=['POST'])
def capture_worker_step():
    """Handle a single step of guided camera capture"""
    data = request.get_json()
    worker_id = data.get('worker_id')
    worker_name = data.get('worker_name')
    image_data = data.get('image')
    step = int(data.get('step', 0))
    
    if not all([worker_id, worker_name, image_data]):
        return jsonify({'success': False, 'message': 'Data tidak lengkap'})

    try:
        # Decode image
        import base64
        import numpy as np
        header, encoded = image_data.split(",", 1)
        data_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(data_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Quality Check: Brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        if avg_brightness < 40:
            return jsonify({'success': False, 'message': 'Ruangan terlalu gelap. Mohon tambah pencahayaan.'})
            
        # 2. Quality Check: Face Detection
        faces = detector.face_recognizer.detect_faces(img)
        if not faces:
            return jsonify({'success': False, 'message': 'Wajah tidak terdeteksi. Pastikan wajah masuk dalam kotak.'})
            
        # Pilih wajah terbesar
        best_face = max(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]))
        
        # 3. Quality Check: Positioning (Wajah harus cukup besar > 150px)
        fw = best_face['bbox'][2] - best_face['bbox'][0]
        if fw < 150:
            return jsonify({'success': False, 'message': 'Maju sedikit lagi ke arah kamera.'})

        # Save to worker folder
        temp_dir = os.path.join(PROJECT_ROOT, 'data', 'workers', worker_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = f"capture_{step}.jpg"
        cv2.imwrite(os.path.join(temp_dir, filename), img)
        
        # If last step, trigger sync
        if step == 9:
            # Add to SQLite
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('INSERT OR REPLACE INTO workers (worker_id, name) VALUES (?, ?)', (worker_id, worker_name))
            conn.commit()
            conn.close()
            
            # Sync recognition
            detector.face_recognizer.register_worker(worker_id, worker_name, temp_dir)
            return jsonify({'success': True, 'message': 'Registrasi selesai! Semua foto tersimpan.', 'done': True})

        return jsonify({'success': True, 'message': f'Foto {step+1} berhasil diambil.', 'done': False})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/workers')
def get_workers():
    # Fetch from SQLite first (Source of Truth for Names)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT worker_id, name, created_at FROM workers')
    db_workers = {row[0]: {'name': row[1], 'created_at': row[2]} for row in cursor.fetchall()}
    conn.close()
    
    workers_list = []
    if hasattr(detector, 'face_recognizer') and detector.face_recognizer:
        fr_workers = detector.face_recognizer.get_registered_workers()
        for fw in fr_workers:
            wid = fw['worker_id']
            # Override name with DB name if available
            if wid in db_workers:
                fw['name'] = db_workers[wid]['name']
                # If registration_date is missing from metadata, use created_at
                if not fw.get('registration_date'):
                    fw['registration_date'] = db_workers[wid]['created_at']
            workers_list.append(fw)
    
    return jsonify({'workers': workers_list})

@app.route('/api/workers/register', methods=['POST'])
def register_worker():
    worker_id = request.form.get('worker_id')
    name = request.form.get('name')
    files = request.files.getlist('images')
    
    if not worker_id or not name or not files:
        return jsonify({'success': False, 'message': 'Missing data'})
        
    # Limit to max 10 photos
    if len(files) > 10:
        files = files[:10]
        print(f"ℹ️ Limited upload to 10 photos for worker {worker_id}")
        
    # Ensure FaceRecognizer is initialized
    if not hasattr(detector, 'face_recognizer') or not detector.face_recognizer:
        from src.face_recognition import FaceRecognitionSystem
        detector.face_recognizer = FaceRecognitionSystem()
        
    # Save files to a temporary folder
    # Use absolute path to avoid duplication in web_app subfolder
    worker_folder = os.path.join(PROJECT_ROOT, 'data', 'workers', worker_id)
    os.makedirs(worker_folder, exist_ok=True)
    
    for f in files:
        if f.filename:
            f.save(os.path.join(worker_folder, f.filename))
            
    # Register faces in Pickle
    success = detector.face_recognizer.register_worker(worker_id, name, worker_folder)
    
    if success:
        # Save to SQLite Workers table as source of truth
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO workers (worker_id, name) 
                VALUES (?, ?)
            ''', (worker_id, name))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ Failed to save worker to SQLite: {e}")
            
        detector.use_face_recognition = True
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Failed to extract face features from images'})

@app.route('/api/workers/<worker_id>', methods=['PUT', 'DELETE'])
def worker_detail_api(worker_id):
    if request.method == 'PUT':
        data = request.get_json() or {}
        new_name = data.get('name')
        if not new_name:
            return jsonify({'success': False, 'error': 'Name is required'}), 400
            
        # 1. Update SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('UPDATE workers SET name = ? WHERE worker_id = ?', (new_name, worker_id))
        conn.commit()
        conn.close()
        
        # 2. Update Pickle
        success = False
        if hasattr(detector, 'face_recognizer') and detector.face_recognizer:
            success = detector.face_recognizer.update_worker_name(worker_id, new_name)
            
        return jsonify({'success': success or True})
        
    elif request.method == 'DELETE':
        # 1. Remove from SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM workers WHERE worker_id = ?', (worker_id,))
        conn.commit()
        conn.close()
        
        # 2. Remove from Pickle (Face Recognition Memory)
        if hasattr(detector, 'face_recognizer') and detector.face_recognizer:
            detector.face_recognizer.remove_worker(worker_id)
            
        # 3. Remove physical photo folder
        worker_folder = os.path.join(PROJECT_ROOT, 'data', 'workers', worker_id)
        if os.path.exists(worker_folder):
            import shutil
            try:
                shutil.rmtree(worker_folder)
            except Exception as e:
                print(f"Error deleting folder: {e}")
                
        return jsonify({'success': True})

    # DELETE logic
    if hasattr(detector, 'face_recognizer') and detector.face_recognizer:
        # Remove from Pickle
        success = detector.face_recognizer.remove_worker(worker_id)
        
        # Remove from SQLite
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM workers WHERE worker_id = ?', (worker_id,))
            conn.commit()
            conn.close()
        except:
            pass

        if len(detector.face_recognizer.face_encodings) == 0:
            detector.use_face_recognition = False
        return jsonify({'success': success})
    return jsonify({'success': False})

@app.route('/api/cameras')
def get_cameras():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM cameras ORDER BY id')
    db_cameras = cursor.fetchall()
    conn.close()
    
    camera_list = []
    for cam in db_cameras:
        # Check if camera is actually running in the global cameras dict
        is_active = cam[0] in cameras
        fps = camera_stats.get(cam[0], {}).get('fps', 0.0)
        camera_list.append({
            'id': cam[0],
            'name': cam[1],
            'source': cam[2],
            'status': 'active' if is_active else 'inactive',
            'created_at': cam[4],
            'fps': fps
        })
    
    return jsonify({'cameras': camera_list})

@app.route('/api/cameras', methods=['POST'])
def add_camera():
    data = request.get_json()
    name = data.get('name')
    source = data.get('source')
    if isinstance(source, str):
        source = source.strip()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO cameras (name, source) VALUES (?, ?)', (name, source))
    camera_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Keep camera inactive by default (lighter load). User can start manually.
    return jsonify({'success': True, 'camera_id': camera_id})

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings_api():
    global detection_cooldown
    
    if request.method == 'GET':
        return jsonify({
            'cooldown': detection_cooldown,
            'confidence': detector.confidence_threshold
        })
        
    # POST
    data = request.get_json() or {}
    cooldown = data.get('cooldown')
    confidence = data.get('confidence')
    
    if cooldown is not None:
        try:
            detection_cooldown = float(cooldown)
        except ValueError:
            pass
            
    if confidence is not None:
        try:
            detector.confidence_threshold = float(confidence)
            print(f"🔧 [API] Updated confidence threshold to {detector.confidence_threshold}")
        except ValueError:
            pass
            
    return jsonify({'success': True})

@app.route('/api/camera/<int:camera_id>/start', methods=['POST'])
def start_camera(camera_id):
    # Get camera info from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT source FROM cameras WHERE id = ?', (camera_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'success': False, 'error': 'Camera not found'})
    
    camera_source = result[0]
    if isinstance(camera_source, str):
        camera_source = camera_source.strip()
    success = start_camera_monitoring(camera_id, camera_source)
    
    if success:
        global_stats['active_cameras'] = len(cameras)
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Failed to start camera'})

@app.route('/api/camera/<int:camera_id>/stop', methods=['POST'])
def stop_camera(camera_id):
    success = stop_camera_monitoring(camera_id)
    
    if success:
        global_stats['active_cameras'] = len(cameras)
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Failed to stop camera'})

@app.route('/api/test_rtsp', methods=['POST'])
def test_rtsp():
    """Test a given RTSP URL before adding camera (quick test)"""
    data = request.get_json() or {}
    url = (data.get('url') or '').strip()
    
    if not url:
        return jsonify({'success': False, 'message': 'RTSP URL is required'}), 400
    
    # Quick test dengan timeout pendek
    result = test_rtsp_connection(url, timeout_sec=5)
    
    if result['success'] and result['cap']:
        try:
            ret, frame = result['cap'].read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                result['cap'].release()
                return jsonify({
                    'success': True,
                    'message': f'Connection OK ({width}x{height})',
                    'width': width,
                    'height': height
                })
        except:
            if result['cap']:
                result['cap'].release()
    
    return jsonify({
        'success': False,
        'message': result.get('error', 'Failed to connect to RTSP stream')
    }), 500

@app.route('/api/diagnose_rtsp', methods=['POST'])
def diagnose_rtsp():
    """Diagnose RTSP connection dengan detail lengkap"""
    data = request.get_json() or {}
    url = (data.get('url') or '').strip()
    
    if not url:
        return jsonify({'success': False, 'message': 'RTSP URL is required'}), 400
    
    print(f"\n{'='*60}")
    print(f"🔍 DIAGNOSING RTSP CONNECTION")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"{'='*60}\n")
    
    # Run diagnostic
    diagnostic = diagnose_rtsp_connection(url, timeout_sec=8)
    
    print(f"\n{'='*60}")
    print(f"📊 DIAGNOSTIC RESULTS")
    print(f"{'='*60}")
    print(f"Network Reachable: {diagnostic['network_reachable']}")
    print(f"RTSP Port Open: {diagnostic['rtsp_port_open']}")
    print(f"Tests Performed: {len(diagnostic['connection_tests'])}")
    print(f"{'='*60}\n")
    
    # Find successful connection
    successful_test = None
    for test in diagnostic['connection_tests']:
        if test['success']:
            successful_test = test
            break
    
    return jsonify({
        'success': successful_test is not None,
        'diagnostic': diagnostic,
        'working_url': successful_test['url'] if successful_test else None,
        'message': successful_test['name'] + ' - ' + successful_test['url'] if successful_test else diagnostic.get('recommendations', ['Connection failed'])[0]
    })

@app.route('/api/camera/<int:camera_id>', methods=['PUT', 'POST', 'DELETE'])
def camera_detail(camera_id):
    """Update or delete a single camera (CRUD support)"""
    global global_stats
    
    if request.method in ['PUT', 'POST']:
        data = request.get_json() or {}
        name = data.get('name')
        source = data.get('source')
        if isinstance(source, str):
            source = source.strip()
        
        if not name or not source:
            return jsonify({'success': False, 'error': 'Name and source are required'}), 400
        
        # Update camera info in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('UPDATE cameras SET name = ?, source = ? WHERE id = ?', (name, source, camera_id))
        conn.commit()
        conn.close()
        
        # If camera is currently running, stop it so user can start again with new config
        if camera_id in cameras:
            stop_camera_monitoring(camera_id)
            global_stats['active_cameras'] = len(cameras)
        
        return jsonify({'success': True})
    
    # DELETE
    success = stop_camera_monitoring(camera_id)
    
    # Remove camera from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM cameras WHERE id = ?', (camera_id,))
    conn.commit()
    conn.close()
    
    global_stats['active_cameras'] = len(cameras)
    
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Failed to delete camera'}), 500

@app.route('/api/daily_stats')
def get_daily_stats():
    """Get daily statistics for chart"""
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    conditions = []
    params = []
    
    if start_date:
        conditions.append("DATE(timestamp) >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(timestamp) <= ?")
        params.append(end_date)
        
    if conditions:
        query = f'''
            SELECT DATE(timestamp) as date, 
                   violation_type, 
                   COUNT(*) as count
            FROM violations 
            WHERE {" AND ".join(conditions)}
            GROUP BY DATE(timestamp), violation_type
        '''
        cursor.execute(query, params)
    else:
        # Last 7 days
        cursor.execute('''
            SELECT DATE(timestamp) as date, 
                   violation_type, 
                   COUNT(*) as count
            FROM violations 
            WHERE DATE(timestamp) >= DATE('now', '-7 days')
            GROUP BY DATE(timestamp), violation_type
        ''')
    
    results = cursor.fetchall()
    conn.close()
    
    # Organize data by date
    daily_data = {}
    for date, violation_type, count in results:
        if date not in daily_data:
            daily_data[date] = {'no_helmet': 0, 'no_vest': 0}
        
        if violation_type == 'nohelmet':
            daily_data[date]['no_helmet'] = count
        elif violation_type == 'novest':
            daily_data[date]['no_vest'] = count
    
    return jsonify(daily_data)

@app.route('/api/export')
def export_data():
    """Export violation data in different formats"""
    format_type = request.args.get('format', 'csv')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = '''
        SELECT v.*, c.name as camera_name
        FROM violations v
        LEFT JOIN cameras c ON v.camera_id = c.id
    '''
    params = []
    
    conditions = []
    if start_date:
        conditions.append("DATE(v.timestamp) >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(v.timestamp) <= ?")
        params.append(end_date)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += ' ORDER BY v.timestamp DESC'
    cursor.execute(query, params)
    violations = cursor.fetchall()
    conn.close()
    
    if format_type == 'csv':
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Waktu', 'Tgl', 'Kamera', 'Pelanggaran', 'ID Pekerja', 'Kepercayaan'])
        
        for v in violations:
            wid = v[8] if v[8] and v[8] not in ('', 'Unknown') else 'Unknown ID'
            writer.writerow([v[0], v[5], v[5][:10], v[9] or f"Camera {v[1]}", v[2].replace('no', 'No ').title(), wid, f"{v[3]*100:.1f}%"])
        
        response = app.response_class(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=log_pelanggaran.csv'})
        return response
    
    elif format_type == 'excel':
        import pandas as pd
        import io
        
        data = []
        for v in violations:
            wid = v[8] if v[8] and v[8] not in ('', 'Unknown') else 'Unknown ID'
            data.append({
                'ID': v[0],
                'Waktu': v[5],
                'Tgl': v[5][:10],
                'Kamera': v[9] or f"Camera {v[1]}",
                'Jenis Pelanggaran': v[2].replace('no', 'No ').title(),
                'ID Pekerja': wid,
                'Confidence': f"{v[3]*100:.1f}%"
            })
        
        df = pd.DataFrame(data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Log Pelanggaran')
            # Styling via openpyxl could be added here if needed, but basic df.to_excel is already a real table
        
        response = app.response_class(output.getvalue(), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={'Content-Disposition': 'attachment; filename=log_pelanggaran.xlsx'})
        return response
    
    elif format_type == 'pdf':
        from fpdf import FPDF
        import io
        
        class PDF(FPDF):
            def header(self):
                self.set_font('helvetica', 'B', 16)
                self.cell(0, 10, 'LAPORAN REAL-TIME MONITORING APD', ln=True, align='C')
                self.set_font('helvetica', 'I', 10)
                self.cell(0, 10, 'Sistem Pengawasan Keselamatan Kerja Terautomasi', ln=True, align='C')
                self.ln(10)
            def footer(self):
                self.set_y(-15)
                self.set_font('helvetica', 'I', 8)
                self.cell(0, 10, f'Halaman {self.page_no()}', align='C')
        
        pdf = PDF()
        pdf.add_page()
        pdf.set_font('helvetica', '', 10)
        
        # Metadata
        pdf.cell(0, 10, f"Dicetak pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Total Pelanggaran: {len(violations)}", ln=True)
        pdf.ln(5)
        
        # Table Header
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font('helvetica', 'B', 10)
        col_widths = [10, 40, 35, 40, 30, 25]
        headers = ['No', 'Waktu', 'Lokasi', 'Pelanggaran', 'ID Pekerja', 'Konf.']
        
        for i in range(len(headers)):
            pdf.cell(col_widths[i], 10, headers[i], border=1, fill=True, align='C')
        pdf.ln()
        
        # Table Body
        pdf.set_font('helvetica', '', 9)
        for i, v in enumerate(violations):
            wid = v[8] if v[8] and v[8] not in ('', 'Unknown') else 'Unknown'
            # id(0) camera_id(1) violation_type(2) confidence(3) bbox(4) timestamp(5) image_path(6) processed(7) worker_id(8), camera_name(9)
            row = [
                str(i+1),
                v[5][:19], # timestamp
                v[9] or f"Cam {v[1]}",
                v[2].replace('no', 'No ').title(),
                wid,
                f"{v[3]*100:.0f}%"
            ]
            
            # Check for page break
            if pdf.get_y() > 260:
                pdf.add_page()
                # Re-add headers
                pdf.set_font('helvetica', 'B', 10)
                for j in range(len(headers)):
                    pdf.cell(col_widths[j], 10, headers[j], border=1, fill=True, align='C')
                pdf.ln()
                pdf.set_font('helvetica', '', 9)
                
            for j in range(len(row)):
                pdf.cell(col_widths[j], 10, row[j], border=1, align='C')
            pdf.ln()
            
        response = app.response_class(pdf.output(), mimetype='application/pdf', headers={'Content-Disposition': 'attachment; filename=laporan_pelanggaran.pdf'})
        return response
    
    return jsonify({'error': 'Unsupported format'}), 400

@app.route('/api/statistics')
def get_statistics():
    global global_stats
    
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Filter builder
    where_clauses = []
    params = []
    if start_date:
        where_clauses.append("DATE(timestamp) >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("DATE(timestamp) <= ?")
        params.append(end_date)
    
    where_str = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    # 1. Total Violations
    cursor.execute(f"SELECT COUNT(*) FROM violations{where_str}", params)
    total_violations = cursor.fetchone()[0]
    
    # 2. No Helmet
    helmet_where = " AND ".join(where_clauses + ["violation_type = 'nohelmet'"])
    cursor.execute(f"SELECT COUNT(*) FROM violations WHERE {helmet_where}" if where_clauses else "SELECT COUNT(*) FROM violations WHERE violation_type = 'nohelmet'", params)
    no_helmet_count = cursor.fetchone()[0]
    
    # 3. No Vest
    vest_where = " AND ".join(where_clauses + ["violation_type = 'novest'"])
    cursor.execute(f"SELECT COUNT(*) FROM violations WHERE {vest_where}" if where_clauses else "SELECT COUNT(*) FROM violations WHERE violation_type = 'novest'", params)
    no_vest_count = cursor.fetchone()[0]
    
    # 4. Active Cameras (Always current)
    cursor.execute("SELECT COUNT(*) FROM cameras WHERE status = 'active'")
    active_cameras = cursor.fetchone()[0]
    
    # 5. Average APD Accuracy
    cursor.execute(f"SELECT AVG(confidence) FROM violations{where_str}", params)
    row = cursor.fetchone()
    avg_apd_accuracy = round((row[0] or 0.0) * 100, 1)
    
    # 6. Average Face Accuracy
    avg_face_accuracy = 0.0
    try:
        # Note: face_recognition_log might have different timestamp column or structure
        cursor.execute(f"SELECT AVG(similarity) FROM face_recognition_log{where_str}", params)
        row_face = cursor.fetchone()
        if row_face and row_face[0] is not None:
            avg_face_accuracy = round(row_face[0] * 100, 1)
    except Exception:
        pass
        
    conn.close()
    
    return jsonify({
        'total_violations': total_violations,
        'no_helmet_count': no_helmet_count,
        'no_vest_count': no_vest_count,
        'active_cameras': active_cameras,
        'avg_apd_accuracy': avg_apd_accuracy,
        'avg_face_accuracy': avg_face_accuracy
    })

@app.route('/api/violations')
def get_violations():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    start_date = request.args.get('start', '')
    end_date   = request.args.get('end', '')

    query = '''
        SELECT v.id, v.camera_id, v.violation_type, v.confidence,
               v.timestamp, v.processed, v.worker_id,
               c.name as camera_name
        FROM violations v
        LEFT JOIN cameras c ON v.camera_id = c.id
    '''
    conditions = []
    params = []
    if start_date:
        conditions.append("DATE(v.timestamp) >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(v.timestamp) <= ?")
        params.append(end_date)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += ' ORDER BY v.timestamp DESC LIMIT 100'

    cursor.execute(query, params)
    violations = cursor.fetchall()
    conn.close()

    violation_list = []
    for v in violations:
        # v: id(0) camera_id(1) violation_type(2) confidence(3)
        #    timestamp(4) processed(5) worker_id(6) camera_name(7)
        worker_id = v[6] if v[6] and v[6] != '' else 'Unknown ID'
        violation_list.append({
            'id':            v[0],
            'camera_id':     v[1],
            'violation_type':v[2],
            'confidence':    v[3],
            'timestamp':     v[4],
            'processed':     bool(v[5]),
            'worker_id':     worker_id,
            'camera_name':   v[7] or f'Camera {v[1]}',
        })

    return jsonify({'violations': violation_list})
    
@app.route('/api/violations/reset', methods=['POST'])
def reset_violations():
    """Clear all violations and face recognition statistics"""
    global global_stats
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Clear tables
        cursor.execute("DELETE FROM violations")
        cursor.execute("DELETE FROM face_recognition_log")
        
        conn.commit()
        conn.close()
        
        # Reset in-memory stats too
        global_stats['total_violations'] = 0
        global_stats['no_helmet_count'] = 0
        global_stats['no_vest_count'] = 0
        
        print("🧹 [DB] Statistik dan Log Pelanggaran telah direset.")
        return jsonify({'success': True, 'message': 'Data statistik berhasil direset'})
    except Exception as e:
        print(f"❌ [DB] Error saat mereset data: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/violations/export')
def export_violations():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT v.*, c.name as camera_name 
        FROM violations v 
        LEFT JOIN cameras c ON v.camera_id = c.id 
        ORDER BY v.timestamp DESC
    ''')
    violations = cursor.fetchall()
    conn.close()
    
    # Generate CSV
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Time', 'Camera', 'Type', 'Confidence', 'Status'])
    
    for v in violations:
        writer.writerow([
            v[5],  # timestamp
            v[7] or 'Unknown',  # camera_name
            v[2],  # violation_type
            f"{v[3]:.2f}",  # confidence
            'Processed' if v[6] else 'Pending'  # processed
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=violations.csv'}
    )

@app.route('/camera_feed/<int:camera_id>')
def camera_feed(camera_id):
    return Response(generate_camera_feed(camera_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/captures')
def get_captures():
    captures_path = os.path.join(PROJECT_ROOT, "data", "captures")
    if not os.path.exists(captures_path):
        return jsonify({'captures': []})
    
    captures = []
    for temp_id in os.listdir(captures_path):
        temp_dir = os.path.join(captures_path, temp_id)
        if os.path.isdir(temp_dir):
            images = [img for img in os.listdir(temp_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                # Convert first image to base64 for preview
                try:
                    with open(os.path.join(temp_dir, images[0]), "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                except:
                    encoded_string = ""
                
                captures.append({
                    'id': temp_id,
                    'image_count': len(images),
                    'preview': encoded_string
                })
    
    return jsonify({'captures': captures})

@app.route('/api/register_from_capture', methods=['POST'])
def register_from_capture():
    data = request.json
    temp_id = data.get('temp_id')
    worker_id = data.get('worker_id')
    name = data.get('name')
    
    if not all([temp_id, worker_id, name]):
        return jsonify({'success': False, 'message': 'Missing data'})
    
    temp_dir = os.path.join(PROJECT_ROOT, "data/captures", temp_id)
    if not os.path.exists(temp_dir):
        return jsonify({'success': False, 'message': 'Capture folder not found'})
    
    # Register to Face Recognition System
    if detector.face_recognizer:
        success = detector.face_recognizer.register_worker(worker_id, name, temp_dir)
        if success:
            # Add to SQLite database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('INSERT OR IGNORE INTO workers (worker_id, name) VALUES (?, ?)', (worker_id, name))
            conn.commit()
            conn.close()
            
            # Delete temp folder
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ [PENTING] Gagal menghapus folder sementara {temp_id}: {str(e)}")
            
            # Sync back to detector if needed
            detector.use_face_recognition = True
            
            return jsonify({'success': True, 'message': f'Worker {name} registered successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to register with face recognition system'})
            
    return jsonify({'success': False, 'message': 'Face recognition system not available'})

@app.route('/api/captures/<temp_id>', methods=['DELETE'])
def delete_capture(temp_id):
    capture_id = temp_id.replace('..', '')
    print(f"🗑️ [API] Permintaan hapus capture: {capture_id}")
    
    import shutil
    temp_dir = os.path.join(PROJECT_ROOT, "data/captures", capture_id)
    
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✅ [API] Berhasil menghapus: {temp_dir}")
            return jsonify({'success': True})
        else:
            print(f"⚠️ [API] Folder tidak ditemukan: {temp_dir}")
            return jsonify({'success': False, 'message': 'Folder tidak ditemukan'})
    except Exception as e:
        print(f"❌ [API] Error saat menghapus folder: {str(e)}")
        return jsonify({'success': False, 'message': f'Error sistem: {str(e)}'})

# --- User Management API ---
@app.route('/api/users', methods=['GET', 'POST'])
def api_users():
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if request.method == 'GET':
        cursor.execute('SELECT username, role, created_at FROM users ORDER BY created_at DESC')
        users = [{'username': u[0], 'role': u[1], 'created_at': u[2]} for u in cursor.fetchall()]
        conn.close()
        return jsonify(users)
    
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'petugas')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Missing data'}), 400
        
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                     (username, hashed_password, role))
        conn.commit()
        return jsonify({'success': True})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Username sudah terdaftar'})
    finally:
        conn.close()

@app.route('/api/users/<username>', methods=['DELETE'])
def delete_user(username):
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
    if username == 'admin':
        return jsonify({'success': False, 'message': 'Cannot delete main admin'}), 400
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

if __name__ == '__main__':
    import socket
    
    # Get local LAN IP Address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = '127.0.0.1'

    print("🚀 Starting Advanced APD Monitoring System...")
    print("🔐 Features: Login, Multi-Camera, Data Recap")
    print(f"📊 Open http://{local_ip}:5000 in your browser to access from another device")
    print("👤 Default Login: admin / admin123")
    
    # Run the app on all interfaces bound to port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)