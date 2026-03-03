#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script untuk mengecek koneksi RTSP yang ada di database dan konfigurasi
"""

import sys
import io
# Set UTF-8 encoding untuk Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import sqlite3
import cv2
import socket
from urllib.parse import urlparse
import time

def test_tcp_connection(host, port, timeout=5):
    """Test TCP connectivity ke host:port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"   ⚠️ TCP test error: {e}")
        return False

def test_rtsp_connection(rtsp_url, timeout_sec=5):
    """Test RTSP connection dengan berbagai method"""
    print(f"\n🎯 Testing RTSP: {rtsp_url}")
    
    # Parse URL
    try:
        parsed = urlparse(rtsp_url)
        host = parsed.hostname
        port = parsed.port or 554
        path = parsed.path or '/'
        
        print(f"   🌐 Host: {host}")
        print(f"   🔌 Port: {port}")
        print(f"   📡 Path: {path}")
        
        # Test 1: TCP connectivity
        print(f"   🔍 Step 1: Testing TCP connectivity...")
        if test_tcp_connection(host, port, timeout=5):
            print(f"   ✅ TCP connection OK")
        else:
            print(f"   ❌ TCP connection FAILED - Host mungkin tidak reachable")
            return False, "TCP connection failed"
        
        # Test 2: OpenCV connection
        print(f"   🔍 Step 2: Testing OpenCV RTSP connection...")
        timeout_ms = timeout_sec * 1000
        
        # Method 1: Direct connection
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            print(f"   ✅ Connection opened successfully")
            
            # Try to read frame
            start_time = time.time()
            ret, frame = cap.read()
            elapsed = time.time() - start_time
            
            if ret and frame is not None and frame.size > 0:
                print(f"   ✅ Frame received successfully!")
                print(f"   📐 Frame shape: {frame.shape}")
                print(f"   ⏱️ Read time: {elapsed:.2f}s")
                cap.release()
                return True, "Direct connection successful"
            else:
                print(f"   ❌ Failed to read frame")
                cap.release()
                
                # Try TCP transport
                print(f"   🔍 Step 3: Trying TCP transport...")
                tcp_url = f"{rtsp_url}?transport=tcp"
                cap = cv2.VideoCapture(tcp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"   ✅ TCP transport successful!")
                        print(f"   📐 Frame shape: {frame.shape}")
                        cap.release()
                        return True, "TCP transport successful"
                    cap.release()
                
                return False, "Connection opened but cannot read frame"
        else:
            print(f"   ❌ Failed to open connection")
            cap.release()
            return False, "Failed to open connection"
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False, f"Exception: {str(e)}"

def check_database_rtsp():
    """Cek RTSP cameras di database"""
    print("=" * 70)
    print("📊 CHECKING RTSP CONNECTIONS IN DATABASE")
    print("=" * 70)
    
    # Cek database di root dan di data/
    db_paths = ['apd_monitoring.db', 'data/apd_monitoring.db', 'web_app/apd_monitoring.db']
    rtsp_cameras = []
    
    for db_path in db_paths:
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            # Cek apakah tabel cameras ada
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cameras'")
            if cur.fetchone():
                cur.execute('SELECT id, name, source FROM cameras WHERE source LIKE "rtsp://%"')
                cameras = cur.fetchall()
                if cameras:
                    print(f"\n📂 Found {len(cameras)} RTSP camera(s) in {db_path}:")
                    for cid, name, source in cameras:
                        rtsp_cameras.append((cid, name, source.strip(), db_path))
                        print(f"   - Camera {cid}: {name}")
                        print(f"     Source: {source.strip()}")
            
            conn.close()
        except Exception as e:
            # Database tidak ada atau error, skip
            pass
    
    return rtsp_cameras

def main():
    print("\n" + "=" * 70)
    print("🔍 RTSP CONNECTION CHECKER")
    print("=" * 70)
    
    # 1. Cek database
    rtsp_cameras = check_database_rtsp()
    
    # 2. Test RTSP URLs yang ditemukan
    if rtsp_cameras:
        print(f"\n🎯 Testing {len(rtsp_cameras)} RTSP connection(s)...")
        print("-" * 70)
        
        results = []
        for cid, name, source, db_path in rtsp_cameras:
            print(f"\n📹 Camera ID: {cid} | Name: {name}")
            success, message = test_rtsp_connection(source, timeout_sec=8)
            results.append({
                'camera_id': cid,
                'name': name,
                'source': source,
                'success': success,
                'message': message
            })
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        
        for result in results:
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"{status} | Camera {result['camera_id']}: {result['name']}")
            print(f"   URL: {result['source']}")
            print(f"   Message: {result['message']}")
            print()
        
        print(f"📈 Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # Test predefined URLs dari app_advanced.py
    print("\n" + "=" * 70)
    print("💡 TESTING PREDEFINED RTSP URLs")
    print("=" * 70)
    
    predefined_urls = [
        "rtsp://service:cctv@172.19.156.152/live.sdp",
        "rtsp://service:cctv@172.19.156.152:554/live.sdp"
    ]
    
    for url in predefined_urls:
        success, message = test_rtsp_connection(url, timeout_sec=8)
        if success:
            print(f"\n✅ RTSP URL dapat dipanggil: {url}")
        else:
            print(f"\n❌ RTSP URL tidak dapat dipanggil: {url}")
            print(f"   Error: {message}")
    
    # Rekomendasi untuk URL yang salah format
    print("\n" + "=" * 70)
    print("⚠️ REKOMENDASI")
    print("=" * 70)
    
    # Cek URL yang mungkin salah format
    if rtsp_cameras:
        print("\n🔍 URL yang mungkin perlu diperbaiki:")
        for cid, name, source, db_path in rtsp_cameras:
            if '/live.sd' in source and '/live.sdp' not in source:
                print(f"   ❌ Camera {cid} ({name}):")
                print(f"      Current: {source}")
                print(f"      Suggested: {source.replace('/live.sd', '/live.sdp')}")
                print(f"      Issue: Format URL mungkin salah (live.sd -> live.sdp)")
    
    print("\n💡 Tips:")
    print("   1. Pastikan camera dalam jaringan yang sama atau VPN terhubung")
    print("   2. Pastikan camera sudah menyala dan RTSP service aktif")
    print("   3. Cek firewall tidak memblokir port RTSP (554, 8554)")
    print("   4. Verifikasi username dan password di URL RTSP")
    print("   5. Coba test dengan tool seperti VLC media player")
    
    print("\n" + "=" * 70)
    print("✅ Check selesai!")
    print("=" * 70)

if __name__ == '__main__':
    main()
