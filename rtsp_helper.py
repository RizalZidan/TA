#!/usr/bin/env python3
"""
RTSP Connection Helper Tool
Advanced RTSP testing and connection diagnostics
"""

import cv2
import time
import subprocess
import platform
import socket
import requests
from urllib.parse import urlparse

def test_rtsp_connection(rtsp_url, timeout_ms=10000):
    """Test RTSP connection with detailed diagnostics"""
    print(f"🎯 Testing RTSP: {rtsp_url}")
    print(f"⏱️ Timeout: {timeout_ms}ms")
    
    try:
        # Parse RTSP URL
        parsed = urlparse(rtsp_url)
        host = parsed.hostname
        port = parsed.port or 554
        path = parsed.path
        
        print(f"🌐 Host: {host}")
        print(f"🔌 Port: {port}")
        print(f"📡 Path: {path}")
        
        # Test network connectivity first
        if host:
            print(f"🔍 Testing network connectivity...")
            try:
                # Test TCP connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result_conn = sock.connect_ex((host, port))
                sock.close()
                
                if result_conn == 0:
                    print(f"✅ TCP connection successful to {host}:{port}")
                else:
                    print(f"❌ TCP connection failed to {host}:{port}")
                    return {
                        'success': False,
                        'message': f'TCP connection failed to {host}:{port}',
                        'error': 'Network unreachable'
                    }
            except Exception as e:
                print(f"⚠️ TCP test error: {e}")
                return {
                    'success': False,
                    'message': f'TCP test error: {str(e)}',
                    'error': str(e)
                }
        
        # Test with OpenCV
        print(f"📹 Testing OpenCV connection...")
        cap = cv2.VideoCapture(rtsp_url)
        
        # Set timeouts
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Test if opened
        if cap.isOpened():
            print(f"✅ OpenCV: Connection opened successfully")
            
            # Try to read frame
            start_time = time.time()
            ret, frame = cap.read()
            elapsed_ms = (time.time() - start_time) * 1000
            
            if ret and frame is not None:
                print(f"✅ OpenCV: Frame received successfully!")
                print(f"📐 Frame shape: {frame.shape}")
                print(f"⏱️ Read time: {elapsed_ms:.0f}ms")
                
                # Test multiple reads
                success_count = 0
                for i in range(3):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        success_count += 1
                        print(f"✅ OpenCV: Test frame {i+1} received")
                    else:
                        print(f"❌ OpenCV: Test frame {i+1} failed")
                
                print(f"📊 Frame read success rate: {success_count}/3 ({(success_count/3)*100:.1f}%)")
                
                cap.release()
                return {
                    'success': True,
                    'message': f'RTSP connection successful to {host}:{port}',
                    'frame_shape': frame.shape if frame is not None else None,
                    'read_time_ms': elapsed_ms,
                    'success_rate': f"{(success_count/3)*100:.1f}%"
                }
            else:
                print(f"❌ OpenCV: Failed to read frame")
                cap.release()
                return {
                    'success': False,
                    'message': f'RTSP connection opened but failed to read frame',
                    'error': 'Frame read timeout'
                }
        else:
            print(f"❌ OpenCV: Failed to open connection")
            return {
                'success': False,
                'message': f'RTSP connection failed to {host}:{port}',
                'error': 'Connection failed'
            }
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return {
            'success': False,
            'message': f'Exception: {str(e)}',
            'error': str(e)
        }

def test_multiple_rtsp_endpoints(base_url, ports=None):
    """Test multiple RTSP endpoints"""
    if ports is None:
        ports = [554, 8554, 1935, 8080, 8081, 9090]
    
    print(f"🎯 Testing RTSP endpoints for: {base_url}")
    
    results = []
    for port in ports:
        print(f"\n🔍 Testing port {port}...")
        
        # Common RTSP path patterns
        paths = [
            '',
            '/live',
            '/stream',
            '/camera1',
            '/cam/realmonitor',
            '/h264/stream',
            '/mjpeg/stream',
            '/video',
            '/media',
            '/axis-media/media.amp'
        ]
        
        for path in paths:
            rtsp_url = f"rtsp://{base_url}:{port}{path}"
            result = test_rtsp_connection(rtsp_url, timeout_ms=5000)
            results.append({
                'url': rtsp_url,
                'port': port,
                'path': path,
                **result
            })
            
            if result['success']:
                print(f"✅ SUCCESS: {rtsp_url}")
                break
        else:
            print(f"❌ FAILED: {rtsp_url}")
    
    return results

def analyze_rtsp_url(rtsp_url):
    """Analyze RTSP URL and provide recommendations"""
    print(f"\n🔍 Analyzing RTSP URL: {rtsp_url}")
    
    try:
        parsed = urlparse(rtsp_url)
        host = parsed.hostname
        port = parsed.port or 554
        path = parsed.path
        
        print(f"🌐 Host: {host}")
        print(f"🔌 Port: {port}")
        print(f"📡 Path: {path}")
        
        # Recommendations based on common patterns
        recommendations = []
        
        # Check if it's a common IP camera
        if any(keyword in rtsp_url.lower() for keyword in ['camera', 'cam', 'dvr', 'ipcam', 'hikvision']):
            recommendations.append("📹 This looks like an IP camera")
            recommendations.append("💡 Try common paths: /live, /stream, /camera1")
            recommendations.append("🔧 Check if camera requires authentication")
            recommendations.append("🌐 Verify network connectivity")
        
        # Check for common RTSP patterns
        if 'live' in rtsp_url.lower():
            recommendations.append("📹 /live is a common RTSP path")
        elif 'stream' in rtsp_url.lower():
            recommendations.append("📹 /stream is a common RTSP path")
        elif 'camera1' in rtsp_url.lower():
            recommendations.append("📹 /camera1 is a common RTSP path")
        
        # Port recommendations
        if port == 554:
            recommendations.append("🔌 Port 554 is the standard RTSP port")
        elif port == 8554:
            recommendations.append("🔌 Port 8554 is commonly used by IP cameras")
        elif port == 1935:
            recommendations.append("🔌 Port 1935 is sometimes used by IP cameras")
        elif port == 8080:
            recommendations.append("🔌 Port 8080 is sometimes used by IP cameras")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return {
            'host': host,
            'port': port,
            'path': path,
            'recommendations': recommendations
        }
        
    except Exception as e:
        print(f"❌ Error analyzing URL: {e}")
        return {
            'error': str(e)
        }

def main():
    print("🎯 RTSP Connection Helper Tool")
    print("=" * 50)
    
    # Get RTSP URL from user
    rtsp_url = input("Enter RTSP URL to test (or press Enter to skip): ").strip()
    
    if not rtsp_url:
        print("❌ No RTSP URL provided")
        return
    
    print(f"\n🔍 Testing RTSP URL: {rtsp_url}")
    
    # Analyze the URL first
    analysis = analyze_rtsp_url(rtsp_url)
    
    if 'error' in analysis:
        print(f"\n❌ URL Analysis Error: {analysis['error']}")
        return
    
    print(f"\n📊 URL Analysis:")
    print(f"   Host: {analysis['host']}")
    print(f"   Port: {analysis['port']}")
    print(f"   Path: {analysis['path']}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Test the connection
    print(f"\n🎯 Testing connection...")
    result = test_rtsp_connection(rtsp_url, timeout_ms=10000)
    
    print(f"\n📊 Test Results:")
    if result['success']:
        print(f"   ✅ SUCCESS: {result['message']}")
        print(f"   📐 Frame shape: {result.get('frame_shape', 'N/A')}")
        print(f"   ⏱️ Read time: {result.get('read_time_ms', 'N/A')}ms")
        print(f"   📊 Success rate: {result.get('success_rate', 'N/A')}")
    else:
        print(f"   ❌ FAILED: {result['message']}")
        if 'error' in result:
            print(f"   🔥 Error: {result['error']}")
    
    print("\n" + "=" * 50)

if __name__ == '__main__':
    main()
