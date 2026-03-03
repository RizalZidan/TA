#!/usr/bin/env python3
"""
Network Scanner for IP Cameras
Scans network untuk mencari RTSP cameras
"""

import socket
import threading
import time
import subprocess
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed

def scan_port(host, port, timeout=2):
    """Scan single port on host"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def scan_host(host, ports, timeout=2):
    """Scan multiple ports on host"""
    results = {}
    for port in ports:
        if scan_port(host, port, timeout):
            results[port] = True
    return results

def get_network_range():
    """Get network range based on current IP"""
    try:
        # Get current IP
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        # Extract network range
        if '.' in local_ip:
            parts = local_ip.split('.')
            network_base = f"{parts[0]}.{parts[1]}.{parts[2]}"
            return network_base, int(parts[3])
        return None, None
    except:
        return None, None

def scan_network_for_cameras(network_base, start_ip=1, end_ip=254, ports=None, timeout=2):
    """Scan network for RTSP cameras"""
    if ports is None:
        ports = [554, 8554, 1935, 8080, 8081, 9090]
    
    print(f"🔍 Scanning network: {network_base}.1-{network_base}.254")
    print(f"🔌 Ports: {ports}")
    print(f"⏱️ Timeout: {timeout}s")
    print("=" * 60)
    
    found_cameras = []
    
    # Use ThreadPoolExecutor for faster scanning
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        
        # Submit scan tasks
        for ip_suffix in range(start_ip, end_ip + 1):
            host = f"{network_base}.{ip_suffix}"
            future = executor.submit(scan_host, host, ports, timeout)
            futures.append((host, future))
        
        # Process results
        for host, future in futures:
            try:
                results = future.result(timeout=timeout + 1)
                if results:
                    print(f"✅ Found device: {host}")
                    for port in results:
                        print(f"   🔌 Port {port}: Open")
                        found_cameras.append({
                            'host': host,
                            'port': port,
                            'type': 'RTSP' if port in [554, 8554, 1935] else 'HTTP'
                        })
            except Exception as e:
                pass
    
    return found_cameras

def test_rtsp_urls(found_cameras):
    """Generate and test RTSP URLs for found cameras"""
    print(f"\n🎯 Testing RTSP URLs...")
    print("=" * 60)
    
    rtsp_urls = []
    
    for camera in found_cameras:
        host = camera['host']
        port = camera['port']
        
        # Common RTSP paths
        paths = ['/live', '/stream', '/camera1', '/cam/realmonitor', '/media']
        
        for path in paths:
            if camera['type'] == 'RTSP':
                url = f"rtsp://{host}:{port}{path}"
            else:
                url = f"http://{host}:{port}{path}"
            
            rtsp_urls.append(url)
    
    return rtsp_urls

def main():
    print("🎯 Network Scanner for IP Cameras")
    print("=" * 60)
    
    # Get network range
    network_base, current_ip = get_network_range()
    
    if not network_base:
        print("❌ Could not determine network range")
        return
    
    print(f"🌐 Your IP: {network_base}.{current_ip}")
    print(f"🔍 Network Range: {network_base}.1-{network_base}.254")
    
    # Ask user for scan range
    try:
        start = input("Start IP (1-254, default=1): ").strip()
        end = input("End IP (1-254, default=254): ").strip()
        
        start_ip = int(start) if start else 1
        end_ip = int(end) if end else 254
        
        if start_ip < 1 or end_ip > 254 or start_ip > end_ip:
            print("❌ Invalid IP range")
            return
            
    except ValueError:
        print("❌ Invalid input")
        return
    
    # Scan network
    start_time = time.time()
    found_cameras = scan_network_for_cameras(network_base, start_ip, end_ip)
    scan_time = time.time() - start_time
    
    print(f"\n📊 Scan completed in {scan_time:.1f} seconds")
    print(f"📹 Found {len(found_cameras)} potential camera devices")
    
    if found_cameras:
        print(f"\n🎯 Camera Devices Found:")
        print("-" * 40)
        for i, camera in enumerate(found_cameras, 1):
            print(f"{i}. {camera['host']}:{camera['port']} ({camera['type']})")
        
        # Generate RTSP URLs
        rtsp_urls = test_rtsp_urls(found_cameras)
        
        print(f"\n🔗 RTSP URLs to Test:")
        print("-" * 40)
        for i, url in enumerate(rtsp_urls[:10], 1):  # Show first 10
            print(f"{i}. {url}")
        
        if len(rtsp_urls) > 10:
            print(f"... and {len(rtsp_urls) - 10} more URLs")
        
        print(f"\n💡 Next Steps:")
        print(f"1. Use RTSP Helper: python rtsp_helper.py")
        print(f"2. Test these URLs one by one")
        print(f"3. Add working URL to APD Monitoring System")
        
    else:
        print(f"\n❌ No cameras found")
        print(f"💡 Suggestions:")
        print(f"1. Check if cameras are powered on")
        print(f"2. Verify network connection")
        print(f"3. Try different IP range")
        print(f"4. Check camera documentation for default IP")

if __name__ == '__main__':
    main()
