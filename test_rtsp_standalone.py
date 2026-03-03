import cv2
import time
import socket
import os

url = "rtsp://service:cctv@172.19.156.152/live.sdp"
host = "172.19.156.152"
port = 554

print(f"Testing connectivity to {host}:{port}...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((host, port))
    if result == 0:
        print(f"✅ Port {port} is OPEN")
    else:
        print(f"❌ Port {port} is CLOSED or FILTERED (Error: {result})")
    sock.close()
except Exception as e:
    print(f"❌ Socket error: {e}")

print(f"\nTesting RTSP URL: {url}")
# Set OPEN_TIMEOUT_MSEC to avoid long waits
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp" # Force TCP

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("❌ Failed to open RTSP stream")
else:
    print("✅ Stream opened successfully")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Read frame: {frame.shape}")
    else:
        print("❌ Failed to read frame")
    cap.release()
