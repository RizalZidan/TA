import cv2
from flask import Flask, Response
import socket
import os

app = Flask(__name__)

def generate_frames():
    # Menggunakan CAP_DSHOW di Windows untuk menghindari error kamera bawaan
    if os.name == 'nt':
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Jika gagal atau hasil frame kosong, coba backend default MSMF
        ret, _ = camera.read()
        if not ret:
            print("[WARN] DSHOW gagal membaca frame, mencoba backend default...")
            camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Tidak dapat membuka Webcam Laptop.")
        return

    # Turunkan resolusi agar pengiriman via WiFi lebih ringan
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Turunkan FPS agar tidak berat di jaringan WiFi
    camera.set(cv2.CAP_PROP_FPS, 15)

    print("[INFO] Kamera siap, mulai menyiarkan (streaming)...")

    while True:
        success, frame = camera.read()
        if not success:
            print("[ERROR] Gagal membaca frame dari Webcam")
            break
        else:
            # Kompresi frame ke format JPEG dengan kualitas 70% (memperkecil delay)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            frame_bytes = buffer.tobytes()
            
            # Broadcast frame dalam bentuk parts MIME HTTP/MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    # Ini adalah endpoint / URL yang dihubungi oleh PC Utama
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Laptop Camera Streamer berjalan!</h1><p>Gunakan link <a href='/video'>/video</a> untuk streaming.</p>"

if __name__ == "__main__":
    # Dapatkan IP LAN lokal
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = '127.0.0.1'
        
    print("\n" + "="*70)
    print("--- LAPTOP CAMERA STREAMER ---")
    print("="*70)
    print("Program ini akan mengubah laptop/PC Anda menjadi IP Camera!")
    print(f"\n[INFO] LINK VIDEO SOURCE UNTUK DIMASUKKAN KE DASHBOARD:")
    print(f"   http://{local_ip}:5050/video")
    print("\nTekan CTRL+C di sini untuk menghentikan rekaman webcam.")
    print("="*70 + "\n")
    
    # Jalankan server Flask pada port 5050
    app.run(host='0.0.0.0', port=5050, threaded=True, debug=False)
