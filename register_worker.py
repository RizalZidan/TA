import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.face_recognition import FaceRecognitionSystem

def main():
    parser = argparse.ArgumentParser(description="Pendaftaran Pekerja untuk Face Recognition")
    parser.add_argument("--id", required=True, help="ID Pekerja (misal: w001)")
    parser.add_argument("--name", required=True, help="Nama Pekerja")
    parser.add_argument("--images", required=True, help="Folder berisi foto-foto wajah pekerja")
    args = parser.parse_args()

    fr_system = FaceRecognitionSystem()
    print("\n[+] Memulai pendaftaran wajah...")
    success = fr_system.register_worker(args.id, args.name, args.images)
    
    if success:
        print("\n[✔] Berhasil Mendaftar! Sistem Face Recognition akan mengenali pekerja ini pada kamera.")
    else:
        print("\n[✖] Gagal mendaftarkan wajah. Pastikan folder berisi gambar (PNG/JPG) yang jelas memperlihatkan wajah (1 wajah per gambar).")

if __name__ == "__main__":
    main()
