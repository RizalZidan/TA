import sys
import os
import sqlite3
import shutil

# Tambahkan path ke folder src agar bisa mengimpor face_recognition
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.face_recognition import FaceRecognitionSystem

def get_worker_name(db_path, worker_id):
    """
    Mencoba mengambil nama pekerja dari database SQLite.
    Jika tidak ketemu, gunakan ID sebagai nama basis.
    """
    if not os.path.exists(db_path):
        return f"Pekerja_{worker_id}"
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM workers WHERE worker_id=?", (worker_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
    except Exception as e:
        print(f"Peringatan: Gagal membaca DB untuk worker {worker_id}: {e}")
        pass
        
    return f"Pekerja_{worker_id}"

def retrain_all_workers():
    # 1. Tentukan path database pkl dan folder dataset gambar (Root-relative)
    pkl_db_path = os.path.join("data", "face_database.pkl")
    workers_dir = os.path.join("data", "workers")
    sqlite_db_path = os.path.join("data", "apd_monitoring.db")
    
    # 2. Hapus file pkl lama (untuk reset) atau kita mulai bersih
    if os.path.exists(pkl_db_path):
        # Buat backup sekadar berjaga-jaga
        backup_path = pkl_db_path + ".backup"
        if os.path.exists(backup_path):
            os.remove(backup_path) # Clean old backup
        shutil.copy2(pkl_db_path, backup_path)
        os.remove(pkl_db_path)
        print(f"[*] Menghapus database lama (Backup tersimpan di {backup_path})")

    # 3. Inisiasi sistem face recognition dan arahkan ke path yang benar
    print("[*] Memulai ulang proses training (Feature Extraction) dataset pekerja...")
    fr_system = FaceRecognitionSystem()
    fr_system.database_path = pkl_db_path  # Arahkan penyimpanan ke /web_app/data/
    
    # Kosongkan encodings kalau sisa load
    fr_system.face_encodings = {}
    fr_system.face_metadata = {}

    if not os.path.exists(workers_dir):
        print(f"[!] Folder workers tidak ditemukan di {workers_dir}")
        return

    # 4. Loop setiap folder pekerja
    worker_folders = [f for f in os.listdir(workers_dir) if os.path.isdir(os.path.join(workers_dir, f))]
    
    if not worker_folders:
        print("[!] Tidak ada folder pekerja di dalam direktori dataset.")
        return
        
    print(f"\n[*] Ditemukan dataset dari {len(worker_folders)} pekerja. Melakukan ekstraksi...\n")
    
    for worker_id in worker_folders:
        face_images_path = os.path.join(workers_dir, worker_id)
        worker_name = get_worker_name(sqlite_db_path, worker_id)
        
        print((f"="*40))
        print(f"Training Pekerja: {worker_name} (ID: {worker_id})")
        print((f"="*40))
        
        # Lakukan registrasi yang akan membaca dan mengekstrak gambar-gambar di folder
        fr_system.register_worker(worker_id, worker_name, face_images_path)
        print("\n")

    print(f"[✔] Selesai! Semua fitur wajah terbaru telah disimpan secara terpusat di {pkl_db_path}")

if __name__ == "__main__":
    retrain_all_workers()
