import sqlite3
import os

def create_folders():
    db_path = 'data/apd_monitoring.db'
    base_dataset_path = 'data/dataset'
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} tidak ditemukan.")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT worker_id, name FROM workers')
        workers = cursor.fetchall()
        
        os.makedirs(base_dataset_path, exist_ok=True)
        
        for idx, (worker_id, name) in enumerate(workers, 1):
            # Format nama folder: workerid_nama_pekerja
            folder_name = f"{worker_id}_{name.replace(' ', '_')}"
            folder_path = os.path.join(base_dataset_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            print(f"[{idx}/{len(workers)}] Dibuat folder: {folder_path}")
            
        print("\nPenyiapan folder selesai. Silakan letakkan foto wajah di dalam folder masing-masing.")
    except Exception as e:
        print(f"Error mengakses database: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    create_folders()
