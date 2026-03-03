#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script untuk memperbaiki format URL RTSP yang salah di database
"""

import sys
import io
# Set UTF-8 encoding untuk Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import sqlite3
from urllib.parse import urlparse, urlunparse

def fix_rtsp_url(url):
    """Perbaiki format URL RTSP yang salah"""
    if not url or not isinstance(url, str):
        return url
    
    url = url.strip()
    
    # Perbaiki /live.sd menjadi /live.sdp
    if '/live.sd' in url and '/live.sdp' not in url:
        url = url.replace('/live.sd', '/live.sdp')
        return url, True
    
    # Perbaiki format lainnya jika diperlukan
    # Misalnya: /live menjadi /live.sdp jika tidak ada ekstensi
    parsed = urlparse(url)
    if parsed.path and parsed.path.endswith('/live') and not parsed.path.endswith('.sdp'):
        # Hanya perbaiki jika memang RTSP dan path adalah /live
        if url.startswith('rtsp://'):
            new_path = parsed.path + '.sdp'
            new_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                new_path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
            return new_url, True
    
    return url, False

def fix_database_rtsp_urls(db_path):
    """Perbaiki semua RTSP URL di database"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Cek apakah tabel cameras ada
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cameras'")
        if not cur.fetchone():
            print(f"   ⚠️ Tabel 'cameras' tidak ditemukan di {db_path}")
            conn.close()
            return []
        
        # Ambil semua RTSP cameras
        cur.execute('SELECT id, name, source FROM cameras WHERE source LIKE "rtsp://%"')
        cameras = cur.fetchall()
        
        if not cameras:
            print(f"   ℹ️ Tidak ada RTSP cameras di {db_path}")
            conn.close()
            return []
        
        fixed_cameras = []
        
        for cid, name, source in cameras:
            original_url = source.strip()
            fixed_url, was_fixed = fix_rtsp_url(original_url)
            
            if was_fixed:
                # Update database
                cur.execute('UPDATE cameras SET source = ? WHERE id = ?', (fixed_url, cid))
                fixed_cameras.append({
                    'id': cid,
                    'name': name,
                    'original': original_url,
                    'fixed': fixed_url
                })
                print(f"   ✅ Camera {cid} ({name}):")
                print(f"      Before: {original_url}")
                print(f"      After:  {fixed_url}")
        
        if fixed_cameras:
            conn.commit()
            print(f"   💾 {len(fixed_cameras)} URL(s) diperbaiki dan disimpan")
        else:
            print(f"   ℹ️ Tidak ada URL yang perlu diperbaiki")
        
        conn.close()
        return fixed_cameras
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []

def main():
    print("=" * 70)
    print("🔧 RTSP URL FIXER")
    print("=" * 70)
    print("\n📂 Memeriksa database...")
    
    # Cek semua database yang mungkin
    db_paths = [
        'apd_monitoring.db',
        'data/apd_monitoring.db',
        'web_app/apd_monitoring.db'
    ]
    
    all_fixed = []
    
    for db_path in db_paths:
        try:
            # Cek apakah file ada
            import os
            if not os.path.exists(db_path):
                continue
            
            print(f"\n📂 Processing: {db_path}")
            print("-" * 70)
            fixed = fix_database_rtsp_urls(db_path)
            all_fixed.extend(fixed)
        except Exception as e:
            print(f"   ⚠️ Error processing {db_path}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    if all_fixed:
        print(f"\n✅ Total {len(all_fixed)} URL(s) berhasil diperbaiki:")
        for cam in all_fixed:
            print(f"   - Camera {cam['id']} ({cam['name']})")
            print(f"     {cam['original']} → {cam['fixed']}")
    else:
        print("\nℹ️ Tidak ada URL yang perlu diperbaiki")
        print("   Semua URL sudah dalam format yang benar!")
    
    print("\n" + "=" * 70)
    print("✅ Proses selesai!")
    print("=" * 70)
    
    # Tanyakan apakah user ingin verify
    if all_fixed:
        print("\n💡 Tips:")
        print("   - Jalankan 'python check_rtsp_connection.py' untuk verify koneksi")
        print("   - Pastikan camera dalam network yang sama untuk testing")

if __name__ == '__main__':
    main()
