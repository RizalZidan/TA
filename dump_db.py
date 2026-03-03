import sqlite3
import os

db_path = 'web_app/apd_monitoring.db'
if not os.path.exists(db_path):
    db_path = 'apd_monitoring.db'

print(f"Reading from {db_path}...")
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute('SELECT id, name, source FROM cameras')
rows = cur.fetchall()

with open('db_dump.txt', 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(f"ID: {r[0]}\n")
        f.write(f"NAME: {repr(r[1])}\n")
        f.write(f"SOURCE: {repr(r[2])}\n")
        f.write("-" * 20 + "\n")

print("Dump complete: db_dump.txt")
conn.close()
