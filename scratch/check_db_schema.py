import sqlite3
conn = sqlite3.connect("data/futebot.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cursor.fetchall())
cursor.execute("PRAGMA table_info(partidas)")
print("Partidas columns:", cursor.fetchall())
conn.close()
