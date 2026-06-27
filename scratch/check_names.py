import sqlite3
conn = sqlite3.connect("data/futebot.db")
cursor = conn.cursor()
cursor.execute("SELECT id, nome FROM selecoes")
for r in cursor.fetchall():
    print(r[0], repr(r[1]))
conn.close()
