import sqlite3
import pandas as pd

conn = sqlite3.connect("data/futebot.db")
df = pd.read_sql_query("SELECT distinct fase, grupo, status FROM partidas WHERE ano_copa = 2026", conn)
print(df)
conn.close()
