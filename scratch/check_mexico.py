import sqlite3
import pandas as pd

conn = sqlite3.connect("data/futebot.db")
df = pd.read_sql_query("""
    SELECT p.*, m.nome as mandante_nome, v.nome as visitante_nome 
    FROM partidas p
    JOIN selecoes m ON p.mandante_id = m.id
    JOIN selecoes v ON p.visitante_id = v.id
    WHERE p.ano_copa = 2026 AND (m.nome = 'México' OR v.nome = 'México')
""", conn)
print(df)
conn.close()
