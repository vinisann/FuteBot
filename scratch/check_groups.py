import sqlite3
conn = sqlite3.connect("data/futebot.db")
cursor = conn.cursor()
cursor.execute("""
    SELECT distinct grupo, mandante_id, (select nome from selecoes where id = mandante_id) as nome 
    FROM partidas 
    WHERE ano_copa = 2026 AND grupo IS NOT NULL
""")
groups = {}
for r in cursor.fetchall():
    g = r[0]
    t = r[2]
    if g not in groups:
        groups[g] = []
    groups[g].append(t)
for g in sorted(groups.keys()):
    print(g, len(set(groups[g])), sorted(list(set(groups[g]))))
conn.close()
