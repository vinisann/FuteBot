import sqlite3
import os
import pandas as pd
from datetime import datetime
import requests

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "futebot.db")
VALID_MATCH_STATUSES = ("SCHEDULED", "LIVE", "FINISHED", "POSTPONED", "CANCELLED", "SUSPENDED")

def get_connection():
    """Retorna uma conexão ativa com o banco SQLite com WAL e Foreign Keys habilitados."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
    except Exception:
        pass
    return conn

def init_db():
    """Cria as tabelas do banco de dados se elas não existirem."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Criar tabela de seleções
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS selecoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT UNIQUE NOT NULL,
            sigla TEXT NOT NULL,
            elo_rating REAL NOT NULL,
            ranking_fifa INTEGER NOT NULL
        )
        """)
        
        # Criar tabela de partidas
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS partidas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ano_copa INTEGER NOT NULL,
            data_hora TEXT NOT NULL,
            mandante_id INTEGER NOT NULL,
            visitante_id INTEGER NOT NULL,
            gols_mandante INTEGER,
            gols_visitante INTEGER,
            fase TEXT NOT NULL,
            grupo TEXT,
            status TEXT NOT NULL, -- 'SCHEDULED', 'LIVE', 'FINISHED', 'POSTPONED', 'CANCELLED', 'SUSPENDED'
            vencedor_id INTEGER,
            origem_dados TEXT NOT NULL DEFAULT 'manual',
            FOREIGN KEY (mandante_id) REFERENCES selecoes(id),
            FOREIGN KEY (visitante_id) REFERENCES selecoes(id),
            FOREIGN KEY (vencedor_id) REFERENCES selecoes(id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS previsoes_partidas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            partida_id INTEGER NOT NULL,
            modelo_versao TEXT NOT NULL,
            created_at TEXT NOT NULL,
            xg_mandante REAL NOT NULL,
            xg_visitante REAL NOT NULL,
            prob_mandante REAL NOT NULL,
            prob_empate REAL NOT NULL,
            prob_visitante REAL NOT NULL,
            prev_gols_mandante INTEGER NOT NULL,
            prev_gols_visitante INTEGER NOT NULL,
            gols_mandante_real INTEGER,
            gols_visitante_real INTEGER,
            outcome_correct INTEGER,
            score_exact INTEGER,
            goal_error REAL,
            brier_score REAL,
            evaluated_at TEXT,
            FOREIGN KEY (partida_id) REFERENCES partidas(id),
            UNIQUE (partida_id, modelo_versao)
        )
        """)
        
        # Criar índices para performance
        _ensure_partidas_schema(cursor)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_partidas_ano ON partidas(ano_copa);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_partidas_status ON partidas(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_partidas_origem ON partidas(origem_dados);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_partida ON previsoes_partidas(partida_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_evaluated ON previsoes_partidas(evaluated_at);")
        _dedupe_group_matches(cursor)
        
        conn.commit()
    finally:
        conn.close()
    
    # Popular dados se o banco estiver vazio
    populate_initial_data()

def _ensure_partidas_schema(cursor):
    """Aplica migracoes leves para bancos SQLite criados por versoes antigas."""
    cursor.execute("PRAGMA table_info(partidas)")
    columns = {row[1] for row in cursor.fetchall()}
    if "origem_dados" not in columns:
        cursor.execute("ALTER TABLE partidas ADD COLUMN origem_dados TEXT NOT NULL DEFAULT 'manual'")
        cursor.execute("UPDATE partidas SET origem_dados = 'seed' WHERE ano_copa = 2026")

def _dedupe_group_matches(cursor):
    """Remove duplicatas de confrontos de grupo, preferindo fontes reais a seed."""
    cursor.execute(
        """
        SELECT
            ano_copa,
            grupo,
            MIN(mandante_id, visitante_id) AS team_a,
            MAX(mandante_id, visitante_id) AS team_b,
            COUNT(*) AS total
        FROM partidas
        WHERE grupo IS NOT NULL AND grupo != ''
        GROUP BY ano_copa, grupo, team_a, team_b
        HAVING COUNT(*) > 1
        """
    )
    duplicate_groups = cursor.fetchall()

    source_priority = {
        "api": 0,
        "openfootball": 1,
        "manual": 2,
        "seed": 3,
    }

    for group in duplicate_groups:
        cursor.execute(
            """
            SELECT id, origem_dados
            FROM partidas
            WHERE ano_copa = ?
              AND grupo = ?
              AND MIN(mandante_id, visitante_id) = ?
              AND MAX(mandante_id, visitante_id) = ?
            """,
            (group[0], group[1], group[2], group[3]),
        )
        rows = cursor.fetchall()
        rows = sorted(
            rows,
            key=lambda row: (source_priority.get(row[1], 4), row[0]),
        )
        for duplicate in rows[1:]:
            cursor.execute("DELETE FROM partidas WHERE id = ?", (duplicate[0],))

TRANSLATE_TEAM_NAME = {
    "Argentina": "Argentina",
    "Australia": "Austrália",
    "Belgium": "Bélgica",
    "Brazil": "Brasil",
    "Colombia": "Colômbia",
    "Costa Rica": "Costa Rica",
    "Croatia": "Croácia",
    "Denmark": "Dinamarca",
    "Egypt": "Egito",
    "England": "Inglaterra",
    "France": "França",
    "Germany": "Alemanha",
    "Iceland": "Islândia",
    "Iran": "Irã",
    "Japan": "Japão",
    "Mexico": "México",
    "Morocco": "Marrocos",
    "Nigeria": "Nigéria",
    "Panama": "Panamá",
    "Peru": "Peru",
    "Poland": "Polônia",
    "Portugal": "Portugal",
    "Russia": "Rússia",
    "Saudi Arabia": "Arábia Saudita",
    "Senegal": "Senegal",
    "Serbia": "Sérvia",
    "South Korea": "Coreia do Sul",
    "Spain": "Espanha",
    "Sweden": "Suécia",
    "Switzerland": "Suíça",
    "Tunisia": "Tunísia",
    "Uruguay": "Uruguai",
    "Cameroon": "Camarões",
    "Canada": "Canadá",
    "Ghana": "Gana",
    "Netherlands": "Holanda",
    "Qatar": "Catar",
    "USA": "EUA",
    "United States": "EUA",
    "Wales": "País de Gales",
    "Ecuador": "Equador",
    "South Africa": "África do Sul",
    "Czech Republic": "Chéquia",
    "Czechia": "Chéquia",
    "Bosnia and Herzegovina": "Bósnia e Herzegovina",
    "Bosnia-Herzegovina": "Bósnia e Herzegovina",
    "Paraguai": "Paraguai",
    "Paraguay": "Paraguai",
    "Cape Verde Islands": "Cabo Verde",
    "Congo DR": "RD Congo",
    "Turkey": "Turquia",
    "Curaçao": "Curaçao",
    "Ivory Coast": "Costa do Marfim",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "RD Congo",
    "Uzbekistan": "Uzbequistão",
    "Haiti": "Haiti",
    "Scotland": "Escócia",
    "Norway": "Noruega",
    "Iraq": "Iraque",
    "Austria": "Áustria",
    "Jordan": "Jordânia",
    "Algeria": "Argélia",
    "New Zealand": "Nova Zelândia",
    "Italy": "Itália",
    "Chile": "Chile",
    "Honduras": "Honduras",
    "Jamaica": "Jamaica",
    "Guatemala": "Guatemala",
    "El Salvador": "El Salvador",
    "Trinidad and Tobago": "Trinidad e Tobago",
    "Venezuela": "Venezuela",
    "Bolivia": "Bolívia",
    "China PR": "China",
    "China": "China",
    "Indonesia": "Indonésia",
    "Thailand": "Tailândia",
    "United Arab Emirates": "Emirados Árabes",
    "Bahrain": "Bahrein",
    "Palestine": "Palestina",
    "Cote d'Ivoire": "Costa do Marfim",
    "Korea Republic": "Coreia do Sul",
    "IR Iran": "Irã",
}

def parse_openfootball_matches(data, ano, team_to_id, finished_only=False):
    """Converte JSON do openfootball/worldcup.json para tuplas de partidas."""
    matches = sorted(data.get("matches", []), key=lambda x: x.get("date", ""))
    partidas_db = []
    group_match_counts = {}
    phase_map = {
        "Round of 16": "Oitavas",
        "Quarter-finals": "Quartas",
        "Semi-finals": "Semifinal",
        "Match for third place": "Disputa 3Âº Lugar",
        "Final": "Final",
    }

    for match in matches:
        t1 = match.get("team1")
        t2 = match.get("team2")
        m_pt = TRANSLATE_TEAM_NAME.get(t1)
        v_pt = TRANSLATE_TEAM_NAME.get(t2)
        if not m_pt or not v_pt:
            continue
        if m_pt not in team_to_id or v_pt not in team_to_id:
            continue

        score_dict = match.get("score", {}) or {}
        score_val = score_dict.get("et") or score_dict.get("ft")
        if score_val and len(score_val) == 2:
            gols_m, gols_v = score_val[0], score_val[1]
        else:
            gols_m, gols_v = None, None

        status = "FINISHED" if gols_m is not None and gols_v is not None else "SCHEDULED"
        if finished_only and status != "FINISHED":
            continue

        grupo_raw = match.get("group")
        if grupo_raw and grupo_raw.startswith("Group "):
            grupo = grupo_raw.replace("Group ", "")
            count = group_match_counts.get(grupo, 0) + 1
            group_match_counts[grupo] = count
            if count <= 2:
                fase = "1"
            elif count <= 4:
                fase = "2"
            else:
                fase = "3"
        else:
            fase = phase_map.get(match.get("round", ""), match.get("round", ""))
            grupo = None

        mandante_id = team_to_id[m_pt]
        visitante_id = team_to_id[v_pt]
        vencedor_id = None
        if gols_m is not None and gols_v is not None:
            if gols_m > gols_v:
                vencedor_id = mandante_id
            elif gols_v > gols_m:
                vencedor_id = visitante_id

        time_raw = match.get("time", "")
        time_part = time_raw.split()[0] if time_raw else "00:00"
        data_hora = f"{match.get('date', '')} {time_part}".strip()
        partidas_db.append((
            ano,
            data_hora,
            mandante_id,
            visitante_id,
            gols_m,
            gols_v,
            fase,
            grupo,
            status,
            vencedor_id,
        ))

    return partidas_db

def load_openfootball_data(ano, team_to_id, finished_only=False):
    """
    Consome a base openfootball/worldcup.json para obter confrontos reais de um ano específico.
    Retorna a lista de partidas prontas para inserção.
    """
    url = f"https://raw.githubusercontent.com/openfootball/worldcup.json/master/{ano}/worldcup.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return parse_openfootball_matches(data, ano, team_to_id, finished_only=finished_only)
    
    matches = data.get("matches", [])
    matches = sorted(matches, key=lambda x: x.get("date", ""))
    
    partidas_db = []
    group_match_counts = {}
    
    phase_map = {
        "Round of 16": "Oitavas",
        "Quarter-finals": "Quartas",
        "Semi-finals": "Semifinal",
        "Match for third place": "Disputa 3º Lugar",
        "Final": "Final"
    }
    
    for m in matches:
        t1 = m.get("team1")
        t2 = m.get("team2")
        
        m_pt = TRANSLATE_TEAM_NAME.get(t1)
        v_pt = TRANSLATE_TEAM_NAME.get(t2)
        
        if not m_pt or not v_pt:
            continue
        if m_pt not in team_to_id or v_pt not in team_to_id:
            continue
            
        mandante_id = team_to_id[m_pt]
        visitante_id = team_to_id[v_pt]
        
        date_str = m.get("date", "")
        time_raw = m.get("time", "")
        time_part = time_raw.split()[0] if time_raw else "00:00"
        data_hora = f"{date_str} {time_part}".strip()
        
        score_dict = m.get("score", {})
        if score_dict:
            score_val = score_dict.get("et") or score_dict.get("ft")
            if score_val and len(score_val) == 2:
                gols_m = score_val[0]
                gols_v = score_val[1]
            else:
                gols_m, gols_v = None, None
        else:
            gols_m, gols_v = None, None
            
        fase_antiga = m.get("round", "")
        grupo_raw = m.get("group")
        
        if grupo_raw and grupo_raw.startswith("Group "):
            grupo = grupo_raw.replace("Group ", "")
            count = group_match_counts.get(grupo, 0) + 1
            group_match_counts[grupo] = count
            if count <= 2:
                fase = "1"
            elif count <= 4:
                fase = "2"
            else:
                fase = "3"
        else:
            fase = phase_map.get(fase_antiga, fase_antiga)
            grupo = None
            
        status = "FINISHED"
        
        vencedor_id = None
        if gols_m is not None and gols_v is not None:
            if gols_m > gols_v:
                vencedor_id = mandante_id
            elif gols_v > gols_m:
                vencedor_id = visitante_id
            
        partidas_db.append((
            ano,
            data_hora,
            mandante_id,
            visitante_id,
            gols_m,
            gols_v,
            fase,
            grupo,
            status,
            vencedor_id
        ))
        
    return partidas_db

def populate_initial_data():
    """Popula dados iniciais reais da Copa do Mundo 2026, 2022 e 2018 para dar funcionalidade ao MVP."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Verificar se já existem dados
        cursor.execute("SELECT COUNT(*) FROM selecoes")
        selecoes_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM partidas")
        partidas_count = cursor.fetchone()[0]
        if selecoes_count > 0 and partidas_count > 0:
            return
    finally:
        conn.close()

    print("Populando dados iniciais no banco de dados SQLite...")

    # 1. Inserir Seleções com ELO aproximado pós-Copa 2022 e ranking FIFA de Dez/2022
    selecoes = [
        # Nome, Sigla, ELO, Ranking FIFA
        ("Brasil", "BRA", 1986.0, 6),
        ("Argentina", "ARG", 2128.0, 1),
        ("França", "FRA", 2084.0, 3),
        ("Bélgica", "BEL", 1940.0, 9),
        ("Inglaterra", "ENG", 2055.0, 4),
        ("Holanda", "NED", 1972.0, 8),
        ("Croácia", "CRO", 1910.0, 11),
        ("Itália", "ITA", 1900.0, 13),
        ("Portugal", "POR", 1975.0, 5),
        ("Espanha", "ESP", 2129.0, 2),
        ("Marrocos", "MAR", 1920.0, 7),
        ("Suíça", "SUI", 1880.0, 12),
        ("EUA", "USA", 1840.0, 11),
        ("Alemanha", "GER", 1935.0, 10),
        ("Uruguai", "URU", 1960.0, 16),
        ("Senegal", "SEN", 1820.0, 19),
        ("Japão", "JPN", 1860.0, 20),
        ("Polônia", "POL", 1780.0, 22),
        ("Coreia do Sul", "KOR", 1800.0, 25),
        ("Austrália", "AUS", 1760.0, 27),
        ("Equador", "ECU", 1830.0, 41),
        ("Camarões", "CMR", 1720.0, 43),
        ("Arábia Saudita", "KSA", 1700.0, 49),
        ("Catar", "QAT", 1650.0, 60),
        ("Gana", "GHA", 1680.0, 58),
        ("Canadá", "CAN", 1740.0, 53),
        ("Costa Rica", "CRC", 1700.0, 32),
        ("Sérvia", "SRB", 1810.0, 29),
        ("Dinamarca", "DEN", 1870.0, 18),
        ("Tunísia", "TUN", 1750.0, 30),
        ("México", "MEX", 1800.0, 15),
        ("Irã", "IRN", 1790.0, 24),
        ("País de Gales", "WAL", 1750.0, 28),
        ("Áustria", "AUT", 1850.0, 25),
        ("Iraque", "IRQ", 1610.0, 55),
        ("Noruega", "NOR", 1929.0, 45),
        ("Jordânia", "JOR", 1580.0, 70),
        ("Argélia", "ALG", 1780.0, 35),
        # Novas seleções para a Copa 2026
        ("Haiti", "HAI", 1450.0, 90),
        ("Escócia", "SCO", 1720.0, 40),
        ("África do Sul", "RSA", 1550.0, 60),
        ("Chéquia", "CZE", 1750.0, 36),
        ("Bósnia e Herzegovina", "BIH", 1680.0, 47),
        ("Paraguai", "PAR", 1710.0, 42),
        ("Turquia", "TUR", 1810.0, 26),
        ("Curaçao", "CUW", 1350.0, 85),
        ("Costa do Marfim", "CIV", 1700.0, 38),
        ("Suécia", "SWE", 1810.0, 23),
        ("Egito", "EGY", 1680.0, 37),
        ("Nova Zelândia", "NZL", 1500.0, 94),
        ("Cabo Verde", "CPV", 1520.0, 72),
        ("Colômbia", "COL", 1998.0, 12),
        ("RD Congo", "COD", 1580.0, 52),
        ("Uzbequistão", "UZB", 1620.0, 62),
        ("Panamá", "PAN", 1650.0, 48),
        ("Rússia", "RUS", 1700.0, 38),
        ("Islândia", "ISL", 1550.0, 60),
        ("Nigéria", "NGA", 1720.0, 35),
        ("Peru", "PER", 1750.0, 21)
    ]
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if selecoes_count == 0:
            cursor.executemany(
                "INSERT INTO selecoes (nome, sigla, elo_rating, ranking_fifa) VALUES (?, ?, ?, ?)",
                selecoes
            )
            conn.commit()

        # Mapear nomes para IDs para facilitar a inserção de partidas
        cursor.execute("SELECT nome, id FROM selecoes")
        team_to_id = {row["nome"]: row["id"] for row in cursor.fetchall()}
    finally:
        conn.close()

    # 2. Inserir Partidas Reais da Copa do Mundo 2022
    partidas_2022 = [
        # Data, Mandante, Visitante, Gols Mandante, Gols Visitante, Fase, Status
        # Fase de Grupos
        ("2022-11-20 13:00", "Catar", "Equador", 0, 2, "Grupo A", "FINISHED"),
        ("2022-11-21 13:00", "Senegal", "Holanda", 0, 2, "Grupo A", "FINISHED"),
        ("2022-11-25 10:00", "Catar", "Senegal", 1, 3, "Grupo A", "FINISHED"),
        ("2022-11-25 13:00", "Holanda", "Equador", 1, 1, "Grupo A", "FINISHED"),
        ("2022-11-29 12:00", "Holanda", "Catar", 2, 0, "Grupo A", "FINISHED"),
        ("2022-11-29 12:00", "Equador", "Senegal", 1, 2, "Grupo A", "FINISHED"),
        
        ("2022-11-21 10:00", "Inglaterra", "Irã", 6, 2, "Grupo B", "FINISHED"),
        ("2022-11-21 16:00", "EUA", "País de Gales", 1, 1, "Grupo B", "FINISHED"),
        ("2022-11-25 07:00", "País de Gales", "Irã", 0, 2, "Grupo B", "FINISHED"),
        ("2022-11-25 16:00", "Inglaterra", "EUA", 0, 0, "Grupo B", "FINISHED"),
        ("2022-11-29 16:00", "País de Gales", "Inglaterra", 0, 3, "Grupo B", "FINISHED"),
        ("2022-11-29 16:00", "Irã", "EUA", 0, 1, "Grupo B", "FINISHED"),
        
        ("2022-11-22 07:00", "Argentina", "Arábia Saudita", 1, 2, "Grupo C", "FINISHED"),
        ("2022-11-22 13:00", "México", "Polônia", 0, 0, "Grupo C", "FINISHED"),
        ("2022-11-26 10:00", "Polônia", "Arábia Saudita", 2, 0, "Grupo C", "FINISHED"),
        ("2022-11-26 16:00", "Argentina", "México", 2, 0, "Grupo C", "FINISHED"),
        ("2022-11-30 16:00", "Polônia", "Argentina", 0, 2, "Grupo C", "FINISHED"),
        ("2022-11-30 16:00", "Arábia Saudita", "México", 1, 2, "Grupo C", "FINISHED"),
        
        ("2022-11-22 10:00", "Dinamarca", "Tunísia", 0, 0, "Grupo D", "FINISHED"),
        ("2022-11-22 16:00", "França", "Austrália", 4, 1, "Grupo D", "FINISHED"),
        ("2022-11-26 07:00", "Tunísia", "Austrália", 0, 1, "Grupo D", "FINISHED"),
        ("2022-11-26 13:00", "França", "Dinamarca", 2, 1, "Grupo D", "FINISHED"),
        ("2022-11-30 12:00", "Tunísia", "França", 1, 0, "Grupo D", "FINISHED"),
        ("2022-11-30 12:00", "Austrália", "Dinamarca", 1, 0, "Grupo D", "FINISHED"),
        
        ("2022-11-23 10:00", "Alemanha", "Japão", 1, 2, "Grupo E", "FINISHED"),
        ("2022-11-23 13:00", "Espanha", "Costa Rica", 7, 0, "Grupo E", "FINISHED"),
        ("2022-11-27 07:00", "Japão", "Costa Rica", 0, 1, "Grupo E", "FINISHED"),
        ("2022-11-27 16:00", "Espanha", "Alemanha", 1, 1, "Grupo E", "FINISHED"),
        ("2022-12-01 16:00", "Japão", "Espanha", 2, 1, "Grupo E", "FINISHED"),
        ("2022-12-01 16:00", "Costa Rica", "Alemanha", 2, 4, "Grupo E", "FINISHED"),
        
        ("2022-11-23 07:00", "Marrocos", "Croácia", 0, 0, "Grupo F", "FINISHED"),
        ("2022-11-23 16:00", "Bélgica", "Canadá", 1, 0, "Grupo F", "FINISHED"),
        ("2022-11-27 10:00", "Bélgica", "Marrocos", 0, 2, "Grupo F", "FINISHED"),
        ("2022-11-27 13:00", "Croácia", "Canadá", 4, 1, "Grupo F", "FINISHED"),
        ("2022-12-01 12:00", "Croácia", "Bélgica", 0, 0, "Grupo F", "FINISHED"),
        ("2022-12-01 12:00", "Canadá", "Marrocos", 1, 2, "Grupo F", "FINISHED"),
        
        ("2022-11-24 07:00", "Suíça", "Camarões", 1, 0, "Grupo G", "FINISHED"),
        ("2022-11-24 16:00", "Brasil", "Sérvia", 2, 0, "Grupo G", "FINISHED"),
        ("2022-11-28 07:00", "Camarões", "Sérvia", 3, 3, "Grupo G", "FINISHED"),
        ("2022-11-28 13:00", "Brasil", "Suíça", 1, 0, "Grupo G", "FINISHED"),
        ("2022-12-02 16:00", "Camarões", "Brasil", 1, 0, "Grupo G", "FINISHED"),
        ("2022-12-02 16:00", "Sérvia", "Suíça", 2, 3, "Grupo G", "FINISHED"),
        
        ("2022-11-24 10:00", "Uruguai", "Coreia do Sul", 0, 0, "Grupo H", "FINISHED"),
        ("2022-11-24 13:00", "Portugal", "Gana", 3, 2, "Grupo H", "FINISHED"),
        ("2022-11-28 10:00", "Coreia do Sul", "Gana", 2, 3, "Grupo H", "FINISHED"),
        ("2022-11-28 16:00", "Portugal", "Uruguai", 2, 0, "Grupo H", "FINISHED"),
        ("2022-12-02 12:00", "Coreia do Sul", "Portugal", 2, 1, "Grupo H", "FINISHED"),
        ("2022-12-02 12:00", "Gana", "Uruguai", 0, 2, "Grupo H", "FINISHED"),
        
        # Oitavas de Final
        ("2022-12-03 12:00", "Holanda", "EUA", 3, 1, "Oitavas", "FINISHED"),
        ("2022-12-03 16:00", "Argentina", "Austrália", 2, 1, "Oitavas", "FINISHED"),
        ("2022-12-04 12:00", "França", "Polônia", 3, 1, "Oitavas", "FINISHED"),
        ("2022-12-04 16:00", "Inglaterra", "Senegal", 3, 0, "Oitavas", "FINISHED"),
        ("2022-12-05 12:00", "Japão", "Croácia", 1, 1, "Oitavas", "FINISHED"),
        ("2022-12-05 16:00", "Brasil", "Coreia do Sul", 4, 1, "Oitavas", "FINISHED"),
        ("2022-12-06 12:00", "Marrocos", "Espanha", 0, 0, "Oitavas", "FINISHED"),
        ("2022-12-06 16:00", "Portugal", "Suíça", 6, 1, "Oitavas", "FINISHED"),
        
        # Quartas de Final
        ("2022-12-09 12:00", "Croácia", "Brasil", 1, 1, "Quartas", "FINISHED"),
        ("2022-12-09 16:00", "Holanda", "Argentina", 2, 2, "Quartas", "FINISHED"),
        ("2022-12-10 12:00", "Marrocos", "Portugal", 1, 0, "Quartas", "FINISHED"),
        ("2022-12-10 16:00", "Inglaterra", "França", 1, 2, "Quartas", "FINISHED"),
        
        # Semifinais
        ("2022-12-13 16:00", "Argentina", "Croácia", 3, 0, "Semifinal", "FINISHED"),
        ("2022-12-14 16:00", "França", "Marrocos", 2, 0, "Semifinal", "FINISHED"),
        
        # Terceiro Lugar
        ("2022-12-17 12:00", "Croácia", "Marrocos", 2, 1, "Disputa 3º Lugar", "FINISHED"),
        
        # Final
        ("2022-12-18 12:00", "Argentina", "França", 3, 3, "Final", "FINISHED"),
    ]

    # Converter partidas da Copa 2022 para inserir
    partidas_db = []

    # 2. Inserir Partidas Reais
    # Tentar carregar dados de 2022 dinamicamente
    try:
        print("Buscando dados da Copa 2022 do openfootball...")
        partidas_2022_dynamic = load_openfootball_data(2022, team_to_id)
        if partidas_2022_dynamic:
            partidas_db.extend(partidas_2022_dynamic)
            print(f"Sucesso: {len(partidas_2022_dynamic)} partidas de 2022 carregadas dinamicamente.")
        else:
            raise ValueError("Lista vazia de partidas retornada.")
    except Exception as e:
        print(f"Falha ao carregar 2022 dinamicamente ({e}). Usando fallback offline.")
        group_match_counts_2022 = {}
        for data, mandante, visitante, gols_m, gols_v, fase_antiga, status in partidas_2022:
            if fase_antiga.startswith("Grupo "):
                grupo = fase_antiga.replace("Grupo ", "")
                count = group_match_counts_2022.get(grupo, 0) + 1
                group_match_counts_2022[grupo] = count
                if count <= 2:
                    fase = "1"
                elif count <= 4:
                    fase = "2"
                else:
                    fase = "3"
            else:
                fase = fase_antiga
                grupo = None

            if mandante in team_to_id and visitante in team_to_id:
                m_id = team_to_id[mandante]
                v_id = team_to_id[visitante]
                vencedor_id = None
                if gols_m is not None and gols_v is not None:
                    if gols_m > gols_v:
                        vencedor_id = m_id
                    elif gols_v > gols_m:
                        vencedor_id = v_id
                partidas_db.append((
                    2022,
                    data,
                    m_id,
                    v_id,
                    gols_m,
                    gols_v,
                    fase,
                    grupo,
                    status,
                    vencedor_id
                ))

    # Adicionar também algumas partidas marcantes da Copa 2018 para aumentar o histórico de gols e confrontos
    partidas_2018 = [
        # Fase Final 2018
        ("2018-06-30 11:00", "França", "Argentina", 4, 3, "Oitavas", "FINISHED"),
        ("2018-06-30 15:00", "Uruguai", "Portugal", 2, 1, "Oitavas", "FINISHED"),
        ("2018-06-15 15:00", "Portugal", "Espanha", 3, 3, "Grupo B", "FINISHED"),
        ("2018-07-02 11:00", "Brasil", "México", 2, 0, "Oitavas", "FINISHED"),
        ("2018-07-02 15:00", "Bélgica", "Japão", 3, 2, "Oitavas", "FINISHED"),
        ("2018-07-06 11:00", "Uruguai", "França", 0, 2, "Quartas", "FINISHED"),
        ("2018-07-06 15:00", "Brasil", "Bélgica", 1, 2, "Quartas", "FINISHED"),
        ("2018-07-07 11:00", "Suécia", "Inglaterra", 0, 2, "Quartas", "FINISHED"),
        ("2018-07-10 15:00", "França", "Bélgica", 1, 0, "Semifinal", "FINISHED"),
        ("2018-07-11 15:00", "Croácia", "Inglaterra", 2, 1, "Semifinal", "FINISHED"),
        ("2018-07-14 11:00", "Bélgica", "Inglaterra", 2, 0, "Disputa 3º Lugar", "FINISHED"),
        ("2018-07-15 12:00", "França", "Croácia", 4, 2, "Final", "FINISHED"),
    ]

    # Tentar carregar dados de 2018 dinamicamente
    try:
        print("Buscando dados da Copa 2018 do openfootball...")
        partidas_2018_dynamic = load_openfootball_data(2018, team_to_id)
        if partidas_2018_dynamic:
            partidas_db.extend(partidas_2018_dynamic)
            print(f"Sucesso: {len(partidas_2018_dynamic)} partidas de 2018 carregadas dinamicamente.")
        else:
            raise ValueError("Lista vazia de partidas retornada.")
    except Exception as e:
        print(f"Falha ao carregar 2018 dinamicamente ({e}). Usando fallback offline.")
        group_match_counts_2018 = {}
        for data, mandante, visitante, gols_m, gols_v, fase_antiga, status in partidas_2018:
            if fase_antiga.startswith("Grupo "):
                grupo = fase_antiga.replace("Grupo ", "")
                count = group_match_counts_2018.get(grupo, 0) + 1
                group_match_counts_2018[grupo] = count
                if count <= 2:
                    fase = "1"
                elif count <= 4:
                    fase = "2"
                else:
                    fase = "3"
            else:
                fase = fase_antiga
                grupo = None

            if mandante in team_to_id and visitante in team_to_id:
                m_id = team_to_id[mandante]
                v_id = team_to_id[visitante]
                vencedor_id = None
                if gols_m is not None and gols_v is not None:
                    if gols_m > gols_v:
                        vencedor_id = m_id
                    elif gols_v > gols_m:
                        vencedor_id = v_id
                partidas_db.append((
                    2018,
                    data,
                    m_id,
                    v_id,
                    gols_m,
                    gols_v,
                    fase,
                    grupo,
                    status,
                    vencedor_id
                ))

    # Adicionar partidas reais finalizadas da Copa 2026 (Matchday 1 & 2)
    partidas_2026_reais = [
        # Matchday 1
        ("2026-06-11 14:00", "México", "África do Sul", 2, 0, "Grupo A", "FINISHED"),
        ("2026-06-11 17:00", "Coreia do Sul", "Chéquia", 2, 1, "Grupo A", "FINISHED"),
        ("2026-06-12 14:00", "Canadá", "Bósnia e Herzegovina", 1, 1, "Grupo B", "FINISHED"),
        ("2026-06-12 17:00", "EUA", "Paraguai", 4, 1, "Grupo D", "FINISHED"),
        ("2026-06-13 11:00", "Catar", "Suíça", 1, 1, "Grupo B", "FINISHED"),
        ("2026-06-13 14:00", "Brasil", "Marrocos", 1, 1, "Grupo C", "FINISHED"),
        ("2026-06-13 17:00", "Escócia", "Haiti", 1, 0, "Grupo C", "FINISHED"),
        ("2026-06-13 20:00", "Austrália", "Turquia", 2, 0, "Grupo D", "FINISHED"),
        ("2026-06-14 11:00", "Alemanha", "Curaçao", 7, 1, "Grupo E", "FINISHED"),
        ("2026-06-14 14:00", "Holanda", "Japão", 2, 2, "Grupo F", "FINISHED"),
        ("2026-06-14 17:00", "Costa do Marfim", "Equador", 1, 0, "Grupo E", "FINISHED"),
        ("2026-06-14 20:00", "Suécia", "Tunísia", 5, 1, "Grupo F", "FINISHED"),
        ("2026-06-15 11:00", "Espanha", "Cabo Verde", 0, 0, "Grupo H", "FINISHED"),
        ("2026-06-15 14:00", "Bélgica", "Egito", 1, 1, "Grupo G", "FINISHED"),
        ("2026-06-15 17:00", "Arábia Saudita", "Uruguai", 1, 1, "Grupo H", "FINISHED"),
        ("2026-06-15 20:00", "Irã", "Nova Zelândia", 2, 2, "Grupo G", "FINISHED"),
        ("2026-06-16 14:00", "França", "Senegal", 3, 1, "Grupo I", "FINISHED"),
        ("2026-06-16 17:00", "Noruega", "Iraque", 4, 1, "Grupo I", "FINISHED"),
        ("2026-06-17 14:00", "Argentina", "Argélia", 3, 0, "Grupo J", "FINISHED"),
        ("2026-06-17 17:00", "Áustria", "Jordânia", 3, 1, "Grupo J", "FINISHED"),
        ("2026-06-17 20:00", "Portugal", "Colômbia", 2, 1, "Grupo K", "FINISHED"),
        ("2026-06-18 11:00", "RD Congo", "Uzbequistão", 0, 0, "Grupo K", "FINISHED"),
        ("2026-06-18 14:00", "Inglaterra", "Panamá", 3, 0, "Grupo L", "FINISHED"),
        ("2026-06-18 17:00", "Croácia", "Gana", 2, 1, "Grupo L", "FINISHED"),

        # Matchday 2
        ("2026-06-18 20:00", "Canadá", "Catar", 6, 0, "Grupo B", "FINISHED"),
        ("2026-06-19 14:00", "África do Sul", "Coreia do Sul", 0, 1, "Grupo A", "FINISHED"),
        ("2026-06-19 17:00", "Chéquia", "México", 1, 3, "Grupo A", "FINISHED"),
        ("2026-06-19 20:00", "Paraguai", "Austrália", 1, 2, "Grupo D", "FINISHED"),
        ("2026-06-20 11:00", "Haiti", "Brasil", 0, 3, "Grupo C", "FINISHED"),
        ("2026-06-20 14:00", "Marrocos", "Escócia", 3, 0, "Grupo C", "FINISHED"),
        ("2026-06-20 17:00", "Turquia", "EUA", 0, 2, "Grupo D", "FINISHED"),
        ("2026-06-20 20:00", "Alemanha", "Costa do Marfim", 2, 1, "Grupo E", "FINISHED"),
        ("2026-06-20 20:30", "Equador", "Curaçao", 0, 0, "Grupo E", "FINISHED"),
        ("2026-06-21 11:00", "Suíça", "Bósnia e Herzegovina", 4, 1, "Grupo B", "FINISHED"),
        ("2026-06-21 14:00", "Japão", "Tunísia", 4, 0, "Grupo F", "FINISHED"),
        ("2026-06-21 17:00", "Holanda", "Suécia", 5, 1, "Grupo F", "FINISHED"),
        ("2026-06-21 20:00", "Cabo Verde", "Uruguai", 2, 2, "Grupo H", "FINISHED"),
        ("2026-06-21 14:00", "Nova Zelândia", "Egito", 1, 3, "Grupo G", "FINISHED"),
        ("2026-06-21 17:00", "Bélgica", "Irã", 0, 0, "Grupo G", "FINISHED"),
        ("2026-06-21 20:00", "Espanha", "Arábia Saudita", 4, 0, "Grupo H", "FINISHED"),
        ("2026-06-22 13:00", "Argentina", "Áustria", 1, 3, "Grupo J", "FINISHED"),
        ("2026-06-22 16:00", "Jordânia", "Argélia", None, None, "Grupo J", "SCHEDULED"),
        ("2026-06-22 19:00", "França", "Iraque", None, None, "Grupo I", "SCHEDULED"),
        ("2026-06-22 22:00", "Senegal", "Noruega", None, None, "Grupo I", "SCHEDULED"),
        ("2026-06-23 14:00", "Portugal", "Uzbequistão", None, None, "Grupo K", "SCHEDULED"),
        ("2026-06-23 17:00", "Colômbia", "RD Congo", None, None, "Grupo K", "SCHEDULED"),
        ("2026-06-23 20:00", "Inglaterra", "Croácia", None, None, "Grupo L", "SCHEDULED"),
        ("2026-06-23 22:00", "Panamá", "Gana", None, None, "Grupo L", "SCHEDULED"),

        # Matchday 3
        ("2026-06-24 14:00", "México", "Coreia do Sul", None, None, "Grupo A", "SCHEDULED"),
        ("2026-06-24 14:00", "Chéquia", "África do Sul", None, None, "Grupo A", "SCHEDULED"),
        ("2026-06-24 17:00", "Suíça", "Canadá", None, None, "Grupo B", "SCHEDULED"),
        ("2026-06-24 17:00", "Bósnia e Herzegovina", "Catar", None, None, "Grupo B", "SCHEDULED"),
        ("2026-06-24 20:00", "Brasil", "Escócia", None, None, "Grupo C", "SCHEDULED"),
        ("2026-06-24 20:00", "Haiti", "Marrocos", None, None, "Grupo C", "SCHEDULED"),
        ("2026-06-24 22:00", "EUA", "Austrália", None, None, "Grupo D", "SCHEDULED"),
        ("2026-06-24 22:00", "Turquia", "Paraguai", None, None, "Grupo D", "SCHEDULED"),
        ("2026-06-25 14:00", "Equador", "Alemanha", None, None, "Grupo E", "SCHEDULED"),
        ("2026-06-25 14:00", "Curaçao", "Costa do Marfim", None, None, "Grupo E", "SCHEDULED"),
        ("2026-06-25 17:00", "Japão", "Suécia", None, None, "Grupo F", "SCHEDULED"),
        ("2026-06-25 17:00", "Tunísia", "Holanda", None, None, "Grupo F", "SCHEDULED"),
        ("2026-06-25 20:00", "Nova Zelândia", "Bélgica", None, None, "Grupo G", "SCHEDULED"),
        ("2026-06-25 20:00", "Egito", "Irã", None, None, "Grupo G", "SCHEDULED"),
        ("2026-06-25 22:00", "Uruguai", "Espanha", None, None, "Grupo H", "SCHEDULED"),
        ("2026-06-25 22:00", "Cabo Verde", "Arábia Saudita", None, None, "Grupo H", "SCHEDULED"),
        ("2026-06-26 14:00", "Noruega", "França", None, None, "Grupo I", "SCHEDULED"),
        ("2026-06-26 14:00", "Iraque", "Senegal", None, None, "Grupo I", "SCHEDULED"),
        ("2026-06-26 17:00", "Argélia", "Áustria", None, None, "Grupo J", "SCHEDULED"),
        ("2026-06-26 17:00", "Jordânia", "Argentina", None, None, "Grupo J", "SCHEDULED"),
        ("2026-06-27 14:00", "RD Congo", "Portugal", None, None, "Grupo K", "SCHEDULED"),
        ("2026-06-27 14:00", "Uzbequistão", "Colômbia", None, None, "Grupo K", "SCHEDULED"),
        ("2026-06-27 17:00", "Croácia", "Panamá", None, None, "Grupo L", "SCHEDULED"),
        ("2026-06-27 17:00", "Gana", "Inglaterra", None, None, "Grupo L", "SCHEDULED")
    ]

    group_match_counts_2026 = {}
    for data, mandante, visitante, gols_m, gols_v, fase_antiga, status in partidas_2026_reais:
        if fase_antiga.startswith("Grupo "):
            grupo = fase_antiga.replace("Grupo ", "")
            count = group_match_counts_2026.get(grupo, 0) + 1
            group_match_counts_2026[grupo] = count
            if count <= 2:
                fase = "1"
            elif count <= 4:
                fase = "2"
            else:
                fase = "3"
        else:
            fase = fase_antiga
            grupo = None

        if mandante in team_to_id and visitante in team_to_id:
            m_id = team_to_id[mandante]
            v_id = team_to_id[visitante]
            vencedor_id = None
            if gols_m is not None and gols_v is not None:
                if gols_m > gols_v:
                    vencedor_id = m_id
                elif gols_v > gols_m:
                    vencedor_id = v_id
            partidas_db.append((
                2026,
                data,
                m_id,
                v_id,
                gols_m,
                gols_v,
                fase,
                grupo,
                status,
                vencedor_id
            ))

    # Inserir partidas
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO partidas 
            (ano_copa, data_hora, mandante_id, visitante_id, gols_mandante, gols_visitante, fase, grupo, status, vencedor_id, origem_dados) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [tuple(partida) + ("seed",) for partida in partidas_db]
        )
        conn.commit()
        print("Banco de dados SQLite populado com sucesso!")
    finally:
        conn.close()

def load_historical_matches(include_seed_2026=False):
    """Retorna um DataFrame do Pandas com todas as partidas finalizadas."""
    conn = get_connection()
    try:
        query = """
            SELECT 
                p.id, p.ano_copa, p.data_hora, p.fase, p.grupo, p.status, p.origem_dados,
                p.gols_mandante, p.gols_visitante, p.vencedor_id,
                m.nome AS mandante_nome, m.sigla AS mandante_sigla, m.elo_rating AS mandante_elo,
                v.nome AS visitante_nome, v.sigla AS visitante_sigla, v.elo_rating AS visitante_elo
            FROM partidas p
            JOIN selecoes m ON p.mandante_id = m.id
            JOIN selecoes v ON p.visitante_id = v.id
            WHERE p.status = 'FINISHED'
              AND (? OR NOT (p.ano_copa = 2026 AND p.origem_dados = 'seed'))
        """
        df = pd.read_sql_query(query, conn, params=(include_seed_2026,))
        return df
    finally:
        conn.close()

def load_all_teams():
    """Retorna um DataFrame contendo todas as seleções."""
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM selecoes ORDER BY nome ASC", conn)
        return df
    finally:
        conn.close()

def load_matches_by_status(status):
    """Retorna partidas filtradas por status ('SCHEDULED', 'LIVE', 'FINISHED')."""
    if status not in VALID_MATCH_STATUSES:
        raise ValueError(f"Status inválido: {status}")
    conn = get_connection()
    try:
        query = """
            SELECT 
                p.id, p.ano_copa, p.data_hora, p.fase, p.grupo, p.status, p.origem_dados,
                p.gols_mandante, p.gols_visitante, p.vencedor_id,
                m.nome AS mandante_nome, m.sigla AS mandante_sigla, m.elo_rating AS mandante_elo,
                v.nome AS visitante_nome, v.sigla AS visitante_sigla, v.elo_rating AS visitante_elo
            FROM partidas p
            JOIN selecoes m ON p.mandante_id = m.id
            JOIN selecoes v ON p.visitante_id = v.id
            WHERE p.status = ?
        """
        df = pd.read_sql_query(query, conn, params=(status,))
        return df
    finally:
        conn.close()

def load_2026_matches():
    """Retorna todas as partidas da Copa de 2026 cadastradas no banco."""
    conn = get_connection()
    try:
        query = """
            SELECT 
                p.id, p.ano_copa, p.data_hora, p.fase, p.grupo, p.status, p.origem_dados,
                p.gols_mandante, p.gols_visitante, p.vencedor_id,
                m.nome AS mandante_nome, m.sigla AS mandante_sigla, m.elo_rating AS mandante_elo,
                v.nome AS visitante_nome, v.sigla AS visitante_sigla, v.elo_rating AS visitante_elo
            FROM partidas p
            JOIN selecoes m ON p.mandante_id = m.id
            JOIN selecoes v ON p.visitante_id = v.id
            WHERE p.ano_copa = 2026
            ORDER BY p.data_hora ASC
        """
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def save_prediction_snapshot(partida_id, prediction, modelo_versao="calibrated-v1"):
    """Salva uma previsao pre-jogo uma unica vez por partida e versao do modelo."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, status FROM partidas WHERE id = ?",
            (partida_id,),
        )
        match_row = cursor.fetchone()
        if not match_row or match_row["status"] == "FINISHED":
            return None

        cursor.execute(
            """
            SELECT id FROM previsoes_partidas
            WHERE partida_id = ? AND modelo_versao = ?
            """,
            (partida_id, modelo_versao),
        )
        existing = cursor.fetchone()
        if existing:
            return existing["id"]

        placar = prediction.get("placar_mais_provavel", (0, 0, 0.0))
        cursor.execute(
            """
            INSERT INTO previsoes_partidas
            (partida_id, modelo_versao, created_at, xg_mandante, xg_visitante,
             prob_mandante, prob_empate, prob_visitante, prev_gols_mandante, prev_gols_visitante)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                partida_id,
                modelo_versao,
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                float(prediction["xG_mandante"]),
                float(prediction["xG_visitante"]),
                float(prediction["prob_vitoria_mandante"]),
                float(prediction["prob_empate"]),
                float(prediction["prob_vitoria_visitante"]),
                int(placar[0]),
                int(placar[1]),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def _outcome_index(gols_m, gols_v):
    if gols_m > gols_v:
        return 0
    if gols_m == gols_v:
        return 1
    return 2

def _brier_score(prob_m, prob_e, prob_v, actual_idx):
    probs = [float(prob_m), float(prob_e), float(prob_v)]
    return sum((prob - (1.0 if idx == actual_idx else 0.0)) ** 2 for idx, prob in enumerate(probs))

def evaluate_finished_predictions():
    """Avalia snapshots de previsao quando a partida real ja terminou."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        rows = cursor.execute(
            """
            SELECT
                pp.id, pp.prob_mandante, pp.prob_empate, pp.prob_visitante,
                pp.prev_gols_mandante, pp.prev_gols_visitante,
                p.gols_mandante, p.gols_visitante
            FROM previsoes_partidas pp
            JOIN partidas p ON p.id = pp.partida_id
            WHERE pp.evaluated_at IS NULL
              AND p.status = 'FINISHED'
              AND p.origem_dados != 'seed'
              AND p.gols_mandante IS NOT NULL
              AND p.gols_visitante IS NOT NULL
            """
        ).fetchall()

        evaluated = 0
        for row in rows:
            actual_idx = _outcome_index(row["gols_mandante"], row["gols_visitante"])
            predicted_idx = int(
                max(
                    range(3),
                    key=lambda idx: [row["prob_mandante"], row["prob_empate"], row["prob_visitante"]][idx],
                )
            )
            score_exact = (
                int(row["prev_gols_mandante"]) == int(row["gols_mandante"])
                and int(row["prev_gols_visitante"]) == int(row["gols_visitante"])
            )
            goal_error = abs(int(row["gols_mandante"]) - int(row["prev_gols_mandante"])) + abs(
                int(row["gols_visitante"]) - int(row["prev_gols_visitante"])
            )
            cursor.execute(
                """
                UPDATE previsoes_partidas
                SET gols_mandante_real = ?, gols_visitante_real = ?, outcome_correct = ?,
                    score_exact = ?, goal_error = ?, brier_score = ?, evaluated_at = ?
                WHERE id = ?
                """,
                (
                    int(row["gols_mandante"]),
                    int(row["gols_visitante"]),
                    1 if predicted_idx == actual_idx else 0,
                    1 if score_exact else 0,
                    float(goal_error),
                    float(_brier_score(row["prob_mandante"], row["prob_empate"], row["prob_visitante"], actual_idx)),
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    row["id"],
                ),
            )
            evaluated += 1

        conn.commit()
        return evaluated
    finally:
        conn.close()

def load_prediction_evaluations():
    """Retorna snapshots e metricas de previsoes pre-jogo."""
    conn = get_connection()
    try:
        query = """
            SELECT
                pp.*,
                p.ano_copa, p.data_hora, p.fase, p.grupo, p.status, p.origem_dados,
                m.nome AS mandante_nome, v.nome AS visitante_nome
            FROM previsoes_partidas pp
            JOIN partidas p ON p.id = pp.partida_id
            JOIN selecoes m ON m.id = p.mandante_id
            JOIN selecoes v ON v.id = p.visitante_id
            ORDER BY p.data_hora ASC, pp.created_at ASC
        """
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()

def sync_openfootball_finished_matches(ano=2026):
    """
    Sincroniza partidas finalizadas do openfootball/worldcup.json.
    Jogos sem placar final sao ignorados; agendas futuras ficam para Football-Data.org.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT nome, id FROM selecoes")
        team_to_id = {row["nome"]: row["id"] for row in cursor.fetchall()}
    finally:
        conn.close()

    try:
        partidas = load_openfootball_data(ano, team_to_id, finished_only=True)
    except Exception as exc:
        return 0, f"erro: {str(exc)[:120]}"

    updated = 0
    conn = get_connection()
    try:
        cursor = conn.cursor()
        for partida in partidas:
            (
                ano_copa,
                data_hora,
                mandante_id,
                visitante_id,
                gols_m,
                gols_v,
                fase,
                grupo,
                status,
                vencedor_id,
            ) = partida

            if grupo:
                cursor.execute(
                    """
                    SELECT id, mandante_id
                    FROM partidas
                    WHERE ano_copa = ? AND grupo = ? AND (
                        (mandante_id = ? AND visitante_id = ?) OR
                        (mandante_id = ? AND visitante_id = ?)
                    )
                    ORDER BY CASE origem_dados WHEN 'seed' THEN 0 ELSE 1 END, id
                    """,
                    (ano_copa, grupo, mandante_id, visitante_id, visitante_id, mandante_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, mandante_id
                    FROM partidas
                    WHERE ano_copa = ? AND fase = ? AND (
                        (mandante_id = ? AND visitante_id = ?) OR
                        (mandante_id = ? AND visitante_id = ?)
                    )
                    ORDER BY CASE origem_dados WHEN 'seed' THEN 0 ELSE 1 END, id
                    """,
                    (ano_copa, fase, mandante_id, visitante_id, visitante_id, mandante_id),
                )
            match_rows = cursor.fetchall()
            match_row = match_rows[0] if match_rows else None

            if match_row:
                cursor.execute(
                    """
                    UPDATE partidas
                    SET mandante_id = ?, visitante_id = ?, gols_mandante = ?, gols_visitante = ?,
                        status = ?, fase = ?, grupo = ?, data_hora = ?, vencedor_id = ?,
                        origem_dados = 'openfootball'
                    WHERE id = ?
                    """,
                    (
                        mandante_id,
                        visitante_id,
                        gols_m,
                        gols_v,
                        status,
                        fase,
                        grupo,
                        data_hora,
                        vencedor_id,
                        match_row["id"],
                    ),
                )
                duplicate_ids = [row["id"] for row in match_rows[1:]]
                if duplicate_ids:
                    cursor.executemany(
                        "DELETE FROM partidas WHERE id = ?",
                        [(duplicate_id,) for duplicate_id in duplicate_ids],
                    )
            else:
                cursor.execute(
                    """
                    INSERT INTO partidas
                    (ano_copa, data_hora, mandante_id, visitante_id, gols_mandante, gols_visitante,
                     fase, grupo, status, vencedor_id, origem_dados)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'openfootball')
                    """,
                    partida,
                )
            updated += 1
        conn.commit()
    finally:
        conn.close()

    return updated, "ok"

def sync_api_match_to_db(game):
    """
    Sincroniza uma partida vinda da API com o banco de dados local.
    Se a partida correspondente for encontrada (em qualquer ordem de mandante/visitante), 
    atualiza a ordem dos times, o placar, a data, o status e o vencedor.
    """
    m_name = game.get("mandante_nome", "TBD")
    v_name = game.get("visitante_nome", "TBD")
    ano = game.get("ano_copa", 2026)
    gols_m = game.get("gols_mandante")
    gols_v = game.get("gols_visitante")
    status = game.get("status", "SCHEDULED")
    fase = game.get("fase", "Grupo")
    grupo = game.get("grupo")
    data_hora = game.get("data_hora", "")
    vencedor_nome = game.get("vencedor_nome")
    
    if m_name == "TBD" or v_name == "TBD":
        return None
        
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Encontrar os IDs das seleções
        cursor.execute("SELECT id FROM selecoes WHERE nome = ?", (m_name,))
        row_m = cursor.fetchone()
        cursor.execute("SELECT id FROM selecoes WHERE nome = ?", (v_name,))
        row_v = cursor.fetchone()
        
        if row_m and row_v:
            m_id = row_m["id"]
            v_id = row_v["id"]
            
            # Encontrar vencedor_id
            vencedor_id = None
            if vencedor_nome:
                cursor.execute("SELECT id FROM selecoes WHERE nome = ?", (vencedor_nome,))
                row_w = cursor.fetchone()
                if row_w:
                    vencedor_id = row_w["id"]
            elif gols_m is not None and gols_v is not None:
                # Fallback se a API não especificou vencedor mas tem saldo
                if gols_m > gols_v:
                    vencedor_id = m_id
                elif gols_v > gols_m:
                    vencedor_id = v_id
            
            # Procurar se a partida já existe no banco (incluindo a fase para evitar problemas de remates)
            cursor.execute(
                """
                SELECT id, mandante_id, visitante_id, gols_mandante, gols_visitante, status, data_hora, vencedor_id
                FROM partidas 
                WHERE ano_copa = ? AND fase = ? AND (
                    (mandante_id = ? AND visitante_id = ?) OR
                    (mandante_id = ? AND visitante_id = ?)
                )
                """,
                (ano, fase, m_id, v_id, v_id, m_id)
            )
            match_row = cursor.fetchone()
            
            if match_row:
                p_id = match_row["id"]
                db_m_id = match_row["mandante_id"]
                db_gols_m = match_row["gols_mandante"]
                db_gols_v = match_row["gols_visitante"]
                db_status = match_row["status"]
                db_data_hora = match_row["data_hora"]
                db_vencedor_id = match_row["vencedor_id"]
                
                # Se a ordem dos times estiver invertida em relação à API
                if db_m_id != m_id:
                    cursor.execute(
                        """
                        UPDATE partidas 
                        SET mandante_id = ?, visitante_id = ?, gols_mandante = ?, gols_visitante = ?, status = ?, fase = ?, grupo = ?, data_hora = ?, vencedor_id = ?, origem_dados = 'api'
                        WHERE id = ?
                        """,
                        (m_id, v_id, gols_m, gols_v, status, fase, grupo, data_hora, vencedor_id, p_id)
                    )
                    conn.commit()
                    return p_id
                else:
                    # Ordem correta, atualiza se houver qualquer diferença
                    if (db_status != status or db_gols_m != gols_m or 
                        db_gols_v != gols_v or db_data_hora != data_hora or db_vencedor_id != vencedor_id):
                        cursor.execute(
                            """
                            UPDATE partidas 
                        SET gols_mandante = ?, gols_visitante = ?, status = ?, fase = ?, grupo = ?, data_hora = ?, vencedor_id = ?, origem_dados = 'api'
                            WHERE id = ?
                            """,
                            (gols_m, gols_v, status, fase, grupo, data_hora, vencedor_id, p_id)
                        )
                        conn.commit()
                    return p_id
            else:
                # Caso não exista, insere no banco
                cursor.execute(
                    """
                    INSERT INTO partidas (ano_copa, data_hora, mandante_id, visitante_id, gols_mandante, gols_visitante, fase, grupo, status, vencedor_id, origem_dados)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'api')
                    """,
                    (ano, data_hora, m_id, v_id, gols_m, gols_v, fase, grupo, status, vencedor_id)
                )
                conn.commit()
                return cursor.lastrowid
    finally:
        conn.close()

    return None

def update_live_match(partida_id, gols_m, gols_v, status):
    """Atualiza o placar, o status e define o vencedor para jogos finalizados."""
    if (gols_m is not None and gols_m < 0) or (gols_v is not None and gols_v < 0):
        raise ValueError("Gols não podem ser negativos.")
    if status not in VALID_MATCH_STATUSES:
        raise ValueError(f"Status inválido: {status}")
        
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Obter dados da partida para saber os times
        cursor.execute("SELECT mandante_id, visitante_id, fase FROM partidas WHERE id = ?", (partida_id,))
        row = cursor.fetchone()
        
        vencedor_id = None
        if row and status == "FINISHED" and gols_m is not None and gols_v is not None:
            if gols_m > gols_v:
                vencedor_id = row["mandante_id"]
            elif gols_v > gols_m:
                vencedor_id = row["visitante_id"]
                
        cursor.execute(
            """
            UPDATE partidas 
            SET gols_mandante = ?, gols_visitante = ?, status = ?, vencedor_id = ?
            WHERE id = ?
            """,
            (gols_m, gols_v, status, vencedor_id, partida_id)
        )
        conn.commit()
    finally:
        conn.close()
    if status == "FINISHED":
        evaluate_finished_predictions()

def clear_2026_matches():
    """Remove todas as partidas de 2026 do banco para evitar duplicações com a API."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM partidas WHERE ano_copa = 2026")
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Old database removed.")
    init_db()
