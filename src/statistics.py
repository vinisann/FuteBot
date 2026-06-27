import pandas as pd


TEAM_STATS_COLUMNS = [
    "Seleção",
    "Sigla",
    "Ranking FIFA",
    "ELO Rating",
    "Jogos",
    "Vitórias",
    "Empates",
    "Derrotas",
    "Gols Pró",
    "Gols Contra",
    "Saldo de Gols",
    "Ataque (Fator)",
    "Defesa (Fator)",
]


def build_team_stats(df_matches_filtered, df_teams, ataque, defesa):
    stats_rows = []
    for _, team_row in df_teams.iterrows():
        team_name = team_row["nome"]
        sigla = team_row["sigla"]
        elo = team_row["elo_rating"]
        fifa = team_row["ranking_fifa"]

        m_matches = df_matches_filtered[df_matches_filtered["mandante_nome"] == team_name]
        v_matches = df_matches_filtered[df_matches_filtered["visitante_nome"] == team_name]

        total_jogos = len(m_matches) + len(v_matches)
        if total_jogos == 0:
            continue

        gols_marcados = m_matches["gols_mandante"].sum() + v_matches["gols_visitante"].sum()
        gols_sofridos = m_matches["gols_visitante"].sum() + v_matches["gols_mandante"].sum()

        vitorias = (
            len(m_matches[m_matches["gols_mandante"] > m_matches["gols_visitante"]])
            + len(v_matches[v_matches["gols_visitante"] > v_matches["gols_mandante"]])
        )
        empates = (
            len(m_matches[m_matches["gols_mandante"] == m_matches["gols_visitante"]])
            + len(v_matches[v_matches["gols_visitante"] == v_matches["gols_mandante"]])
        )
        derrotas = total_jogos - vitorias - empates

        stats_rows.append(
            {
                "Seleção": team_name,
                "Sigla": sigla,
                "Ranking FIFA": fifa,
                "ELO Rating": elo,
                "Jogos": total_jogos,
                "Vitórias": vitorias,
                "Empates": empates,
                "Derrotas": derrotas,
                "Gols Pró": gols_marcados,
                "Gols Contra": gols_sofridos,
                "Saldo de Gols": gols_marcados - gols_sofridos,
                "Ataque (Fator)": ataque.get(team_name, 1.0),
                "Defesa (Fator)": defesa.get(team_name, 1.0),
            }
        )

    return pd.DataFrame(stats_rows, columns=TEAM_STATS_COLUMNS)
