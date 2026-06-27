import sqlite3
import pandas as pd

def calculate_real_group_standings(df_2026, team_elo):
    df_group = df_2026[df_2026["fase"].isin(["1", "2", "3"])].copy()
    
    group_teams = {}
    for _, row in df_group.iterrows():
        g = row["grupo"]
        if not g:
            continue
        if g not in group_teams:
            group_teams[g] = set()
        group_teams[g].add(row["mandante_nome"])
        group_teams[g].add(row["visitante_nome"])
        
    standings = {}
    for g, teams in group_teams.items():
        group_rows = []
        for team in teams:
            df_team = df_group[
                ((df_group["mandante_nome"] == team) | (df_group["visitante_nome"] == team))
            ].copy()
            df_team = df_team.sort_values("data_hora")
            
            pj = 0
            vit = 0
            e = 0
            der = 0
            gm = 0
            gc = 0
            form = []
            
            for _, match in df_team.iterrows():
                status = match["status"]
                is_mandante = match["mandante_nome"] == team
                gols_pro = match["gols_mandante"] if is_mandante else match["gols_visitante"]
                gols_concedidos = match["gols_visitante"] if is_mandante else match["gols_mandante"]
                
                if status == "FINISHED":
                    pj += 1
                    gm += gols_pro
                    gc += gols_concedidos
                    
                    if gols_pro > gols_concedidos:
                        vit += 1
                        form.append("V")
                    elif gols_pro < gols_concedidos:
                        der += 1
                        form.append("D")
                    else:
                        e += 1
                        form.append("E")
                elif status == "SCHEDULED":
                    form.append("O")
                    
            while len(form) < 5:
                form.append("O")
                
            pts = 3 * vit + e
            sg = gm - gc
            elo = team_elo.get(team, 0)
            
            group_rows.append({
                "team": team,
                "Pts": pts,
                "PJ": pj,
                "VIT": vit,
                "E": e,
                "DER": der,
                "GM": gm,
                "GC": gc,
                "SG": sg,
                "elo": elo,
                "form": form[:5]
            })
            
        group_rows.sort(key=lambda x: (x["Pts"], x["SG"], x["GM"], x["elo"]), reverse=True)
        standings[g] = group_rows
        
    return standings

# Run test
from src.database import load_2026_matches, load_all_teams
df_teams = load_all_teams()
team_elo = {row["nome"]: row["elo_rating"] for _, row in df_teams.iterrows()}
df_2026 = load_2026_matches()
stds = calculate_real_group_standings(df_2026, team_elo)
print("Groups calculated:", sorted(stds.keys()))
print("Grupo A Standings:")
for rank, r in enumerate(stds["A"], 1):
    print(f"{rank}. {r['team']}: Pts={r['Pts']}, PJ={r['PJ']}, SG={r['SG']}, Form={r['form']}")
