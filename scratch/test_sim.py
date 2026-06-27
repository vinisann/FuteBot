import sqlite3
import pandas as pd
import numpy as np
import random
from scipy.stats import poisson

# Import math functions from ML_models
from src.database import load_historical_matches, load_all_teams, load_2026_matches
from src.ML_models import calculate_team_strengths

def get_group_teams(df_matches_2026):
    group_teams = {}
    for _, row in df_matches_2026.iterrows():
        fase = row["fase"]
        if fase.startswith("Grupo"):
            m = row["mandante_nome"]
            v = row["visitante_nome"]
            group_teams.setdefault(fase, set()).add(m)
            group_teams.setdefault(fase, set()).add(v)
    return {g: list(teams) for g, teams in group_teams.items()}

def simulate_group_matches(df_2026, ataque, defesa, avg_m, avg_v, team_elo):
    # Copy matches to simulate
    simulated_matches = df_2026.copy()
    
    # We only simulate matches where status != 'FINISHED'
    for idx, row in simulated_matches.iterrows():
        if row["status"] != "FINISHED":
            m_name = row["mandante_nome"]
            v_name = row["visitante_nome"]
            m_elo = team_elo.get(m_name, 1850.0)
            v_elo = team_elo.get(v_name, 1850.0)
            
            f_ataque_m = ataque.get(m_name, 1.0)
            f_defesa_m = defesa.get(m_name, 1.0)
            f_ataque_v = ataque.get(v_name, 1.0)
            f_defesa_v = defesa.get(v_name, 1.0)
            
            lambda_m = f_ataque_m * f_defesa_v * avg_m
            lambda_v = f_ataque_v * f_defesa_m * avg_v
            
            elo_diff = m_elo - v_elo
            elo_adjustment = 1.15 ** (elo_diff / 100.0)
            elo_adjustment = np.clip(elo_adjustment, 0.4, 2.5)
            
            lambda_m = max(lambda_m * np.sqrt(elo_adjustment), 0.1)
            lambda_v = max(lambda_v / np.sqrt(elo_adjustment), 0.1)
            
            gols_m = np.random.poisson(lambda_m)
            gols_v = np.random.poisson(lambda_v)
            
            simulated_matches.at[idx, "gols_mandante"] = gols_m
            simulated_matches.at[idx, "gols_visitante"] = gols_v
            simulated_matches.at[idx, "status"] = "FINISHED"
            
    return simulated_matches

def get_group_standings(simulated_matches, group_teams, team_elo):
    standings = {}
    for group_name, teams in group_teams.items():
        group_standings = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "name": t, "elo": team_elo.get(t, 1850.0)} for t in teams}
        
        # Filter matches for this group
        g_matches = simulated_matches[simulated_matches["fase"] == group_name]
        for _, match in g_matches.iterrows():
            m = match["mandante_nome"]
            v = match["visitante_nome"]
            gm = match["gols_mandante"]
            gv = match["gols_visitante"]
            
            if gm is None or gv is None:
                continue
                
            group_standings[m]["gf"] += gm
            group_standings[m]["ga"] += gv
            group_standings[v]["gf"] += gv
            group_standings[v]["ga"] += gm
            
            if gm > gv:
                group_standings[m]["pts"] += 3
            elif gm < gv:
                group_standings[v]["pts"] += 3
            else:
                group_standings[m]["pts"] += 1
                group_standings[v]["pts"] += 1
                
        for t in teams:
            group_standings[t]["gd"] = group_standings[t]["gf"] - group_standings[t]["ga"]
            
        # Rank teams: points, gd, gf, elo
        sorted_teams = sorted(
            group_standings.values(),
            key=lambda x: (x["pts"], x["gd"], x["gf"], x["elo"]),
            reverse=True
        )
        standings[group_name] = sorted_teams
    return standings

def simulate_knockout_match(m_name, v_name, ataque, defesa, avg_m, avg_v, team_elo):
    m_elo = team_elo.get(m_name, 1850.0)
    v_elo = team_elo.get(v_name, 1850.0)
    
    f_ataque_m = ataque.get(m_name, 1.0)
    f_defesa_m = defesa.get(m_name, 1.0)
    f_ataque_v = ataque.get(v_name, 1.0)
    f_defesa_v = defesa.get(v_name, 1.0)
    
    lambda_m = f_ataque_m * f_defesa_v * avg_m
    lambda_v = f_ataque_v * f_defesa_m * avg_v
    
    elo_diff = m_elo - v_elo
    elo_adjustment = 1.15 ** (elo_diff / 100.0)
    elo_adjustment = np.clip(elo_adjustment, 0.4, 2.5)
    
    lambda_m = max(lambda_m * np.sqrt(elo_adjustment), 0.1)
    lambda_v = max(lambda_v / np.sqrt(elo_adjustment), 0.1)
    
    gols_m = np.random.poisson(lambda_m)
    gols_v = np.random.poisson(lambda_v)
    
    if gols_m > gols_v:
        return m_name
    elif gols_v > gols_m:
        return v_name
    else:
        # Penalty shootout simulation: ELO-weighted probability
        prob_m = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        return m_name if random.random() < prob_m else v_name

def run_tournament_simulation(df_matches, df_teams, iterations=500):
    df_2026 = load_2026_matches()
    ataque, defesa, avg_m, avg_v = calculate_team_strengths(df_matches)
    team_elo = {row["nome"]: row["elo_rating"] for _, row in df_teams.iterrows()}
    
    group_teams = get_group_teams(df_2026)
    
    # Track statistics
    # Round of 32, Round of 16, Quarterfinals, Semifinals, Finals, Winner
    results = {row["nome"]: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "F": 0, "W": 0} for _, row in df_teams.iterrows() if row["nome"] in team_elo}
    
    for _ in range(iterations):
        # 1. Group Stage
        sim_matches = simulate_group_matches(df_2026, ataque, defesa, avg_m, avg_v, team_elo)
        standings = get_group_standings(sim_matches, group_teams, team_elo)
        
        # Qualifiers
        qualified = [] # Will have 32 teams
        runners_up = []
        third_placed = []
        group_winners = []
        
        for g_name, ranked in standings.items():
            group_winners.append(ranked[0])
            runners_up.append(ranked[1])
            third_placed.append(ranked[2])
            
        # Sort winners, runners-up, third-placed to form seeds
        group_winners = sorted(group_winners, key=lambda x: (x["pts"], x["gd"], x["gf"], x["elo"]), reverse=True)
        runners_up = sorted(runners_up, key=lambda x: (x["pts"], x["gd"], x["gf"], x["elo"]), reverse=True)
        third_placed = sorted(third_placed, key=lambda x: (x["pts"], x["gd"], x["gf"], x["elo"]), reverse=True)
        
        # Best 8 third-placed
        best_third = third_placed[:8]
        
        # Seeds 1-32
        seeds = [x["name"] for x in group_winners] + [x["name"] for x in runners_up] + [x["name"] for x in best_third]
        
        for t in seeds:
            results[t]["R32"] += 1
            
        # 2. Round of 32
        r16_teams = []
        for i in range(16):
            t1 = seeds[i]
            t2 = seeds[31 - i]
            winner = simulate_knockout_match(t1, t2, ataque, defesa, avg_m, avg_v, team_elo)
            r16_teams.append(winner)
            results[winner]["R16"] += 1
            
        # 3. Round of 16 (Oitavas)
        qf_teams = []
        for i in range(8):
            t1 = r16_teams[i]
            t2 = r16_teams[15 - i]
            winner = simulate_knockout_match(t1, t2, ataque, defesa, avg_m, avg_v, team_elo)
            qf_teams.append(winner)
            results[winner]["QF"] += 1
            
        # 4. Quarterfinals (Quartas)
        sf_teams = []
        for i in range(4):
            t1 = qf_teams[i]
            t2 = qf_teams[7 - i]
            winner = simulate_knockout_match(t1, t2, ataque, defesa, avg_m, avg_v, team_elo)
            sf_teams.append(winner)
            results[winner]["SF"] += 1
            
        # 5. Semifinals
        f_teams = []
        for i in range(2):
            t1 = sf_teams[i]
            t2 = sf_teams[3 - i]
            winner = simulate_knockout_match(t1, t2, ataque, defesa, avg_m, avg_v, team_elo)
            f_teams.append(winner)
            results[winner]["F"] += 1
            
        # 6. Final
        champion = simulate_knockout_match(f_teams[0], f_teams[1], ataque, defesa, avg_m, avg_v, team_elo)
        results[champion]["W"] += 1
        
    # Convert results to percentages
    prob_df = []
    for team, stats in results.items():
        if team not in team_elo:
            continue
        prob_df.append({
            "Seleção": team,
            "Quartas (%)": (stats["QF"] / iterations) * 100,
            "Semis (%)": (stats["SF"] / iterations) * 100,
            "Finais (%)": (stats["F"] / iterations) * 100,
            "Campeão (%)": (stats["W"] / iterations) * 100
        })
        
    return pd.DataFrame(prob_df)

if __name__ == "__main__":
    df_matches = load_historical_matches()
    df_teams = load_all_teams()
    probs = run_tournament_simulation(df_matches, df_teams, iterations=200)
    print(probs.sort_values("Campeão (%)", ascending=False).head(10))
