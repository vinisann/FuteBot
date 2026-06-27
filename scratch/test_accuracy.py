import sqlite3
import pandas as pd
import numpy as np
import os
import sys

# Add src to python path to import project modules
sys.path.append(os.path.abspath('.'))

from src.database import load_historical_matches
from src.ML_models import predict_match_probabilities

def calculate_accuracy():
    conn = sqlite3.connect('data/futebot.db')
    query = """
    SELECT 
        p.ano_copa,
        p.data_hora,
        p.fase,
        p.grupo,
        p.gols_mandante,
        p.gols_visitante,
        sm.nome AS mandante_nome,
        sm.elo_rating AS mandante_elo,
        sv.nome AS visitante_nome,
        sv.elo_rating AS visitante_elo
    FROM partidas p
    JOIN selecoes sm ON p.mandante_id = sm.id
    JOIN selecoes sv ON p.visitante_id = sv.id
    WHERE p.status = 'FINISHED'
    """
    df_finished = pd.read_sql_query(query, conn)
    df_matches = load_historical_matches()
    
    total = len(df_finished)
    correct_outcomes = 0
    correct_scores = 0
    
    print(f"Total finished matches found: {total}")
    
    for idx, row in df_finished.iterrows():
        m_name = row["mandante_nome"]
        v_name = row["visitante_nome"]
        m_elo = row["mandante_elo"]
        v_elo = row["visitante_elo"]
        g_m = row["gols_mandante"]
        g_v = row["gols_visitante"]
        
        pred = predict_match_probabilities(m_name, v_name, m_elo, v_elo, df_matches)
        p_m = pred["prob_vitoria_mandante"]
        p_e = pred["prob_empate"]
        p_v = pred["prob_vitoria_visitante"]
        placar_prov = pred["placar_mais_provavel"]
        
        # Actual outcome
        if g_m > g_v:
            actual = "M"
        elif g_m == g_v:
            actual = "E"
        else:
            actual = "V"
            
        # Predicted outcome
        probs = [p_m, p_e, p_v]
        pred_idx = np.argmax(probs)
        if pred_idx == 0:
            predicted = "M"
        elif pred_idx == 1:
            predicted = "E"
        else:
            predicted = "V"
            
        # Exact score match
        actual_score = (int(g_m), int(g_v))
        predicted_score = (int(placar_prov[0]), int(placar_prov[1]))
        
        is_outcome_correct = (predicted == actual)
        is_score_correct = (predicted_score == actual_score)
        
        if is_outcome_correct:
            correct_outcomes += 1
        if is_score_correct:
            correct_scores += 1
            
    outcome_acc = (correct_outcomes / total) * 100 if total > 0 else 0
    score_acc = (correct_scores / total) * 100 if total > 0 else 0
    
    print(f"Winner Outcome (1X2) Accuracy: {outcome_acc:.2f}% ({correct_outcomes}/{total})")
    print(f"Exact Score Accuracy: {score_acc:.2f}% ({correct_scores}/{total})")

if __name__ == "__main__":
    calculate_accuracy()
