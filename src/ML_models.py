import numpy as np
import pandas as pd
import random
from scipy.stats import poisson
from src.model_calibration import apply_calibration_to_lambdas
from src.dixon_coles import apply_dixon_coles_correction, estimate_dixon_coles_rho
from src.dynamic_elo import calculate_recent_form
from src.external_signals import apply_external_signal_adjustment

def calculate_team_strengths(df_matches):
    """
    Calcula os coeficientes de força de ataque e defesa de cada seleção 
    com base nas partidas históricas finalizadas.
    """
    if df_matches.empty:
        return {}, {}, 1.0, 1.0
        
    df_matches = df_matches.dropna(subset=['gols_mandante', 'gols_visitante'])
    if df_matches.empty:
        return {}, {}, 1.0, 1.0
        
    # Calcular médias gerais de gols marcados por mandantes e visitantes
    avg_gols_mandante = df_matches['gols_mandante'].mean()
    avg_gols_visitante = df_matches['gols_visitante'].mean()
    
    # Agrupar dados por time mandante
    mandantes_stats = df_matches.groupby('mandante_nome').agg(
        gols_marcados=('gols_mandante', 'sum'),
        gols_sofridos=('gols_visitante', 'sum'),
        jogos=('id', 'count')
    ).reset_index()
    
    # Agrupar dados por time visitante
    visitantes_stats = df_matches.groupby('visitante_nome').agg(
        gols_marcados=('gols_visitante', 'sum'),
        gols_sofridos=('gols_mandante', 'sum'),
        jogos=('id', 'count')
    ).reset_index()
    
    # Unificar as estatísticas de todas as seleções
    teams = list(set(df_matches['mandante_nome'].unique()) | set(df_matches['visitante_nome'].unique()))
    
    ataque = {}
    defesa = {}
    
    for team in teams:
        # Estatísticas como mandante
        m_row = mandantes_stats[mandantes_stats['mandante_nome'] == team]
        g_marcados_m = m_row['gols_marcados'].values[0] if not m_row.empty else 0
        g_sofridos_m = m_row['gols_sofridos'].values[0] if not m_row.empty else 0
        jogos_m = m_row['jogos'].values[0] if not m_row.empty else 0
        
        # Estatísticas como visitante
        v_row = visitantes_stats[visitantes_stats['visitante_nome'] == team]
        g_marcados_v = v_row['gols_marcados'].values[0] if not v_row.empty else 0
        g_sofridos_v = v_row['gols_sofridos'].values[0] if not v_row.empty else 0
        jogos_v = v_row['jogos'].values[0] if not v_row.empty else 0
        
        total_jogos = jogos_m + jogos_v
        if total_jogos == 0:
            ataque[team] = 1.0
            defesa[team] = 1.0
            continue
            
        # Calcular média de gols marcados e sofridos da seleção com suavização Laplaciana (Bayesiana)
        # Prior de 3 jogos com a média geral para evitar distorções de amostra pequena (ex: times com apenas 1 jogo)
        avg_geral = (avg_gols_mandante + avg_gols_visitante) / 2
        prior_gols = avg_geral * 3.0
        
        avg_marcados_team = (g_marcados_m + g_marcados_v + prior_gols) / (total_jogos + 3.0)
        avg_sofridos_team = (g_sofridos_m + g_sofridos_v + prior_gols) / (total_jogos + 3.0)
        
        ataque[team] = avg_marcados_team / avg_geral if avg_geral > 0 else 1.0
        defesa[team] = avg_sofridos_team / avg_geral if avg_geral > 0 else 1.0
        
    return ataque, defesa, avg_gols_mandante, avg_gols_visitante

def simulate_penalty_shootout(elo_diff):
    """Simula uma disputa de pênaltis realista baseada na diferença de ELO."""
    prob_m = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_m = 0.75  # Probabilidade média de conversão
    bias = (prob_m - 0.5) * 0.1
    prob_conv_m = np.clip(p_m + bias, 0.55, 0.9)
    prob_conv_v = np.clip(p_m - bias, 0.55, 0.9)
    
    goals_m = 0
    goals_v = 0
    # 5 cobradores regulamentares
    for i in range(5):
        if random.random() < prob_conv_m:
            goals_m += 1
        rem_v = 5 - i
        rem_m = 4 - i
        if goals_m > goals_v + rem_v or goals_v > goals_m + rem_m:
            break
            
        if random.random() < prob_conv_v:
            goals_v += 1
        rem_v = 4 - i
        if goals_m > goals_v + rem_v or goals_v > goals_m + rem_v:
            break
            
    # Morte súbita se empatar
    while goals_m == goals_v:
        m_conv = random.random() < prob_conv_m
        v_conv = random.random() < prob_conv_v
        if m_conv:
            goals_m += 1
        if v_conv:
            goals_v += 1
            
    return goals_m, goals_v

def simulate_extra_time_goals(m_elo, v_elo, base_lambda=0.38):
    """Simula gols da prorrogacao com ritmo menor que o tempo normal."""
    elo_diff = float(m_elo) - float(v_elo)
    elo_adjustment = 1.12 ** (elo_diff / 100.0)
    elo_adjustment = np.clip(elo_adjustment, 0.55, 1.90)
    lambda_m = max(float(base_lambda) * np.sqrt(elo_adjustment), 0.05)
    lambda_v = max(float(base_lambda) / np.sqrt(elo_adjustment), 0.05)
    return int(np.random.poisson(lambda_m)), int(np.random.poisson(lambda_v))

def resolve_knockout_after_full_time(m_name, v_name, gols_m, gols_v, m_elo, v_elo):
    """Resolve jogo eliminatorio com prorrogacao antes dos penaltis."""
    gols_m = int(gols_m)
    gols_v = int(gols_v)
    if gols_m > gols_v:
        return {
            "winner": m_name,
            "loser": v_name,
            "resolution": "normal_time",
            "full_time_score": (gols_m, gols_v),
            "extra_time_score": None,
            "final_score": (gols_m, gols_v),
            "penalty_score": None,
        }
    if gols_v > gols_m:
        return {
            "winner": v_name,
            "loser": m_name,
            "resolution": "normal_time",
            "full_time_score": (gols_m, gols_v),
            "extra_time_score": None,
            "final_score": (gols_m, gols_v),
            "penalty_score": None,
        }

    et_m, et_v = simulate_extra_time_goals(m_elo, v_elo)
    final_m = gols_m + et_m
    final_v = gols_v + et_v
    if final_m > final_v:
        return {
            "winner": m_name,
            "loser": v_name,
            "resolution": "extra_time",
            "full_time_score": (gols_m, gols_v),
            "extra_time_score": (et_m, et_v),
            "final_score": (final_m, final_v),
            "penalty_score": None,
        }
    if final_v > final_m:
        return {
            "winner": v_name,
            "loser": m_name,
            "resolution": "extra_time",
            "full_time_score": (gols_m, gols_v),
            "extra_time_score": (et_m, et_v),
            "final_score": (final_m, final_v),
            "penalty_score": None,
        }

    gp_m, gp_v = simulate_penalty_shootout(float(m_elo) - float(v_elo))
    return {
        "winner": m_name if gp_m > gp_v else v_name,
        "loser": v_name if gp_m > gp_v else m_name,
        "resolution": "penalties",
        "full_time_score": (gols_m, gols_v),
        "extra_time_score": (et_m, et_v),
        "final_score": (final_m, final_v),
        "penalty_score": (gp_m, gp_v),
    }

def format_knockout_score(result):
    """Formata placar de mata-mata com prorrogacao ou penaltis quando houver."""
    if result.get("final_score") is not None:
        final_m, final_v = result["final_score"]
    else:
        full_m, full_v = result.get("full_time_score", (0, 0))
        et_m, et_v = result.get("extra_time_score") or (0, 0)
        final_m, final_v = int(full_m) + int(et_m), int(full_v) + int(et_v)
    if result.get("resolution") == "penalties" and result.get("penalty_score"):
        gp_m, gp_v = result["penalty_score"]
        return f"{final_m} ({gp_m}) - ({gp_v}) {final_v}"
    if result.get("resolution") == "extra_time":
        return f"{final_m} - {final_v} a.p."
    return f"{final_m} - {final_v}"

def predict_match_probabilities(
    mandante_nome,
    visitante_nome,
    mandante_elo,
    visitante_elo,
    df_matches,
    max_gols=7,
    calibration=None,
    dynamic_elo=None,
    recent_form=None,
    score_correction=None,
    match_context=None,
    external_signals=None,
):
    """
    Calcula as probabilidades de resultado de uma partida (Vitória Mandante, Empate, Vitória Visitante)
    e gera a matriz de probabilidade de placares exatos usando a distribuição de Poisson e ELO.
    """
    # Coercions de segurança
    max_gols = max(1, int(max_gols))
    if dynamic_elo is not None:
        try:
            dyn_m, dyn_v = dynamic_elo
            if dyn_m is not None and dyn_v is not None:
                mandante_elo = dyn_m
                visitante_elo = dyn_v
        except (ValueError, TypeError):
            pass

    try:
        mandante_elo = float(mandante_elo) if mandante_elo is not None else 1850.0
    except (ValueError, TypeError):
        mandante_elo = 1850.0
    try:
        visitante_elo = float(visitante_elo) if visitante_elo is not None else 1850.0
    except (ValueError, TypeError):
        visitante_elo = 1850.0
        
    ataque, defesa, avg_g_mandante, avg_g_visitante = calculate_team_strengths(df_matches)
    
    # Obter forças das seleções (default 1.0 se não houver histórico)
    f_ataque_m = ataque.get(mandante_nome, 1.0)
    f_defesa_m = defesa.get(mandante_nome, 1.0)
    f_ataque_v = ataque.get(visitante_nome, 1.0)
    f_defesa_v = defesa.get(visitante_nome, 1.0)
    
    # 1. Expectativa básica de gols (Poisson pura)
    lambda_m = f_ataque_m * f_defesa_v * avg_g_mandante
    lambda_v = f_ataque_v * f_defesa_m * avg_g_visitante
    
    # 2. Ajuste pelo ELO Rating (Poisson Híbrida)
    # Diferença de ELO influencia na expectativa de gols esperados
    # 100 pontos de ELO de vantagem aumenta a expectativa de gols em ~15%
    elo_diff = mandante_elo - visitante_elo
    elo_adjustment = 1.15 ** (elo_diff / 100.0)
    
    # Limitar o ajuste para evitar bizarrices estatísticas
    elo_adjustment = np.clip(elo_adjustment, 0.4, 2.5)
    
    lambda_m = lambda_m * np.sqrt(elo_adjustment)
    lambda_v = lambda_v / np.sqrt(elo_adjustment)

    form_m = 1.0
    form_v = 1.0
    if recent_form:
        try:
            form_m = float(recent_form.get("mandante_factor", 1.0))
        except (ValueError, TypeError, AttributeError):
            form_m = 1.0
        try:
            form_v = float(recent_form.get("visitante_factor", 1.0))
        except (ValueError, TypeError, AttributeError):
            form_v = 1.0
        form_m = float(np.clip(form_m, 0.90, 1.10))
        form_v = float(np.clip(form_v, 0.90, 1.10))
        lambda_m *= form_m
        lambda_v *= form_v

    context_m = 1.0
    context_v = 1.0
    context_meta = {
        "modelo_com_contexto": False,
        "fator_contexto_mandante": 1.0,
        "fator_contexto_visitante": 1.0,
        "fator_descanso_mandante": 1.0,
        "fator_descanso_visitante": 1.0,
        "fator_clima": 1.0,
        "fator_fase": 1.0,
        "contexto_resumo": "Contexto neutro",
    }
    if match_context:
        try:
            context_m = float(match_context.get("fator_mandante", 1.0))
        except (ValueError, TypeError, AttributeError):
            context_m = 1.0
        try:
            context_v = float(match_context.get("fator_visitante", 1.0))
        except (ValueError, TypeError, AttributeError):
            context_v = 1.0
        context_m = float(np.clip(context_m, 0.92, 1.08))
        context_v = float(np.clip(context_v, 0.92, 1.08))
        lambda_m *= context_m
        lambda_v *= context_v
        context_meta = {
            "modelo_com_contexto": bool(match_context.get("modelo_com_contexto", True)),
            "fator_contexto_mandante": context_m,
            "fator_contexto_visitante": context_v,
            "fator_descanso_mandante": float(match_context.get("fator_descanso_mandante", 1.0)),
            "fator_descanso_visitante": float(match_context.get("fator_descanso_visitante", 1.0)),
            "fator_clima": float(match_context.get("fator_clima", 1.0)),
            "fator_fase": float(match_context.get("fator_fase", 1.0)),
            "contexto_resumo": str(match_context.get("contexto_resumo", "Contexto aplicado")),
        }

    lambda_m, lambda_v, calibration_meta = apply_calibration_to_lambdas(
        lambda_m, lambda_v, mandante_nome, visitante_nome, calibration
    )
    lambda_m, lambda_v, external_meta = apply_external_signal_adjustment(
        lambda_m, lambda_v, mandante_nome, visitante_nome, external_signals
    )
    
    # Evitar lambdas zerados para não quebrar a distribuição de Poisson
    lambda_m = max(lambda_m, 0.1)
    lambda_v = max(lambda_v, 0.1)
    
    # 3. Gerar distribuições de probabilidade de gols
    prob_gols_m = [poisson.pmf(i, lambda_m) for i in range(max_gols + 1)]
    prob_gols_v = [poisson.pmf(i, lambda_v) for i in range(max_gols + 1)]
    
    # 4. Criar a matriz conjunta de placar exato
    # Linhas = gols do mandante, Colunas = gols do visitante
    matriz_placar = np.outer(prob_gols_m, prob_gols_v)
    
    # Normalizar a matriz para que a soma seja 1.0
    matriz_placar = matriz_placar / matriz_placar.sum()
    score_correction_meta = {
        "modelo_dixon_coles": False,
        "rho_dixon_coles": 0.0,
        "ajuste_placares_baixos": False,
    }
    if score_correction and score_correction.get("method") == "dixon_coles":
        matriz_placar, score_correction_meta = apply_dixon_coles_correction(
            matriz_placar,
            lambda_m,
            lambda_v,
            score_correction.get("rho", -0.08),
        )
    
    # 5. Agregar probabilidades de resultado
    prob_vitoria_m = 0.0
    prob_empate = 0.0
    prob_vitoria_v = 0.0
    
    for m in range(max_gols + 1):
        for v in range(max_gols + 1):
            prob = matriz_placar[m, v]
            if m > v:
                prob_vitoria_m += prob
            elif m == v:
                prob_empate += prob
            else:
                prob_vitoria_v += prob

    fator_zebra = calibration_meta["fator_zebra"]
    if fator_zebra > 0:
        probs = [prob_vitoria_m, prob_empate, prob_vitoria_v]
        favorite_idx = int(np.argmax(probs))
        favorite_prob = probs[favorite_idx]
        shift = min(fator_zebra, max(0.0, favorite_prob - 0.34))
        if shift > 0:
            probs[favorite_idx] -= shift
            if favorite_idx == 1:
                probs[0] += shift / 2.0
                probs[2] += shift / 2.0
            else:
                underdog_idx = 2 if favorite_idx == 0 else 0
                probs[1] += shift * 0.4
                probs[underdog_idx] += shift * 0.6
            prob_vitoria_m, prob_empate, prob_vitoria_v = probs
                
    # Encontrar o placar mais provável
    m_idx, v_idx = np.unravel_index(np.argmax(matriz_placar), matriz_placar.shape)
    placar_mais_provavel = (int(m_idx), int(v_idx), float(matriz_placar[m_idx, v_idx]))
    
    return {
        "mandante": mandante_nome,
        "visitante": visitante_nome,
        "xG_mandante": float(lambda_m),
        "xG_visitante": float(lambda_v),
        "prob_vitoria_mandante": float(prob_vitoria_m),
        "prob_empate": float(prob_empate),
        "prob_vitoria_visitante": float(prob_vitoria_v),
        "placar_mais_provavel": placar_mais_provavel,
        "matriz_placar": matriz_placar.tolist(), # Convertido para lista JSON serializeable
        "gols_range": list(range(max_gols + 1)),
        "elo_mandante_usado": float(mandante_elo),
        "elo_visitante_usado": float(visitante_elo),
        "fator_forma_mandante": float(form_m),
        "fator_forma_visitante": float(form_v),
        "modelo_com_forma": bool(recent_form),
        **context_meta,
        **score_correction_meta,
        **calibration_meta,
        **external_meta,
    }

def simulate_match_in_play(prob_pre_jogo, tempo_atual, gols_m_atual, gols_v_atual, max_gols=7):
    """
    Recalcula probabilidades dinamicamente durante o jogo (In-Play)
    baseado nas chances pré-jogo e no andamento atual.
    """
    # Coercions de segurança para evitar valores negativos ou crashes
    tempo_atual = max(0, min(90, int(tempo_atual))) if tempo_atual is not None else 0
    gols_m_atual = max(0, int(gols_m_atual)) if gols_m_atual is not None else 0
    gols_v_atual = max(0, int(gols_v_atual)) if gols_v_atual is not None else 0
    max_gols = max(1, int(max_gols))
    
    if tempo_atual >= 90:
        # Se o jogo terminou, a probabilidade é 100% para o placar atual
        prob_m = 1.0 if gols_m_atual > gols_v_atual else 0.0
        prob_e = 1.0 if gols_m_atual == gols_v_atual else 0.0
        prob_v = 1.0 if gols_m_atual < gols_v_atual else 0.0
        return {
            "prob_vitoria_mandante": prob_m,
            "prob_empate": prob_e,
            "prob_vitoria_visitante": prob_v,
            "xG_restante_mandante": 0.0,
            "xG_restante_visitante": 0.0,
            "top_5_placares": [
                {"placar": f"{gols_m_atual} x {gols_v_atual}", "probabilidade": 1.0},
                {"placar": "-", "probabilidade": 0.0},
                {"placar": "-", "probabilidade": 0.0},
                {"placar": "-", "probabilidade": 0.0},
                {"placar": "-", "probabilidade": 0.0}
            ]
        }
        
    tempo_restante = 90 - tempo_atual
    proporcao_tempo = tempo_restante / 90.0
    
    # Expectativa de gols restantes proporcional ao tempo que falta
    lambda_m_restante = prob_pre_jogo["xG_mandante"] * proporcao_tempo
    lambda_v_restante = prob_pre_jogo["xG_visitante"] * proporcao_tempo
    
    # Distribuição de gols restantes
    prob_gols_m_restantes = [poisson.pmf(i, lambda_m_restante) for i in range(max_gols + 1)]
    prob_gols_v_restantes = [poisson.pmf(i, lambda_v_restante) for i in range(max_gols + 1)]
    
    matriz_restante = np.outer(prob_gols_m_restantes, prob_gols_v_restantes)
    matriz_restante = matriz_restante / matriz_restante.sum()
    
    prob_vitoria_m = 0.0
    prob_empate = 0.0
    prob_vitoria_v = 0.0
    
    score_probs = {}
    
    # Calcular probabilidade final somando os gols atuais com os que podem acontecer
    for m_extra in range(max_gols + 1):
        for v_extra in range(max_gols + 1):
            gols_m_finais = gols_m_atual + m_extra
            gols_v_finais = gols_v_atual + v_extra
            prob = matriz_restante[m_extra, v_extra]
            
            score_probs[(gols_m_finais, gols_v_finais)] = score_probs.get((gols_m_finais, gols_v_finais), 0.0) + prob
            
            if gols_m_finais > gols_v_finais:
                prob_vitoria_m += prob
            elif gols_m_finais == gols_v_finais:
                prob_empate += prob
            else:
                prob_vitoria_v += prob
                
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
    top_5_placares = []
    for score, prob in sorted_scores[:5]:
        top_5_placares.append({
            "placar": f"{score[0]} x {score[1]}",
            "probabilidade": float(prob)
        })
                
    return {
        "prob_vitoria_mandante": float(prob_vitoria_m),
        "prob_empate": float(prob_empate),
        "prob_vitoria_visitante": float(prob_vitoria_v),
        "xG_restante_mandante": float(lambda_m_restante),
        "xG_restante_visitante": float(lambda_v_restante),
        "top_5_placares": top_5_placares
    }

def get_group_teams(df_matches_2026):
    group_teams = {}
    for _, row in df_matches_2026.iterrows():
        grupo = row.get("grupo")
        if grupo is not None and grupo != "":
            m = row["mandante_nome"]
            v = row["visitante_nome"]
            group_teams.setdefault(grupo, set()).add(m)
            group_teams.setdefault(grupo, set()).add(v)
    return {g: list(teams) for g, teams in group_teams.items()}

def simulate_match_score_from_model(
    mandante_nome,
    visitante_nome,
    team_elo,
    model_history,
    calibration=None,
    score_correction=None,
    target_date=None,
    target_match_id=None,
):
    m_elo = team_elo.get(mandante_nome, 1850.0)
    v_elo = team_elo.get(visitante_nome, 1850.0)
    recent_form = None
    if target_date is not None and model_history is not None and not model_history.empty:
        recent_form = calculate_recent_form(
            model_history,
            mandante_nome,
            visitante_nome,
            target_date,
            target_match_id=target_match_id,
        )

    prediction = predict_match_probabilities(
        mandante_nome,
        visitante_nome,
        m_elo,
        v_elo,
        model_history,
        calibration=calibration,
        recent_form=recent_form,
        score_correction=score_correction,
    )
    matrix = np.array(prediction["matriz_placar"], dtype=float)
    probabilities = matrix.flatten()
    probabilities = probabilities / probabilities.sum()
    sampled_index = int(np.random.choice(len(probabilities), p=probabilities))
    goals_m, goals_v = np.unravel_index(sampled_index, matrix.shape)
    return int(goals_m), int(goals_v)


def simulate_group_matches(
    df_2026,
    ataque,
    defesa,
    avg_m,
    avg_v,
    team_elo,
    calibration=None,
    model_history=None,
    score_correction=None,
):
    simulated_matches = df_2026.copy()
    for idx, row in simulated_matches.iterrows():
        if row["status"] != "FINISHED":
            m_name = row["mandante_nome"]
            v_name = row["visitante_nome"]
            m_elo = team_elo.get(m_name, 1850.0)
            v_elo = team_elo.get(v_name, 1850.0)

            if model_history is not None and not model_history.empty:
                gols_m, gols_v = simulate_match_score_from_model(
                    m_name,
                    v_name,
                    team_elo,
                    model_history,
                    calibration=calibration,
                    score_correction=score_correction,
                    target_date=row.get("data_hora"),
                    target_match_id=row.get("id"),
                )
                simulated_matches.at[idx, "gols_mandante"] = gols_m
                simulated_matches.at[idx, "gols_visitante"] = gols_v
                simulated_matches.at[idx, "status"] = "FINISHED"
                continue
            
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
            lambda_m, lambda_v, _ = apply_calibration_to_lambdas(
                lambda_m, lambda_v, m_name, v_name, calibration
            )
            
            gols_m = np.random.poisson(lambda_m)
            gols_v = np.random.poisson(lambda_v)
            
            simulated_matches.at[idx, "gols_mandante"] = gols_m
            simulated_matches.at[idx, "gols_visitante"] = gols_v
            simulated_matches.at[idx, "status"] = "FINISHED"
    return simulated_matches

def get_group_standings(simulated_matches, group_teams, team_elo):
    import functools
    
    def compare_teams(a, b, matches_in_group):
        # 1. pts
        if a["pts"] != b["pts"]:
            return a["pts"] - b["pts"]
        # 2. gd
        if a["gd"] != b["gd"]:
            return a["gd"] - b["gd"]
        # 3. gf
        if a["gf"] != b["gf"]:
            return a["gf"] - b["gf"]
            
        # 4. Head-to-Head
        h2h_pts_a = 0
        h2h_pts_b = 0
        h2h_gf_a = 0
        h2h_gf_b = 0
        
        for _, match in matches_in_group.iterrows():
            m = match["mandante_nome"]
            v = match["visitante_nome"]
            gm = match["gols_mandante"]
            gv = match["gols_visitante"]
            if gm is None or gv is None or pd.isna(gm) or pd.isna(gv):
                continue
            if m == a["name"] and v == b["name"]:
                h2h_gf_a += gm
                h2h_gf_b += gv
                if gm > gv:
                    h2h_pts_a += 3
                elif gv > gm:
                    h2h_pts_b += 3
                else:
                    h2h_pts_a += 1
                    h2h_pts_b += 1
            elif m == b["name"] and v == a["name"]:
                h2h_gf_a += gv
                h2h_gf_b += gm
                if gv > gm:
                    h2h_pts_a += 3
                elif gm > gv:
                    h2h_pts_b += 3
                else:
                    h2h_pts_a += 1
                    h2h_pts_b += 1
                    
        h2h_gd_a = h2h_gf_a - h2h_gf_b
        h2h_gd_b = h2h_gf_b - h2h_gf_a
        
        if h2h_pts_a != h2h_pts_b:
            return h2h_pts_a - h2h_pts_b
        if h2h_gd_a != h2h_gd_b:
            return h2h_gd_a - h2h_gd_b
        if h2h_gf_a != h2h_gf_b:
            return h2h_gf_a - h2h_gf_b
            
        # 5. ELO fallback
        if a["elo"] != b["elo"]:
            return 1 if a["elo"] > b["elo"] else -1
        return 0

    standings = {}
    for group_name, teams in group_teams.items():
        group_standings = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "name": t, "elo": team_elo.get(t, 1850.0)} for t in teams}
        
        g_matches = simulated_matches[simulated_matches["grupo"] == group_name]
        for _, match in g_matches.iterrows():
            m = match["mandante_nome"]
            v = match["visitante_nome"]
            gm = match["gols_mandante"]
            gv = match["gols_visitante"]
            
            # Tratar explicitamente NaN e None
            if gm is None or gv is None or pd.isna(gm) or pd.isna(gv):
                continue
                
            gm = int(gm)
            gv = int(gv)
                
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
            
        sorted_teams = sorted(
            group_standings.values(),
            key=functools.cmp_to_key(lambda x, y: compare_teams(x, y, g_matches)),
            reverse=True
        )
        standings[group_name] = sorted_teams
    return standings

def simulate_knockout_match(
    m_name,
    v_name,
    ataque,
    defesa,
    avg_m,
    avg_v,
    team_elo,
    calibration=None,
    model_history=None,
    score_correction=None,
    return_details=False,
):
    # Guard contra times fantasma
    if not m_name or m_name == "TBD":
        result = {
            "winner": v_name,
            "loser": m_name,
            "resolution": "walkover",
            "full_time_score": (0, 0),
            "extra_time_score": None,
            "final_score": (0, 0),
            "penalty_score": None,
        }
        return result if return_details else v_name
    if not v_name or v_name == "TBD":
        result = {
            "winner": m_name,
            "loser": v_name,
            "resolution": "walkover",
            "full_time_score": (0, 0),
            "extra_time_score": None,
            "final_score": (0, 0),
            "penalty_score": None,
        }
        return result if return_details else m_name
        
    m_elo = team_elo.get(m_name, 1850.0)
    v_elo = team_elo.get(v_name, 1850.0)
    if model_history is not None and not model_history.empty:
        gols_m, gols_v = simulate_match_score_from_model(
            m_name,
            v_name,
            team_elo,
            model_history,
            calibration=calibration,
            score_correction=score_correction,
        )
        result = resolve_knockout_after_full_time(m_name, v_name, gols_m, gols_v, m_elo, v_elo)
        return result if return_details else result["winner"]
    
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
    lambda_m, lambda_v, _ = apply_calibration_to_lambdas(
        lambda_m, lambda_v, m_name, v_name, calibration
    )
    
    gols_m = np.random.poisson(lambda_m)
    gols_v = np.random.poisson(lambda_v)
    
    result = resolve_knockout_after_full_time(m_name, v_name, gols_m, gols_v, m_elo, v_elo)
    return result if return_details else result["winner"]

def run_tournament_simulation(df_matches, df_teams, df_2026, iterations=100, calibration=None):
    iterations = max(1, min(10000, int(iterations))) if iterations is not None else 100
    ataque, defesa, avg_m, avg_v = calculate_team_strengths(df_matches)
    team_elo = {row["nome"]: row["elo_rating"] for _, row in df_teams.iterrows()}
    score_correction = {"method": "dixon_coles", "rho": estimate_dixon_coles_rho(df_matches)}
    
    # Map team ID to name using df_teams
    team_id_to_name = {r["id"]: r["nome"] for _, r in df_teams.iterrows()}
    
    # Helper to determine winner of a finished database match
    def determine_db_winner(row):
        w_id = row.get("vencedor_id")
        if pd.notna(w_id) and w_id is not None:
            name = team_id_to_name.get(w_id)
            if name:
                return name
        g_m = row["gols_mandante"]
        g_v = row["gols_visitante"]
        t1 = row["mandante_nome"]
        t2 = row["visitante_nome"]
        if g_m is not None and g_v is not None:
            if g_m > g_v:
                return t1
            elif g_v > g_m:
                return t2
            else:
                current_phase = row["fase"]
                for _, next_row in df_2026.iterrows():
                    if next_row["fase"] not in ["1", "2", "3", current_phase]:
                        if next_row["mandante_nome"] in [t1, t2]:
                            return next_row["mandante_nome"]
                        if next_row["visitante_nome"] in [t1, t2]:
                            return next_row["visitante_nome"]
                return t1 if team_elo.get(t1, 0) > team_elo.get(t2, 0) else t2
        return None

    def play_ko(t1, t2):
        # Guard contra times fantasma
        if not t1 or t1 == "TBD":
            return t2
        if not t2 or t2 == "TBD":
            return t1
            
        m_elo = team_elo.get(t1, 1850.0)
        v_elo = team_elo.get(t2, 1850.0)
        if df_matches is not None and not df_matches.empty:
            gols_m, gols_v = simulate_match_score_from_model(
                t1,
                t2,
                team_elo,
                df_matches,
                calibration=calibration,
                score_correction=score_correction,
            )
            return resolve_knockout_after_full_time(t1, t2, gols_m, gols_v, m_elo, v_elo)["winner"]
        
        f_ataque_m = ataque.get(t1, 1.0)
        f_defesa_m = defesa.get(t1, 1.0)
        f_ataque_v = ataque.get(t2, 1.0)
        f_defesa_v = defesa.get(t2, 1.0)
        
        lambda_m = f_ataque_m * f_defesa_v * avg_m
        lambda_v = f_ataque_v * f_defesa_m * avg_v
        
        elo_diff = m_elo - v_elo
        elo_adjustment = 1.15 ** (elo_diff / 100.0)
        elo_adjustment = np.clip(elo_adjustment, 0.4, 2.5)
        
        lambda_m = max(lambda_m * np.sqrt(elo_adjustment), 0.1)
        lambda_v = max(lambda_v / np.sqrt(elo_adjustment), 0.1)
        lambda_m, lambda_v, _ = apply_calibration_to_lambdas(
            lambda_m, lambda_v, t1, t2, calibration
        )
        
        gols_m = np.random.poisson(lambda_m)
        gols_v = np.random.poisson(lambda_v)
        
        return resolve_knockout_after_full_time(t1, t2, gols_m, gols_v, m_elo, v_elo)["winner"]

    def get_real_phase_matches(stage):
        phases = PHASE_MAPPING[stage]
        df_stage = df_2026[df_2026["fase"].isin(phases)].copy()
        df_stage = df_stage[
            df_stage["mandante_nome"].isin(team_elo) & 
            df_stage["visitante_nome"].isin(team_elo)
        ]
        return df_stage.sort_values("data_hora")

    PHASE_MAPPING = {
        "R32": ["Fase de 32", "Dezesseis-avos", "Round of 32"],
        "R16": ["Oitavas de Final", "Oitavas", "Round of 16"],
        "QF": ["Quartas de Final", "Quartas", "Quarter-finals"],
        "SF": ["Semifinais", "Semifinal", "Semi-finals"],
        "F": ["Final"]
    }
    
    results = {row["nome"]: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "F": 0, "W": 0} for _, row in df_teams.iterrows() if row["nome"] in team_elo}
    
    # Pré-busca das fases do mata-mata para otimização de performance
    df_r32 = get_real_phase_matches("R32")
    df_r16 = get_real_phase_matches("R16")
    df_qf = get_real_phase_matches("QF")
    df_sf = get_real_phase_matches("SF")
    df_f = get_real_phase_matches("F")
    
    has_real_r32 = (len(df_r32) == 16)
    has_real_r16 = (len(df_r16) == 8)
    has_real_qf = (len(df_qf) == 4)
    has_real_sf = (len(df_sf) == 2)
    has_real_f = (len(df_f) == 1)
    
    group_teams = get_group_teams(df_2026)
    
    for _ in range(iterations):
        # 1. Round of 32
        r32_winners = {}
        if has_real_r32:
            r32_winners_list = []
            for _, row in df_r32.iterrows():
                t1 = row["mandante_nome"]
                t2 = row["visitante_nome"]
                if row["status"] == "FINISHED":
                    w = determine_db_winner(row)
                else:
                    w = play_ko(t1, t2)
                r32_winners_list.append(w)
                
            for _, row in df_r32.iterrows():
                t1, t2 = row["mandante_nome"], row["visitante_nome"]
                if t1 in results: results[t1]["R32"] += 1
                if t2 in results: results[t2]["R32"] += 1
                
            r32_winners = {73 + idx: w for idx, w in enumerate(r32_winners_list)}
        else:
            # Simula a Fase de Grupos
            sim_matches = simulate_group_matches(
                df_2026,
                ataque,
                defesa,
                avg_m,
                avg_v,
                team_elo,
                calibration=calibration,
                model_history=df_matches,
                score_correction=score_correction,
            )
            standings = get_group_standings(sim_matches, group_teams, team_elo)
            
            winners = {}
            runners = {}
            thirds = {}
            for g_name, ranked in standings.items():
                winners[g_name] = ranked[0]["name"]
                runners[g_name] = ranked[1]["name"]
                thirds[g_name] = {
                    "name": ranked[2]["name"], "group": g_name, "pts": ranked[2]["pts"],
                    "gd": ranked[2]["gd"], "gf": ranked[2]["gf"], "elo": ranked[2]["elo"]
                }
            sorted_thirds = sorted(thirds.values(), key=lambda x: (x["pts"], x["gd"], x["gf"], x["elo"]), reverse=True)
            best_thirds_list = sorted_thirds[:8]
            
            slots = ["E", "I", "A", "L", "D", "G", "B", "K"]
            assigned_thirds = [None] * 8
            remaining_thirds = list(best_thirds_list)
            for idx, winner_group in enumerate(slots):
                found = False
                for t_idx, t in enumerate(remaining_thirds):
                    if t["group"] != winner_group:
                        assigned_thirds[idx] = t["name"]
                        remaining_thirds.pop(t_idx)
                        found = True
                        break
                if not found and remaining_thirds:
                    assigned_thirds[idx] = remaining_thirds.pop(0)["name"]
                    
            r32_matches = [
                (runners["A"], runners["B"], 73),
                (winners["E"], assigned_thirds[0], 74),
                (winners["F"], runners["C"], 75),
                (winners["C"], runners["F"], 76),
                (winners["I"], assigned_thirds[1], 77),
                (runners["E"], runners["I"], 78),
                (winners["A"], assigned_thirds[2], 79),
                (winners["L"], assigned_thirds[3], 80),
                (winners["D"], assigned_thirds[4], 81),
                (winners["G"], assigned_thirds[5], 82),
                (runners["K"], runners["L"], 83),
                (winners["H"], runners["J"], 84),
                (winners["B"], assigned_thirds[6], 85),
                (winners["J"], runners["H"], 86),
                (winners["K"], assigned_thirds[7], 87),
                (runners["D"], runners["G"], 88)
            ]
            
            for t in list(winners.values()) + list(runners.values()) + [x["name"] for x in best_thirds_list]:
                if t in results:
                    results[t]["R32"] += 1
                    
            for t1, t2, m_num in r32_matches:
                r32_winners[m_num] = play_ko(t1, t2)

        # 2. Round of 16
        r16_winners = {}
        if has_real_r16:
            r16_winners_list = []
            for _, row in df_r16.iterrows():
                t1 = row["mandante_nome"]
                t2 = row["visitante_nome"]
                if row["status"] == "FINISHED":
                    w = determine_db_winner(row)
                else:
                    w = play_ko(t1, t2)
                r16_winners_list.append(w)
                
            for _, row in df_r16.iterrows():
                t1, t2 = row["mandante_nome"], row["visitante_nome"]
                if t1 in results: results[t1]["R16"] += 1
                if t2 in results: results[t2]["R16"] += 1
                
            r16_winners = {89 + idx: w for idx, w in enumerate(r16_winners_list)}
        else:
            for t in r32_winners.values():
                if t in results:
                    results[t]["R16"] += 1
                    
            r16_matches = [
                (r32_winners[74], r32_winners[77], 89),
                (r32_winners[73], r32_winners[75], 90),
                (r32_winners[76], r32_winners[78], 91),
                (r32_winners[79], r32_winners[80], 92),
                (r32_winners[83], r32_winners[84], 93),
                (r32_winners[81], r32_winners[82], 94),
                (r32_winners[86], r32_winners[88], 95),
                (r32_winners[85], r32_winners[87], 96)
            ]
            for t1, t2, m_num in r16_matches:
                r16_winners[m_num] = play_ko(t1, t2)

        # 3. Quarter-finals
        qf_winners = {}
        if has_real_qf:
            qf_winners_list = []
            for _, row in df_qf.iterrows():
                t1 = row["mandante_nome"]
                t2 = row["visitante_nome"]
                if row["status"] == "FINISHED":
                    w = determine_db_winner(row)
                else:
                    w = play_ko(t1, t2)
                qf_winners_list.append(w)
                
            for _, row in df_qf.iterrows():
                t1, t2 = row["mandante_nome"], row["visitante_nome"]
                if t1 in results: results[t1]["QF"] += 1
                if t2 in results: results[t2]["QF"] += 1
                
            qf_winners = {97 + idx: w for idx, w in enumerate(qf_winners_list)}
        else:
            for t in r16_winners.values():
                if t in results:
                    results[t]["QF"] += 1
                    
            qf_matches = [
                (r16_winners[89], r16_winners[90], 97),
                (r16_winners[93], r16_winners[94], 98),
                (r16_winners[91], r16_winners[92], 99),
                (r16_winners[95], r16_winners[96], 100)
            ]
            for t1, t2, m_num in qf_matches:
                qf_winners[m_num] = play_ko(t1, t2)

        # 4. Semifinals
        sf_winners = {}
        if has_real_sf:
            sf_winners_list = []
            for _, row in df_sf.iterrows():
                t1 = row["mandante_nome"]
                t2 = row["visitante_nome"]
                if row["status"] == "FINISHED":
                    w = determine_db_winner(row)
                else:
                    w = play_ko(t1, t2)
                sf_winners_list.append(w)
                
            for _, row in df_sf.iterrows():
                t1, t2 = row["mandante_nome"], row["visitante_nome"]
                if t1 in results: results[t1]["SF"] += 1
                if t2 in results: results[t2]["SF"] += 1
                
            sf_winners = {101 + idx: w for idx, w in enumerate(sf_winners_list)}
        else:
            for t in qf_winners.values():
                if t in results:
                    results[t]["SF"] += 1
                    
            sf_matches = [
                (qf_winners[97], qf_winners[98], 101),
                (qf_winners[99], qf_winners[100], 102)
            ]
            for t1, t2, m_num in sf_matches:
                sf_winners[m_num] = play_ko(t1, t2)

        # 5. Final
        if has_real_f:
            row = df_f.iloc[0]
            t1 = row["mandante_nome"]
            t2 = row["visitante_nome"]
            if row["status"] == "FINISHED":
                champion = determine_db_winner(row)
            else:
                champion = play_ko(t1, t2)
                
            if t1 in results: results[t1]["F"] += 1
            if t2 in results: results[t2]["F"] += 1
        else:
            for t in sf_winners.values():
                if t in results:
                    results[t]["F"] += 1
            t1 = sf_winners[101]
            t2 = sf_winners[102]
            champion = play_ko(t1, t2)
            
        if champion in results:
            results[champion]["W"] += 1
            
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

def simulate_single_bracket(df_matches, df_teams, df_2026, calibration=None):
    import random
    ataque, defesa, avg_m, avg_v = calculate_team_strengths(df_matches)
    team_elo = {row["nome"]: row["elo_rating"] for _, row in df_teams.iterrows()}
    score_correction = {"method": "dixon_coles", "rho": estimate_dixon_coles_rho(df_matches)}
    
    # Map team ID to name using df_teams
    team_id_to_name = {r["id"]: r["nome"] for _, r in df_teams.iterrows()}
    
    # Helper to determine winner of a finished database match
    def determine_db_winner(row):
        w_id = row.get("vencedor_id")
        if pd.notna(w_id) and w_id is not None:
            name = team_id_to_name.get(w_id)
            if name:
                return name
        g_m = row["gols_mandante"]
        g_v = row["gols_visitante"]
        t1 = row["mandante_nome"]
        t2 = row["visitante_nome"]
        if g_m is not None and g_v is not None:
            if g_m > g_v:
                return t1
            elif g_v > g_m:
                return t2
            else:
                current_phase = row["fase"]
                for _, next_row in df_2026.iterrows():
                    if next_row["fase"] not in ["1", "2", "3", current_phase]:
                        if next_row["mandante_nome"] in [t1, t2]:
                            return next_row["mandante_nome"]
                        if next_row["visitante_nome"] in [t1, t2]:
                            return next_row["visitante_nome"]
                return t1 if team_elo.get(t1, 0) > team_elo.get(t2, 0) else t2
        return None

    def play_ko(t1, t2):
        # Guard contra times fantasma
        if not t1 or t1 == "TBD":
            return t2, t1, "0 - 0"
        if not t2 or t2 == "TBD":
            return t1, t2, "0 - 0"
            
        m_elo = team_elo.get(t1, 1850.0)
        v_elo = team_elo.get(t2, 1850.0)
        if df_matches is not None and not df_matches.empty:
            gols_m, gols_v = simulate_match_score_from_model(
                t1,
                t2,
                team_elo,
                df_matches,
                calibration=calibration,
                score_correction=score_correction,
            )
            result = resolve_knockout_after_full_time(t1, t2, gols_m, gols_v, m_elo, v_elo)
            return result["winner"], result["loser"], format_knockout_score(result)
        
        f_ataque_m = ataque.get(t1, 1.0)
        f_defesa_m = defesa.get(t1, 1.0)
        f_ataque_v = ataque.get(t2, 1.0)
        f_defesa_v = defesa.get(t2, 1.0)
        
        lambda_m = f_ataque_m * f_defesa_v * avg_m
        lambda_v = f_ataque_v * f_defesa_m * avg_v
        
        elo_diff = m_elo - v_elo
        elo_adjustment = 1.15 ** (elo_diff / 100.0)
        elo_adjustment = np.clip(elo_adjustment, 0.4, 2.5)
        
        lambda_m = max(lambda_m * np.sqrt(elo_adjustment), 0.1)
        lambda_v = max(lambda_v / np.sqrt(elo_adjustment), 0.1)
        lambda_m, lambda_v, _ = apply_calibration_to_lambdas(
            lambda_m, lambda_v, t1, t2, calibration
        )
        
        gols_m = np.random.poisson(lambda_m)
        gols_v = np.random.poisson(lambda_v)
        
        result = resolve_knockout_after_full_time(t1, t2, gols_m, gols_v, m_elo, v_elo)
        return result["winner"], result["loser"], format_knockout_score(result)

    def get_real_phase_matches(stage):
        phases = PHASE_MAPPING[stage]
        df_stage = df_2026[df_2026["fase"].isin(phases)].copy()
        df_stage = df_stage[
            df_stage["mandante_nome"].isin(team_elo) & 
            df_stage["visitante_nome"].isin(team_elo)
        ]
        return df_stage.sort_values("data_hora")

    PHASE_MAPPING = {
        "R32": ["Fase de 32", "Dezesseis-avos", "Round of 32"],
        "R16": ["Oitavas de Final", "Oitavas", "Round of 16"],
        "QF": ["Quartas de Final", "Quartas", "Quarter-finals"],
        "SF": ["Semifinais", "Semifinal", "Semi-finals"],
        "F": ["Final"]
    }

    # 1. Round of 32
    df_r32 = get_real_phase_matches("R32")
    r32_results = []
    r32_winners_list = []
    
    if len(df_r32) == 16:
        for _, row in df_r32.iterrows():
            t1 = row["mandante_nome"]
            t2 = row["visitante_nome"]
            if row["status"] == "FINISHED":
                w = determine_db_winner(row)
                score_str = f"{row['gols_mandante']} - {row['gols_visitante']}"
            else:
                w, l, score_str = play_ko(t1, t2)
            r32_results.append({"t1": t1, "t2": t2, "w": w, "score": score_str})
            r32_winners_list.append(w)
        r32_winners = {73 + idx: w for idx, w in enumerate(r32_winners_list)}
    else:
        group_teams = get_group_teams(df_2026)
        sim_matches = simulate_group_matches(
            df_2026,
            ataque,
            defesa,
            avg_m,
            avg_v,
            team_elo,
            calibration=calibration,
            model_history=df_matches,
            score_correction=score_correction,
        )
        standings = get_group_standings(sim_matches, group_teams, team_elo)
        
        winners = {}
        runners = {}
        thirds = {}
        for g_name, ranked in standings.items():
            winners[g_name] = ranked[0]["name"]
            runners[g_name] = ranked[1]["name"]
            thirds[g_name] = {
                "name": ranked[2]["name"], "group": g_name, "pts": ranked[2]["pts"],
                "gd": ranked[2]["gd"], "gf": ranked[2]["gf"], "elo": ranked[2]["elo"]
            }
        sorted_thirds = sorted(thirds.values(), key=lambda x: (x["pts"], x["gd"], x["gf"], x["elo"]), reverse=True)
        best_thirds_list = sorted_thirds[:8]
        
        slots = ["E", "I", "A", "L", "D", "G", "B", "K"]
        assigned_thirds = [None] * 8
        remaining_thirds = list(best_thirds_list)
        for idx, winner_group in enumerate(slots):
            found = False
            for t_idx, t in enumerate(remaining_thirds):
                if t["group"] != winner_group:
                    assigned_thirds[idx] = t["name"]
                    remaining_thirds.pop(t_idx)
                    found = True
                    break
            if not found and remaining_thirds:
                assigned_thirds[idx] = remaining_thirds.pop(0)["name"]
                
        r32_matches = [
            (runners["A"], runners["B"], 73),
            (winners["E"], assigned_thirds[0], 74),
            (winners["F"], runners["C"], 75),
            (winners["C"], runners["F"], 76),
            (winners["I"], assigned_thirds[1], 77),
            (runners["E"], runners["I"], 78),
            (winners["A"], assigned_thirds[2], 79),
            (winners["L"], assigned_thirds[3], 80),
            (winners["D"], assigned_thirds[4], 81),
            (winners["G"], assigned_thirds[5], 82),
            (runners["K"], runners["L"], 83),
            (winners["H"], runners["J"], 84),
            (winners["B"], assigned_thirds[6], 85),
            (winners["J"], runners["H"], 86),
            (winners["K"], assigned_thirds[7], 87),
            (runners["D"], runners["G"], 88)
        ]
        
        r32_winners = {}
        for t1, t2, m_num in r32_matches:
            w, l, score = play_ko(t1, t2)
            r32_winners[m_num] = w
            r32_results.append({"t1": t1, "t2": t2, "w": w, "score": score})

    # 2. Round of 16
    df_r16 = get_real_phase_matches("R16")
    r16_results = []
    r16_winners_list = []
    
    if len(df_r16) == 8:
        for _, row in df_r16.iterrows():
            t1 = row["mandante_nome"]
            t2 = row["visitante_nome"]
            if row["status"] == "FINISHED":
                w = determine_db_winner(row)
                score_str = f"{row['gols_mandante']} - {row['gols_visitante']}"
            else:
                w, l, score_str = play_ko(t1, t2)
            r16_results.append({"t1": t1, "t2": t2, "w": w, "score": score_str})
            r16_winners_list.append(w)
        r16_winners = {89 + idx: w for idx, w in enumerate(r16_winners_list)}
    else:
        r16_winners = {}
        r16_matches = [
            (r32_winners[74], r32_winners[77], 89),
            (r32_winners[73], r32_winners[75], 90),
            (r32_winners[76], r32_winners[78], 91),
            (r32_winners[79], r32_winners[80], 92),
            (r32_winners[83], r32_winners[84], 93),
            (r32_winners[81], r32_winners[82], 94),
            (r32_winners[86], r32_winners[88], 95),
            (r32_winners[85], r32_winners[87], 96)
        ]
        for t1, t2, m_num in r16_matches:
            w, l, score = play_ko(t1, t2)
            r16_winners[m_num] = w
            r16_results.append({"t1": t1, "t2": t2, "w": w, "score": score})

    # 3. Quarter-finals
    df_qf = get_real_phase_matches("QF")
    qf_results = []
    qf_winners_list = []
    
    if len(df_qf) == 4:
        for _, row in df_qf.iterrows():
            t1 = row["mandante_nome"]
            t2 = row["visitante_nome"]
            if row["status"] == "FINISHED":
                w = determine_db_winner(row)
                score_str = f"{row['gols_mandante']} - {row['gols_visitante']}"
            else:
                w, l, score_str = play_ko(t1, t2)
            qf_results.append({"t1": t1, "t2": t2, "w": w, "score": score_str})
            qf_winners_list.append(w)
        qf_winners = {97 + idx: w for idx, w in enumerate(qf_winners_list)}
    else:
        qf_winners = {}
        qf_matches = [
            (r16_winners[89], r16_winners[90], 97),
            (r16_winners[93], r16_winners[94], 98),
            (r16_winners[91], r16_winners[92], 99),
            (r16_winners[95], r16_winners[96], 100)
        ]
        for t1, t2, m_num in qf_matches:
            w, l, score = play_ko(t1, t2)
            qf_winners[m_num] = w
            qf_results.append({"t1": t1, "t2": t2, "w": w, "score": score})

    # 4. Semifinals
    df_sf = get_real_phase_matches("SF")
    sf_results = []
    sf_winners_list = []
    
    if len(df_sf) == 2:
        for _, row in df_sf.iterrows():
            t1 = row["mandante_nome"]
            t2 = row["visitante_nome"]
            if row["status"] == "FINISHED":
                w = determine_db_winner(row)
                score_str = f"{row['gols_mandante']} - {row['gols_visitante']}"
            else:
                w, l, score_str = play_ko(t1, t2)
            sf_results.append({"t1": t1, "t2": t2, "w": w, "score": score_str})
            sf_winners_list.append(w)
        sf_winners = {101 + idx: w for idx, w in enumerate(sf_winners_list)}
    else:
        sf_winners = {}
        sf_matches = [
            (qf_winners[97], qf_winners[98], 101),
            (qf_winners[99], qf_winners[100], 102)
        ]
        for t1, t2, m_num in sf_matches:
            w, l, score = play_ko(t1, t2)
            sf_winners[m_num] = w
            sf_results.append({"t1": t1, "t2": t2, "w": w, "score": score})

    # 5. Final
    df_f = get_real_phase_matches("F")
    if len(df_f) == 1:
        row = df_f.iloc[0]
        t1 = row["mandante_nome"]
        t2 = row["visitante_nome"]
        if row["status"] == "FINISHED":
            champion = determine_db_winner(row)
            score_str = f"{row['gols_mandante']} - {row['gols_visitante']}"
        else:
            champion, l, score_str = play_ko(t1, t2)
        final_result = {"t1": t1, "t2": t2, "w": champion, "score": score_str}
    else:
        t1 = sf_winners[101]
        t2 = sf_winners[102]
        champion, l, score_str = play_ko(t1, t2)
        final_result = {"t1": t1, "t2": t2, "w": champion, "score": score_str}
        
    return {
        "R32": r32_results,
        "R16": r16_results,
        "QF": qf_results,
        "SF": sf_results,
        "F": final_result,
        "champion": champion
    }
