"""
FuteBot - Copa do Mundo 2026 Live
Página principal: Acompanhamento em tempo real das partidas.
"""

import streamlit as st
import numpy as np
import time
import random
import plotly.graph_objects as go
from datetime import datetime
import os

# Importações locais do projeto
from src.config import get_api_key
from src.database import VALID_MATCH_STATUSES, init_db, load_historical_matches, load_all_teams, update_live_match, sync_api_match_to_db, sync_openfootball_finished_matches, load_2026_matches
from src.ML_models import predict_match_probabilities, simulate_match_in_play
from src.api_client import fetch_live_matches_from_api, calculate_match_minute
from src.styles import inject_css
from src.utils import get_flag, format_fase, get_flag_html

# ============ CONFIGURAÇÃO DA PÁGINA (deve ser a primeira chamada st) ============
st.set_page_config(
    page_title="FuteBot - Copa do Mundo 2026 Live",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()
if "openfootball_sync_2026" not in st.session_state:
    st.session_state["openfootball_sync_2026"] = sync_openfootball_finished_matches(2026)

# Injetar CSS compartilhado (tema branco)
inject_css()

def clean_html(html_str):
    """Remove recuos e espaços vazios por linha para evitar falso-positivo de bloco de código no Streamlit Markdown."""
    return "\n".join(line.strip() for line in html_str.split("\n"))


# ============ FUNÇÕES DE RENDERIZAÇÃO ============

def generate_simulation_data(m_name, v_name, pred_pre):
    """Gera previamente os dados de simulação de 1000 partidas e escolhe a que atinge o placar alvo."""
    # 1. Obter o placar alvo da predição pré-jogo (moda analítica oficial)
    m_target = int(pred_pre["placar_mais_provavel"][0])
    v_target = int(pred_pre["placar_mais_provavel"][1])
    target_score = (m_target, v_target)

    # 2. Obter os Top 5 placares pré-jogo da matriz de probabilidade de Poisson
    matriz = np.array(pred_pre["matriz_placar"])
    score_probs = {}
    for m in range(matriz.shape[0]):
        for v in range(matriz.shape[1]):
            score_probs[(m, v)] = float(matriz[m, v])
            
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
    top_5_placares = []
    for score, prob in sorted_scores[:5]:
        top_5_placares.append({
            "placar": f"{score[0]} x {score[1]}",
            "probabilidade": float(prob)
        })

    # Pre-simulação de 1000 partidas em background para obter a linha do tempo (Opção 2)
    lambda_m_5 = (pred_pre["xG_mandante"] / 90.0) * 5
    lambda_v_5 = (pred_pre["xG_visitante"] / 90.0) * 5
    
    simulations = []
    score_counts = {}
    
    for _ in range(1000):
        g_m = 0
        g_v = 0
        timeline = []
        score_history = {0: (0, 0)}
        
        for minuto in range(5, 91, 5):
            g_m_add = np.random.poisson(lambda_m_5)
            g_v_add = np.random.poisson(lambda_v_5)
            
            if g_m_add > 0:
                g_m += g_m_add
                desc = (
                    f'<div style="display: flex; gap: 12px; margin-bottom: 12px; align-items: flex-start; text-align: left;">'
                    f'<div style="background: #22c55e; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; box-shadow: 0 4px 10px rgba(34, 197, 94, 0.25);">⚽</div>'
                    f'<div style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 10px 14px; border-radius: 10px; flex-grow: 1;">'
                    f'<strong style="color: #64748b; font-size: 11px; display: block; margin-bottom: 2px;">{minuto}\'</strong>'
                    f'<div style="color: #166534; font-size: 13px; font-weight: 600; font-family: \'Outfit\', sans-serif;">GOL do {m_name}! ({g_m} x {g_v})</div>'
                    f'</div>'
                    f'</div>'
                )
                timeline.append((minuto, "goal_mandante", desc))
            if g_v_add > 0:
                g_v += g_v_add
                desc = (
                    f'<div style="display: flex; gap: 12px; margin-bottom: 12px; align-items: flex-start; text-align: left;">'
                    f'<div style="background: #22c55e; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; box-shadow: 0 4px 10px rgba(34, 197, 94, 0.25);">⚽</div>'
                    f'<div style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 10px 14px; border-radius: 10px; flex-grow: 1;">'
                    f'<strong style="color: #64748b; font-size: 11px; display: block; margin-bottom: 2px;">{minuto}\'</strong>'
                    f'<div style="color: #166534; font-size: 13px; font-weight: 600; font-family: \'Outfit\', sans-serif;">GOL do {v_name}! ({g_m} x {g_v})</div>'
                    f'</div>'
                    f'</div>'
                )
                timeline.append((minuto, "goal_visitante", desc))
                
            if random.random() < 0.05:
                time_card = m_name if random.random() < 0.5 else v_name
                desc = (
                    f'<div style="display: flex; gap: 12px; margin-bottom: 12px; align-items: flex-start; text-align: left;">'
                    f'<div style="background: #eab308; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; box-shadow: 0 4px 10px rgba(234, 179, 8, 0.25);">🟨</div>'
                    f'<div style="background: #fffbeb; border: 1px solid #fef08a; padding: 10px 14px; border-radius: 10px; flex-grow: 1;">'
                    f'<strong style="color: #64748b; font-size: 11px; display: block; margin-bottom: 2px;">{minuto}\'</strong>'
                    f'<div style="color: #713f12; font-size: 13px; font-weight: 600; font-family: \'Outfit\', sans-serif;">Cartão Amarelo para {time_card}</div>'
                    f'</div>'
                    f'</div>'
                )
                timeline.append((minuto, "card", desc))
                
            score_history[minuto] = (g_m, g_v)
            
        final_score = (g_m, g_v)
        simulations.append({
            "score_history": score_history,
            "events": timeline,
            "final_score": final_score
        })
        score_counts[final_score] = score_counts.get(final_score, 0) + 1
        
    # Filtra as simulações que terminaram com o placar alvo da IA
    matching_sims = [sim for sim in simulations if sim["final_score"] == target_score]
    
    if not matching_sims:
        most_common_score = max(score_counts.keys(), key=lambda x: score_counts[x])
        matching_sims = [sim for sim in simulations if sim["final_score"] == most_common_score]
        
    chosen_sim = random.choice(matching_sims)
    
    return {
        "chosen_sim": chosen_sim,
        "top_5_placares": top_5_placares
    }


def render_match_card(game, df_matches, team_elo_map, team_sigla_map):
    """Renderiza o card de uma partida com base no seu status."""
    g_id = game["id"]
    m_name = game["mandante_nome"]
    v_name = game["visitante_nome"]
    m_sigla = game.get("mandante_sigla", team_sigla_map.get(m_name, "TBD"))
    v_sigla = game.get("visitante_sigla", team_sigla_map.get(v_name, "TBD"))

    m_elo = team_elo_map.get(m_name, 1850.0)
    v_elo = team_elo_map.get(v_name, 1850.0)

    fase = game["fase"]
    grupo = game.get("grupo")
    status = game["status"]
    gols_m = game["gols_mandante"]
    gols_v = game["gols_visitante"]
    data_hora = game["data_hora"]

    # Determinar label e estilo do badge
    if status == "LIVE":
        badge_html = '<span class="badge badge-live">⚡ AO VIVO</span>'
        card_border = "neon-border-live"
    elif status == "FINISHED":
        badge_html = '<span class="badge badge-finished">✔ ENCERRADO</span>'
        card_border = "neon-border-finished"
    elif status in ("POSTPONED", "CANCELLED", "SUSPENDED"):
        status_label = {
            "POSTPONED": "ADIADO",
            "CANCELLED": "CANCELADO",
            "SUSPENDED": "SUSPENSO",
        }.get(status, "INDISPONIVEL")
        badge_html = f'<span class="badge badge-scheduled">! {status_label}</span>'
        card_border = "neon-border-scheduled"
    else:
        badge_html = '<span class="badge badge-scheduled">📅 AGENDADO</span>'
        card_border = "neon-border-scheduled"

    # Determinar exibição do placar
    if status in ("SCHEDULED", "POSTPONED", "CANCELLED", "SUSPENDED"):
        score_html = '<div class="score-pending">vs</div>'
    else:
        gm = gols_m if gols_m is not None else 0
        gv = gols_v if gols_v is not None else 0
        score_html = f'<div class="score">{gm} - {gv}</div>'

    # Horário formatado
    hora_exibicao = data_hora.split(" ")[1] if " " in data_hora else data_hora
    fase_exibicao = format_fase(fase, grupo)

    with st.container():
        # Exibir o placar em HTML (usa largura total do block-container centralizado)
        # O flex-wrap nowrap e min-width 0 nos times evitam qualquer empilhamento vertical do placar
        st.markdown(clean_html(f"""
        <div class="glass-card {card_border}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="color: #64748b; font-size: 13px; font-weight: 500; font-family: 'Outfit', sans-serif;">
                    🏆 Copa do Mundo 2026 — {fase_exibicao} • {hora_exibicao}
                </span>
                {badge_html}
            </div>
            <div class="scoreboard" style="margin: 0; padding: 16px 20px; border-radius: 12px; display: flex; flex-wrap: nowrap; justify-content: space-between; align-items: center; gap: 15px;">
                <div class="team-name" style="font-size: 16px; min-width: 0; flex: 1; text-align: center; color: #1e293b; font-family: 'Outfit', sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    <div style="margin-bottom: 6px; display: flex; justify-content: center; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.12));">{get_flag_html(m_name, width=38)}</div>
                    <strong>{m_name}</strong>
                    <div style="font-size: 11px; color: #64748b; font-weight: 500; margin-top: 2px;">{m_sigla} • ELO {m_elo:.0f}</div>
                </div>
                <div style="flex-shrink: 0; min-width: 80px; text-align: center;">
                    {score_html}
                </div>
                <div class="team-name" style="font-size: 16px; min-width: 0; flex: 1; text-align: center; color: #1e293b; font-family: 'Outfit', sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    <div style="margin-bottom: 6px; display: flex; justify-content: center; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.12));">{get_flag_html(v_name, width=38)}</div>
                    <strong>{v_name}</strong>
                    <div style="font-size: 11px; color: #64748b; font-weight: 500; margin-top: 2px;">{v_sigla} • ELO {v_elo:.0f}</div>
                </div>
            </div>
        """), unsafe_allow_html=True)

        # ---- JOGO AO VIVO: Probabilidade dinâmica ----
        if status == "LIVE":
            pred_pre = predict_match_probabilities(m_name, v_name, m_elo, v_elo, df_matches)

            minuto = game.get("minuto")
            if minuto is None:
                minuto = calculate_match_minute(game.get("utc_date", ""))

            gols_m_val = gols_m if gols_m is not None else 0
            gols_v_val = gols_v if gols_v is not None else 0
            pred_live = simulate_match_in_play(pred_pre, minuto, gols_m_val, gols_v_val)

            p_m = pred_live["prob_vitoria_mandante"] * 100
            p_e = pred_live["prob_empate"] * 100
            p_v = pred_live["prob_vitoria_visitante"] * 100

            flag_m_html = get_flag_html(m_name, width=20)
            flag_v_html = get_flag_html(v_name, width=20)

            st.markdown(clean_html(f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); margin-top: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size:12px; color:#475569; font-weight:700; font-family: 'Outfit', sans-serif;">🤖 CHANCES DINÂMICAS DA IA ({minuto}')</span>
                    <span style="background:#dbeafe; color:#2563eb; font-size:10px; font-weight:700; padding:2px 6px; border-radius:8px; font-family: 'Outfit', sans-serif;">Modelo Live</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 13px; font-family: 'Outfit', sans-serif;">
                    <div style="display: flex; align-items: center; gap: 6px; font-weight: 600; color: #2563eb;">
                        {flag_m_html} {m_sigla} {p_m:.0f}%
                    </div>
                    <div style="font-weight: 600; color: #64748b;">
                        Empate {p_e:.0f}%
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px; font-weight: 600; color: #7c3aed; flex-direction: row-reverse;">
                        {flag_v_html} {v_sigla} {p_v:.0f}%
                    </div>
                </div>
                <div style="height: 18px; border-radius: 9px; display: flex; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.06); background-color: #f1f5f9; border: 1px solid #e2e8f0; position: relative;">
                    <div style="width: {p_m}%; background: linear-gradient(90deg, #2563eb, #3b82f6); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 11px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); transition: width 0.5s ease; min-width: { '30px' if p_m >= 5 else '0px' };">
                        {f"{p_m:.0f}%" if p_m >= 15 else ""}
                    </div>
                    <div style="width: {p_e}%; background: #94a3b8; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 11px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); transition: width 0.5s ease; min-width: { '30px' if p_e >= 5 else '0px' };">
                        {f"{p_e:.0f}%" if p_e >= 15 else ""}
                    </div>
                    <div style="width: {p_v}%; background: linear-gradient(90deg, #8b5cf6, #7c3aed); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 11px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); transition: width 0.5s ease; min-width: { '30px' if p_v >= 5 else '0px' };">
                        {f"{p_v:.0f}%" if p_v >= 15 else ""}
                    </div>
                </div>
            </div>
            """), unsafe_allow_html=True)

        # ---- JOGO AGENDADO: Previsão pré-jogo + botão de simulação ----
        elif status == "SCHEDULED":
            pred_pre = predict_match_probabilities(m_name, v_name, m_elo, v_elo, df_matches)

            p_m = pred_pre["prob_vitoria_mandante"] * 100
            p_e = pred_pre["prob_empate"] * 100
            p_v = pred_pre["prob_vitoria_visitante"] * 100
            placar_prov = pred_pre["placar_mais_provavel"]

            flag_m_html = get_flag_html(m_name, width=20)
            flag_v_html = get_flag_html(v_name, width=20)

            col_ia1, col_ia2 = st.columns([3, 1])
            with col_ia1:
                st.markdown(clean_html(f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); margin-top: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size:12px; color:#3b82f6; font-weight:700; font-family: 'Outfit', sans-serif;">🔮 PREVISÃO DA IA (PRÉ-JOGO)</span>
                        <span style="font-size:11px; color:#475569; font-weight:600; font-family: 'Outfit', sans-serif;">
                            Placar provável: <strong style="color: #2563eb;">{placar_prov[0]}x{placar_prov[1]}</strong> ({placar_prov[2]*100:.1f}%)
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 13px; font-family: 'Outfit', sans-serif;">
                        <div style="display: flex; align-items: center; gap: 6px; font-weight: 600; color: #2563eb;">
                            {flag_m_html} {m_sigla} {p_m:.0f}%
                        </div>
                        <div style="font-weight: 600; color: #64748b;">
                            Empate {p_e:.0f}%
                        </div>
                        <div style="display: flex; align-items: center; gap: 6px; font-weight: 600; color: #7c3aed; flex-direction: row-reverse;">
                            {flag_v_html} {v_sigla} {p_v:.0f}%
                        </div>
                    </div>
                    <div style="height: 14px; border-radius: 7px; display: flex; overflow: hidden; background-color: #e2e8f0; border: 1px solid #cbd5e1; position: relative;">
                        <div style="width: {p_m}%; background: linear-gradient(90deg, #2563eb, #3b82f6); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 10px; transition: width 0.5s ease; min-width: { '25px' if p_m >= 5 else '0px' };">
                            {f"{p_m:.0f}%" if p_m >= 15 else ""}
                        </div>
                        <div style="width: {p_e}%; background: #94a3b8; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 10px; transition: width 0.5s ease; min-width: { '25px' if p_e >= 5 else '0px' };">
                            {f"{p_e:.0f}%" if p_e >= 15 else ""}
                        </div>
                        <div style="width: {p_v}%; background: linear-gradient(90deg, #8b5cf6, #7c3aed); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 10px; transition: width 0.5s ease; min-width: { '25px' if p_v >= 5 else '0px' };">
                            {f"{p_v:.0f}%" if p_v >= 15 else ""}
                        </div>
                    </div>
                </div>
                """), unsafe_allow_html=True)
            with col_ia2:
                st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
                if st.button(f"🚀 Simular", key=f"sim_{g_id}"):
                    # Inicializar os dados da simulação e salvar no session_state
                    sim_data = generate_simulation_data(m_name, v_name, pred_pre)
                    st.session_state[f"sim_data_{g_id}"] = sim_data
                    st.session_state[f"sim_state_{g_id}"] = "running"
                    st.session_state.pop(f"scrolled_sim_{g_id}", None) # Resetar flag de rolagem se reiniciar
                    st.rerun()

            if st.session_state.get(f"sim_state_{g_id}", "idle") != "idle":
                run_simulation(g_id, m_name, v_name, m_sigla, v_sigla, pred_pre)

        # ---- JOGO FINALIZADO: Resumo ----
        elif status == "FINISHED":
            pred_pre = predict_match_probabilities(m_name, v_name, m_elo, v_elo, df_matches)
            p_m = pred_pre["prob_vitoria_mandante"] * 100
            p_e = pred_pre["prob_empate"] * 100
            p_v = pred_pre["prob_vitoria_visitante"] * 100

            flag_m_html = get_flag_html(m_name, width=20)
            flag_v_html = get_flag_html(v_name, width=20)

            st.markdown(clean_html(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); margin-top: 8px; opacity: 0.85;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size:12px; color:#059669; font-weight:700; font-family: 'Outfit', sans-serif;">📊 ANÁLISE PRÉ-JOGO</span>
                    <span style="background:#e0f2fe; color:#0369a1; font-size:10px; font-weight:700; padding:2px 6px; border-radius:8px; font-family: 'Outfit', sans-serif;">Finalizado</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 13px; font-family: 'Outfit', sans-serif;">
                    <div style="display: flex; align-items: center; gap: 6px; font-weight: 600; color: #2563eb;">
                        {flag_m_html} {m_sigla} {p_m:.0f}%
                    </div>
                    <div style="font-weight: 600; color: #64748b;">
                        Empate {p_e:.0f}%
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px; font-weight: 600; color: #7c3aed; flex-direction: row-reverse;">
                        {flag_v_html} {v_sigla} {p_v:.0f}%
                    </div>
                </div>
                <div style="height: 10px; border-radius: 5px; display: flex; overflow: hidden; background-color: #e2e8f0; border: 1px solid #cbd5e1; position: relative;">
                    <div style="width: {p_m}%; background: linear-gradient(90deg, #2563eb, #3b82f6); transition: width 0.5s ease; min-width: { '10px' if p_m >= 5 else '0px' };"></div>
                    <div style="width: {p_e}%; background: #94a3b8; transition: width 0.5s ease; min-width: { '10px' if p_e >= 5 else '0px' };"></div>
                    <div style="width: {p_v}%; background: linear-gradient(90deg, #8b5cf6, #7c3aed); transition: width 0.5s ease; min-width: { '10px' if p_v >= 5 else '0px' };"></div>
                </div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)




def run_simulation(g_id, m_name, v_name, m_sigla, v_sigla, pred_pre):
    """Executa e exibe a simulação de uma partida, mantendo o resultado estático após a conclusão."""
    # 1. Recuperar dados previamente gerados
    sim_data = st.session_state.get(f"sim_data_{g_id}")
    if not sim_data:
        return
        
    chosen_sim = sim_data["chosen_sim"]
    top_5_placares = sim_data["top_5_placares"]
    sim_state = st.session_state.get(f"sim_state_{g_id}", "idle")
    
    # Criar um container âncora com ID único para rolar a página até ele
    st.markdown(f'<div id="sim-container-{g_id}"></div>', unsafe_allow_html=True)
    
    # Rolar a página suavemente se estiver no início da simulação
    if sim_state == "running" and not st.session_state.get(f"scrolled_sim_{g_id}", False):
        st.components.v1.html(
            f"""
            <script>
                setTimeout(function() {{
                    var element = window.parent.document.getElementById("sim-container-{g_id}");
                    if (element) {{
                        element.scrollIntoView({{behavior: "smooth", block: "center"}});
                    }}
                }}, 300);
            </script>
            """,
            height=0
        )
        st.session_state[f"scrolled_sim_{g_id}"] = True

    st.markdown("---")
    st.subheader(f"⚡ Simulação: {m_name} vs {v_name}")

    if sim_state == "running":
        # Layout de duas colunas para a animação
        col_sim_left, col_sim_right = st.columns([1, 1])

        with col_sim_left:
            scoreboard_placeholder = st.empty()
            events_placeholder = st.empty()

        with col_sim_right:
            top5_placeholder = st.empty()
            st.markdown(
                '<h5 style="margin: 20px 0 8px 0; color: #475569; font-weight: 700; font-family: \'Outfit\', sans-serif; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">'
                '📊 Probabilidades de Vitória (Tempo Real)'
                '</h5>',
                unsafe_allow_html=True
            )
            plot_placeholder = st.empty()

        # Renderizar o top 5 de forma estática durante a simulação
        top5_html = (
            '<div class="glass-card" style="border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px; background: #ffffff; margin-bottom: 16px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.03);">'
            '<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">'
            '<h4 style="margin: 0; color: #1e293b; font-weight: 700; display: flex; align-items: center; gap: 8px; font-family: \'Outfit\', sans-serif; font-size: 15px;">'
            '🔮 Top 5 Resultados Mais Prováveis'
            '</h4>'
            '<span style="background: #e0e7ff; color: #3b82f6; font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 12px; text-transform: uppercase; font-family: \'Outfit\', sans-serif;">Modelo Poisson</span>'
            '</div>'
            '<div style="display: flex; flex-direction: column; gap: 12px;">'
        )
        for idx, item in enumerate(top_5_placares):
            placar = item["placar"]
            prob = item["probabilidade"]
            prob_pct = prob * 100
            
            parts = placar.split(" x ")
            if len(parts) == 2:
                placar_html = f'{get_flag_html(m_name, width=18)} <span style="margin: 0 4px; font-weight: 700;">{parts[0]}</span> x <span style="margin: 0 4px; font-weight: 700;">{parts[1]}</span> {get_flag_html(v_name, width=18)}'
            else:
                placar_html = placar
            
            if idx == 0:
                top5_html += (
                    f'<div style="border: 2px solid #fbbf24; background: #fef3c7; padding: 12px; border-radius: 10px; display: flex; flex-direction: column; gap: 6px; box-shadow: 0 4px 6px rgba(251, 191, 36, 0.1);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<div style="display: flex; align-items: center; gap: 8px;">'
                    f'<span style="font-weight: 800; font-size: 15px; color: #b45309; font-family: \'Outfit\', sans-serif; display: flex; align-items: center; gap: 6px;">{placar_html}</span>'
                    f'<span style="background: #fbbf24; color: #78350f; font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 12px; text-transform: uppercase; font-family: \'Outfit\', sans-serif;">MAIS PROVÁVEL</span>'
                    f'</div>'
                    f'<strong style="color: #b45309; font-size: 15px; font-family: \'Outfit\', sans-serif;">{prob_pct:.1f}%</strong>'
                    f'</div>'
                    f'<div style="width: 100%; background: #fef3c7; border: 1px solid #fde047; height: 6px; border-radius: 3px; overflow: hidden;">'
                    f'<div style="width: {prob_pct:.1f}%; background: #fbbf24; height: 100%;"></div>'
                    f'</div>'
                    f'</div>'
                )
            else:
                bar_width = prob_pct if placar != "-" else 0.0
                top5_html += (
                    f'<div style="border: 1px solid #e2e8f0; background: #f8fafc; padding: 10px 12px; border-radius: 10px; display: flex; flex-direction: column; gap: 6px;">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<span style="font-weight: 600; font-size: 14px; color: #1e293b; font-family: \'Outfit\', sans-serif; display: flex; align-items: center; gap: 6px;">{placar_html}</span>'
                    f'<span style="color: #475569; font-size: 14px; font-weight: 600; font-family: \'Outfit\', sans-serif;">{prob_pct:.1f}%</span>'
                    f'</div>'
                    f'<div style="width: 100%; background: #e2e8f0; height: 6px; border-radius: 3px; overflow: hidden;">'
                    f'<div style="width: {bar_width:.1f}%; background: linear-gradient(90deg, #2563eb, #7c3aed); height: 100%;"></div>'
                    f'</div>'
                    f'</div>'
                )
        top5_html += '</div></div>'
        top5_placeholder.markdown(top5_html, unsafe_allow_html=True)

        # Executar a simulação cronometrada (loop de animação)
        for minuto in range(0, 91, 5):
            g_m_sim, g_v_sim = chosen_sim["score_history"][minuto]
            eventos_atuais = [desc for m, et, desc in chosen_sim["events"] if m <= minuto]
            pred_live = simulate_match_in_play(pred_pre, minuto, g_m_sim, g_v_sim)

            status_str = f"⚡ SIMULAÇÃO — {minuto}'" if minuto < 90 else "🏁 FIM DE JOGO"
            badge_class = "badge-live" if minuto < 90 else "badge-finished"

            scoreboard_html = (
                f'<div class="glass-card {"neon-border-live" if minuto < 90 else "neon-border-finished"}" style="box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05); padding: 20px; border-radius: 16px; margin-bottom: 16px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">'
                f'<span style="color: #64748b; font-size: 13px; font-weight: 600; font-family: \'Outfit\', sans-serif;">Simulador de Partida (IA)</span>'
                f'<span class="badge {badge_class}" style="font-family: \'Outfit\', sans-serif;">{status_str}</span>'
                f'</div>'
                f'<div class="scoreboard" style="margin: 0; padding: 12px 16px; border-radius: 12px; display: flex; flex-wrap: nowrap; justify-content: space-between; align-items: center; gap: 10px;">'
                f'<div class="team-name" style="font-size: 15px; min-width: 0; flex: 1; text-align: center; color: #1e293b; font-family: \'Outfit\', sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">'
                f'<div style="margin-bottom:6px; display: flex; justify-content: center; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.12));">{get_flag_html(m_name, width=38)}</div>'
                f'<strong>{m_name}</strong>'
                f'</div>'
                f'<div class="score" style="font-size: 32px; color: #2563eb; flex-shrink: 0; padding: 0 10px; font-family: \'Outfit\', monospace; font-weight: 800; margin: 0;">{g_m_sim} - {g_v_sim}</div>'
                f'<div class="team-name" style="font-size: 15px; min-width: 0; flex: 1; text-align: center; color: #1e293b; font-family: \'Outfit\', sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">'
                f'<div style="margin-bottom:6px; display: flex; justify-content: center; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.12));">{get_flag_html(v_name, width=38)}</div>'
                f'<strong>{v_name}</strong>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            scoreboard_placeholder.markdown(scoreboard_html, unsafe_allow_html=True)

            p_m = pred_live["prob_vitoria_mandante"] * 100
            p_e = pred_live["prob_empate"] * 100
            p_v = pred_live["prob_vitoria_visitante"] * 100

            flag_m_html = get_flag_html(m_name, width=22)
            flag_v_html = get_flag_html(v_name, width=22)

            sim_bar_html = clean_html(f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); margin-top: 10px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-size: 13px; font-family: 'Outfit', sans-serif;">
                    <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #2563eb;">
                        {flag_m_html} {m_sigla} {p_m:.0f}%
                    </div>
                    <div style="font-weight: 700; color: #64748b;">
                        Empate {p_e:.0f}%
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #7c3aed; flex-direction: row-reverse;">
                        {flag_v_html} {v_sigla} {p_v:.0f}%
                    </div>
                </div>
                
                <div style="height: 24px; border-radius: 12px; display: flex; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.06); background-color: #f1f5f9; border: 1px solid #e2e8f0; position: relative;">
                    <div style="width: {p_m}%; background: linear-gradient(90deg, #2563eb, #3b82f6); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 12px; text-shadow: 0 1px 2px rgba(0,0,0,0.2); transition: width 0.5s ease; min-width: { '40px' if p_m >= 5 else '0px' };">
                        {f"{p_m:.0f}%" if p_m >= 10 else ""}
                    </div>
                    <div style="width: {p_e}%; background: #94a3b8; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 12px; text-shadow: 0 1px 2px rgba(0,0,0,0.2); transition: width 0.5s ease; min-width: { '40px' if p_e >= 5 else '0px' };">
                        {f"{p_e:.0f}%" if p_e >= 10 else ""}
                    </div>
                    <div style="width: {p_v}%; background: linear-gradient(90deg, #8b5cf6, #7c3aed); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 12px; text-shadow: 0 1px 2px rgba(0,0,0,0.2); transition: width 0.5s ease; min-width: { '40px' if p_v >= 5 else '0px' };">
                        {f"{p_v:.0f}%" if p_v >= 10 else ""}
                    </div>
                </div>
            </div>
            """)
            plot_placeholder.markdown(sim_bar_html, unsafe_allow_html=True)

            if eventos_atuais:
                events_html = (
                    '<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; max-height:280px; overflow-y:auto; box-shadow: 0 2px 10px rgba(0,0,0,0.02);">'
                    '<h4 style="margin:0 0 16px 0; color:#1e293b; font-weight:700; font-family: \'Outfit\', sans-serif; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">📋 Linha do Tempo</h4>'
                    '<div class="timeline-container">'
                    + "".join(reversed(eventos_atuais))
                    + '</div>'
                    + '</div>'
                )
                events_placeholder.markdown(events_html, unsafe_allow_html=True)
            else:
                events_placeholder.markdown(
                    '<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; text-align:center; color:#94a3b8; font-size:14px; font-family: \'Outfit\', sans-serif;">'
                    'Aguardando lances da partida... ⏱️'
                    '</div>',
                    unsafe_allow_html=True
                )
            
            time.sleep(0.3)

        # Transicionar para o estado finalizado
        st.session_state[f"sim_state_{g_id}"] = "finished"
        st.success("🔮 Previsão concluída com sucesso! (Esta foi apenas uma simulação temporária e não interfere nos resultados reais do campeonato).")
        st.balloons()
        st.rerun()

    elif sim_state == "finished":
        # Pegar os dados finais da simulação
        g_m_sim, g_v_sim = chosen_sim["final_score"]
        eventos_atuais = [desc for m, et, desc in chosen_sim["events"]]
        pred_live = simulate_match_in_play(pred_pre, 90, g_m_sim, g_v_sim)

        # Layout de duas colunas
        col_sim_left, col_sim_right = st.columns([1, 1])

        with col_sim_left:
            scoreboard_html = (
                f'<div class="glass-card neon-border-finished" style="box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05); padding: 20px; border-radius: 16px; margin-bottom: 16px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">'
                f'<span style="color: #64748b; font-size: 13px; font-weight: 600; font-family: \'Outfit\', sans-serif;">Simulador de Partida (IA)</span>'
                f'<span class="badge badge-finished" style="font-family: \'Outfit\', sans-serif;">🏁 FIM DE JOGO</span>'
                f'</div>'
                f'<div class="scoreboard" style="margin: 0; padding: 12px 16px; border-radius: 12px; display: flex; flex-wrap: nowrap; justify-content: space-between; align-items: center; gap: 10px;">'
                f'<div class="team-name" style="font-size: 15px; min-width: 0; flex: 1; text-align: center; color: #1e293b; font-family: \'Outfit\', sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">'
                f'<div style="margin-bottom:6px; display: flex; justify-content: center; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.12));">{get_flag_html(m_name, width=38)}</div>'
                f'<strong>{m_name}</strong>'
                f'</div>'
                f'<div class="score" style="font-size: 32px; color: #2563eb; flex-shrink: 0; padding: 0 10px; font-family: \'Outfit\', monospace; font-weight: 800; margin: 0;">{g_m_sim} - {g_v_sim}</div>'
                f'<div class="team-name" style="font-size: 15px; min-width: 0; flex: 1; text-align: center; color: #1e293b; font-family: \'Outfit\', sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">'
                f'<div style="margin-bottom:6px; display: flex; justify-content: center; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.12));">{get_flag_html(v_name, width=38)}</div>'
                f'<strong>{v_name}</strong>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(scoreboard_html, unsafe_allow_html=True)

            if eventos_atuais:
                events_html = (
                    '<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; max-height:280px; overflow-y:auto; box-shadow: 0 2px 10px rgba(0,0,0,0.02);">'
                    '<h4 style="margin:0 0 16px 0; color:#1e293b; font-weight:700; font-family: \'Outfit\', sans-serif; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">📋 Linha do Tempo</h4>'
                    '<div class="timeline-container">'
                    + "".join(reversed(eventos_atuais))
                    + '</div>'
                    + '</div>'
                )
                st.markdown(events_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; text-align:center; color:#94a3b8; font-size:14px; font-family: \'Outfit\', sans-serif;">'
                    'Nenhum lance registrado na simulação. ⏱️'
                    '</div>',
                    unsafe_allow_html=True
                )

        with col_sim_right:
            top5_html = (
                '<div class="glass-card" style="border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px; background: #ffffff; margin-bottom: 16px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.03);">'
                '<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">'
                '<h4 style="margin: 0; color: #1e293b; font-weight: 700; display: flex; align-items: center; gap: 8px; font-family: \'Outfit\', sans-serif; font-size: 15px;">'
                '🔮 Top 5 Resultados Mais Prováveis'
                '</h4>'
                '<span style="background: #e0e7ff; color: #3b82f6; font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 12px; text-transform: uppercase; font-family: \'Outfit\', sans-serif;">Modelo Poisson</span>'
                '</div>'
                '<div style="display: flex; flex-direction: column; gap: 12px;">'
            )
            for idx, item in enumerate(top_5_placares):
                placar = item["placar"]
                prob = item["probabilidade"]
                prob_pct = prob * 100
                
                parts = placar.split(" x ")
                if len(parts) == 2:
                    placar_html = f'{get_flag_html(m_name, width=18)} <span style="margin: 0 4px; font-weight: 700;">{parts[0]}</span> x <span style="margin: 0 4px; font-weight: 700;">{parts[1]}</span> {get_flag_html(v_name, width=18)}'
                else:
                    placar_html = placar
                
                if idx == 0:
                    top5_html += (
                        f'<div style="border: 2px solid #fbbf24; background: #fef3c7; padding: 12px; border-radius: 10px; display: flex; flex-direction: column; gap: 6px; box-shadow: 0 4px 6px rgba(251, 191, 36, 0.1);">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                        f'<div style="display: flex; align-items: center; gap: 8px;">'
                        f'<span style="font-weight: 800; font-size: 15px; color: #b45309; font-family: \'Outfit\', sans-serif; display: flex; align-items: center; gap: 6px;">{placar_html}</span>'
                        f'<span style="background: #fbbf24; color: #78350f; font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 12px; text-transform: uppercase; font-family: \'Outfit\', sans-serif;">MAIS PROVÁVEL</span>'
                        f'</div>'
                        f'<strong style="color: #b45309; font-size: 15px; font-family: \'Outfit\', sans-serif;">{prob_pct:.1f}%</strong>'
                        f'</div>'
                        f'<div style="width: 100%; background: #fef3c7; border: 1px solid #fde047; height: 6px; border-radius: 3px; overflow: hidden;">'
                        f'<div style="width: {prob_pct:.1f}%; background: #fbbf24; height: 100%;"></div>'
                        f'</div>'
                        f'</div>'
                    )
                else:
                    bar_width = prob_pct if placar != "-" else 0.0
                    top5_html += (
                        f'<div style="border: 1px solid #e2e8f0; background: #f8fafc; padding: 10px 12px; border-radius: 10px; display: flex; flex-direction: column; gap: 6px;">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                        f'<span style="font-weight: 600; font-size: 14px; color: #1e293b; font-family: \'Outfit\', sans-serif; display: flex; align-items: center; gap: 6px;">{placar_html}</span>'
                        f'<span style="color: #475569; font-size: 14px; font-weight: 600; font-family: \'Outfit\', sans-serif;">{prob_pct:.1f}%</span>'
                        f'</div>'
                        f'<div style="width: 100%; background: #e2e8f0; height: 6px; border-radius: 3px; overflow: hidden;">'
                        f'<div style="width: {bar_width:.1f}%; background: linear-gradient(90deg, #2563eb, #7c3aed); height: 100%;"></div>'
                        f'</div>'
                        f'</div>'
                    )
            top5_html += '</div></div>'
            st.markdown(top5_html, unsafe_allow_html=True)

            p_m = pred_live["prob_vitoria_mandante"] * 100
            p_e = pred_live["prob_empate"] * 100
            p_v = pred_live["prob_vitoria_visitante"] * 100

            flag_m_html = get_flag_html(m_name, width=22)
            flag_v_html = get_flag_html(v_name, width=22)

            sim_bar_html = clean_html(f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); margin-top: 10px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-size: 13px; font-family: 'Outfit', sans-serif;">
                    <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #2563eb;">
                        {flag_m_html} {m_sigla} {p_m:.0f}%
                    </div>
                    <div style="font-weight: 700; color: #64748b;">
                        Empate {p_e:.0f}%
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #7c3aed; flex-direction: row-reverse;">
                        {flag_v_html} {v_sigla} {p_v:.0f}%
                    </div>
                </div>
                
                <div style="height: 24px; border-radius: 12px; display: flex; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.06); background-color: #f1f5f9; border: 1px solid #e2e8f0; position: relative;">
                    <div style="width: {p_m}%; background: linear-gradient(90deg, #2563eb, #3b82f6); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 12px; text-shadow: 0 1px 2px rgba(0,0,0,0.2); transition: width 0.5s ease; min-width: { '40px' if p_m >= 5 else '0px' };">
                        {f"{p_m:.0f}%" if p_m >= 10 else ""}
                    </div>
                    <div style="width: {p_e}%; background: #94a3b8; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 12px; text-shadow: 0 1px 2px rgba(0,0,0,0.2); transition: width 0.5s ease; min-width: { '40px' if p_e >= 5 else '0px' };">
                        {f"{p_e:.0f}%" if p_e >= 10 else ""}
                    </div>
                    <div style="width: {p_v}%; background: linear-gradient(90deg, #8b5cf6, #7c3aed); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 12px; text-shadow: 0 1px 2px rgba(0,0,0,0.2); transition: width 0.5s ease; min-width: { '40px' if p_v >= 5 else '0px' };">
                        {f"{p_v:.0f}%" if p_v >= 10 else ""}
                    </div>
                </div>
            </div>
            """)
            st.markdown(sim_bar_html, unsafe_allow_html=True)
            
        def close_sim_callback(g_id=g_id):
            st.session_state[f"sim_state_{g_id}"] = "idle"
            st.session_state.pop(f"sim_data_{g_id}", None)
            st.session_state.pop(f"scrolled_sim_{g_id}", None)

        st.button("Fechar Simulação ❌", key=f"close_sim_btn_{g_id}", on_click=close_sim_callback)


# ============ LAYOUT PRINCIPAL ============

# Título / Hero Banner
st.markdown(clean_html("""
<div class="hero-banner">
    <!-- Decorative glow balls -->
    <div style="position: absolute; top: -50px; right: -50px; width: 180px; height: 180px; background: rgba(255, 255, 255, 0.1); border-radius: 50%; filter: blur(30px);"></div>
    <div style="position: absolute; bottom: -30px; left: 10%; width: 120px; height: 120px; background: rgba(124, 58, 237, 0.2); border-radius: 50%; filter: blur(20px);"></div>
    
    <div style="display: flex; align-items: center; gap: 20px; position: relative; z-index: 1;">
        <div style="font-size: 56px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.3)); line-height: 1; user-select: none;">⚽</div>
        <div>
            <div class="hero-banner-title">
                FuteBot — Copa do Mundo 2026 Live
            </div>
            <div class="hero-banner-subtitle">
                🌍 Acompanhamento em tempo real e simulações com análises preditivas avançadas (Poisson & ELO Rating)
            </div>
        </div>
    </div>
</div>
"""), unsafe_allow_html=True)


# Sidebar
st.sidebar.markdown("### ⚙️ Configurações")

api_key = st.sidebar.text_input(
    "🔑 Football-Data.org API Token:",
    value=get_api_key(st),
    key="api_key",
    type="password",
    help="Cole sua chave gratuita do site football-data.org para ver jogos oficiais da Copa 2026 em tempo real."
)

filtro_status = st.sidebar.multiselect(
    "Filtrar por status:",
    list(VALID_MATCH_STATUSES),
    default=["LIVE", "SCHEDULED", "FINISHED"]
)

filtro_hoje = st.sidebar.checkbox("Exibir apenas jogos de hoje", value=False)

# Carregar dados da API
is_offline = True
live_games = []
status_conexao = "ℹ️ MODO OFFLINE — Exibindo jogos locais"

if api_key:
    live_games_api, status_api = fetch_live_matches_from_api(api_key)
    status_conexao = status_api
    if live_games_api:
        # Se obtivemos dados válidos da API, sincronizar com o banco local
        # Limpar partidas mockadas na primeira sincronização da sessão
        if "db_synced_once" not in st.session_state:
            st.session_state["db_synced_once"] = True
            
        live_games = live_games_api
        is_offline = False
        for game in live_games:
            sync_api_match_to_db(game)
    else:
        # Fallback para o banco de dados local caso a API retorne vazia ou com erro
        df_2026 = load_2026_matches()
        live_games = df_2026.to_dict(orient="records")
        is_offline = True
else:
    # Sem chave da API: carregamos do banco local no modo offline puro
    df_2026 = load_2026_matches()
    live_games = df_2026.to_dict(orient="records")

# Status de conexão na sidebar
if is_offline:
    st.sidebar.info(status_conexao)
    st.sidebar.markdown(
        '<small style="color: #64748b;">Insira uma chave da API na barra lateral '
        'para sincronização em tempo real.</small>',
        unsafe_allow_html=True
    )
elif "✅" in status_conexao:
    st.sidebar.success(status_conexao)
elif "⚠️" in status_conexao:
    st.sidebar.warning(status_conexao)
elif "⏳" in status_conexao or "⏱️" in status_conexao:
    st.sidebar.info(status_conexao)
else:
    st.sidebar.error(status_conexao)

# Carregar dados históricos atualizados
df_matches = load_historical_matches()
df_teams = load_all_teams()
team_elo_map = {row["nome"]: row["elo_rating"] for _, row in df_teams.iterrows()}
team_sigla_map = {row["nome"]: row["sigla"] for _, row in df_teams.iterrows()}

# Botão de atualização
if st.button("🔄 Atualizar Resultados"):
    for key in list(st.session_state.keys()):
        if key.startswith("run_sim_") or key.startswith("sim_state_") or key.startswith("sim_data_") or key.startswith("scrolled_sim_"):
            st.session_state.pop(key, None)
    st.rerun()

# Filtrar jogos
jogos_filtrados = live_games

if filtro_status:
    jogos_filtrados = [g for g in jogos_filtrados if g["status"] in filtro_status]

if filtro_hoje:
    hoje_str = datetime.now().strftime("%Y-%m-%d")
    jogos_filtrados = [g for g in jogos_filtrados if g["data_hora"].startswith(hoje_str)]

# Mensagem quando não há jogos
if not live_games:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 40px;">
        <div style="font-size: 48px; margin-bottom: 16px;">🏟️</div>
        <h3 style="margin-bottom: 8px;">Nenhum jogo carregado</h3>
        <p style="color: #64748b; font-size: 16px;">
            Para ver os jogos da Copa do Mundo 2026 em tempo real, insira seu token gratuito
            da API <strong>football-data.org</strong> na barra lateral.<br><br>
            <a href="https://www.football-data.org/client/register" target="_blank"
               style="color: #2563eb; text-decoration: underline;">
               Criar conta gratuita no Football-Data.org →
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not jogos_filtrados:
    st.info("Nenhum jogo encontrado para os filtros selecionados.")
    st.stop()

# Separar jogos por status
jogos_live = [g for g in jogos_filtrados if g["status"] == "LIVE"]
jogos_scheduled = [g for g in jogos_filtrados if g["status"] == "SCHEDULED"]
jogos_finished = [g for g in jogos_filtrados if g["status"] == "FINISHED"]
jogos_unavailable = [g for g in jogos_filtrados if g["status"] in ("POSTPONED", "CANCELLED", "SUSPENDED")]

# Renderizar seções
if jogos_live:
    st.markdown(
        '<div class="section-header"><span class="section-header-icon">🔴</span>'
        '<span class="section-header-text">Ao Vivo</span></div>',
        unsafe_allow_html=True
    )
    for game in jogos_live:
        render_match_card(game, df_matches, team_elo_map, team_sigla_map)

if jogos_scheduled:
    st.markdown(
        '<div class="section-header"><span class="section-header-icon">📅</span>'
        '<span class="section-header-text">Agendados</span></div>',
        unsafe_allow_html=True
    )
    for game in jogos_scheduled:
        render_match_card(game, df_matches, team_elo_map, team_sigla_map)

if jogos_finished:
    st.markdown(
        '<div class="section-header"><span class="section-header-icon">✅</span>'
        '<span class="section-header-text">Encerrados</span></div>',
        unsafe_allow_html=True
    )
    for game in jogos_finished:
        render_match_card(game, df_matches, team_elo_map, team_sigla_map)

if jogos_unavailable:
    st.markdown(
        '<div class="section-header"><span class="section-header-icon">!</span>'
        '<span class="section-header-text">Indisponiveis / Adiados</span></div>',
        unsafe_allow_html=True
    )
    for game in jogos_unavailable:
        render_match_card(game, df_matches, team_elo_map, team_sigla_map)
