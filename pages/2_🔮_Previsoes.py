import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import html
import textwrap

# Importações locais do projeto
from src.database import init_db, load_historical_matches, load_all_teams, load_prediction_evaluations
from src.ML_models import predict_match_probabilities
from src.model_calibration import build_model_calibration
from src.model_explainability import explain_prediction, format_explanation_markdown
from src.external_signals import build_match_external_signals
from src.player_impact import build_player_impact_from_lineups
from src.scraper import fetch_match_specific_news, get_probable_lineup
from src.styles import inject_css
from src.utils import get_flag, get_flag_html


def _lineup_to_text(lineup):
    if not lineup or lineup.get("tecnico") == "A confirmar":
        return None
    titulares = lineup.get("titulares", [])
    if not titulares:
        return None
    return f"Provavel escalacao: {'; '.join(str(player) for player in titulares)}."


@st.cache_data(ttl=1800, show_spinner=False)
def load_external_prediction_inputs(mandante_nome, visitante_nome):
    try:
        news = fetch_match_specific_news(mandante_nome, visitante_nome, max_results=3)
    except Exception:
        news = []

    news_by_team = {mandante_nome: [], visitante_nome: []}
    for item in news:
        title = str(item.get("title", ""))
        parsed_lineup = str(item.get("parsed_lineup") or "")
        text = " ".join(part for part in [title, parsed_lineup] if part).strip()
        text_lower = text.lower()
        if not text:
            continue
        if mandante_nome.lower() in text_lower:
            news_by_team[mandante_nome].append(text)
        if visitante_nome.lower() in text_lower:
            news_by_team[visitante_nome].append(text)

    raw_lineups = {
        mandante_nome: get_probable_lineup(mandante_nome),
        visitante_nome: get_probable_lineup(visitante_nome),
    }
    lineups = {
        mandante_nome: _lineup_to_text(raw_lineups[mandante_nome]),
        visitante_nome: _lineup_to_text(raw_lineups[visitante_nome]),
    }
    player_impact = build_player_impact_from_lineups(
        mandante_nome,
        visitante_nome,
        lineups=raw_lineups,
    )
    return {"news": news_by_team, "lineups": lineups, "player_impact": player_impact}


def clean_html(html_str):
    """Remove recuos e espaços vazios por linha para evitar falso-positivo de bloco de código no Streamlit Markdown."""
    return "\n".join(line.strip() for line in html_str.split("\n"))

# Configuração da página
st.set_page_config(page_title="Previsões IA - FuteBot", page_icon="🔮", layout="wide")

init_db()

# CSS compartilhado (tema branco)
inject_css()

st.title("🔮 Previsões Estatísticas e de IA")
st.write("Selecione duas equipes nacionais quaisquer para simular um confronto hipotético. A nossa IA calculará as probabilidades de vitória e placares prováveis com base no modelo Poisson-ELO.")

# Carregar dados
df_matches = load_historical_matches()
df_teams = load_all_teams()
model_calibration = build_model_calibration(df_matches, load_prediction_evaluations())

if df_matches.empty:
    st.error("Dados históricos insuficientes no banco de dados para alimentar a IA preditiva.")
else:
    # Seletores de equipes
    teams_list = sorted(df_teams["nome"].unique())
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        mandante_nome = st.selectbox("Time Mandante (ou Equipe A):", teams_list, index=teams_list.index("Brasil") if "Brasil" in teams_list else 0, format_func=lambda x: f"{get_flag(x)} {x}")
    with col_t2:
        visitante_nome = st.selectbox("Time Visitante (ou Equipe B):", teams_list, index=teams_list.index("Alemanha") if "Alemanha" in teams_list else 1, format_func=lambda x: f"{get_flag(x)} {x}")
        
    if mandante_nome == visitante_nome:
        st.warning("Selecione duas seleções diferentes para simular um jogo!")
    else:
        # Obter dados de ELO e Ranking FIFA
        mandante_data = df_teams[df_teams["nome"] == mandante_nome].iloc[0]
        visitante_data = df_teams[df_teams["nome"] == visitante_nome].iloc[0]
        
        m_elo = mandante_data["elo_rating"]
        v_elo = visitante_data["elo_rating"]
        m_fifa = mandante_data["ranking_fifa"]
        v_fifa = visitante_data["ranking_fifa"]
        
        # Calcular previsões de probabilidade
        external_inputs = load_external_prediction_inputs(mandante_nome, visitante_nome)
        external_signals = build_match_external_signals(
            mandante_nome,
            visitante_nome,
            news_items=external_inputs["news"],
            lineups=external_inputs["lineups"],
        )
        external_signals["player_impact"] = external_inputs["player_impact"]
        pred = predict_match_probabilities(
            mandante_nome,
            visitante_nome,
            m_elo,
            v_elo,
            df_matches,
            calibration=model_calibration,
            external_signals=external_signals,
        )
        explanation = explain_prediction(pred)
        
        p_mandante = pred["prob_vitoria_mandante"] * 100
        p_empate = pred["prob_empate"] * 100
        p_visitante = pred["prob_vitoria_visitante"] * 100
        placar_prov = pred["placar_mais_provavel"]
        
        # Exibição do confronto e dados básicos
        st.markdown(clean_html(f"""
        <div class="glass-card" style="background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);">
            <div style="display: flex; justify-content: space-around; text-align: center; align-items: center; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1; min-width: 200px; padding: 10px;">
                    <div style="margin-bottom:12px; display:flex; justify-content:center; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));">{get_flag_html(mandante_nome, width=80)}</div>
                    <h2 style="font-size: 28px; margin: 0 0 12px 0; color: #1e293b;">{mandante_nome}</h2>
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                        <p style="margin: 0; color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 600;">ELO Rating</p>
                        <p style="margin: 2px 0 0 0; color: #2563eb; font-size: 22px; font-weight: 800;">{m_elo:.0f}</p>
                    </div>
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-top: 8px;">
                        <p style="margin: 0; color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 600;">Ranking FIFA</p>
                        <p style="margin: 2px 0 0 0; color: #475569; font-size: 20px; font-weight: 700;">#{m_fifa}</p>
                    </div>
                </div>
                <div style="flex: 0 0 80px; text-align: center;">
                    <div class="vs-badge">VS</div>
                </div>
                <div style="flex: 1; min-width: 200px; padding: 10px;">
                    <div style="margin-bottom:12px; display:flex; justify-content:center; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));">{get_flag_html(visitante_nome, width=80)}</div>
                    <h2 style="font-size: 28px; margin: 0 0 12px 0; color: #1e293b;">{visitante_nome}</h2>
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                        <p style="margin: 0; color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 600;">ELO Rating</p>
                        <p style="margin: 2px 0 0 0; color: #7c3aed; font-size: 22px; font-weight: 800;">{v_elo:.0f}</p>
                    </div>
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-top: 8px;">
                        <p style="margin: 0; color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 600;">Ranking FIFA</p>
                        <p style="margin: 2px 0 0 0; color: #475569; font-size: 20px; font-weight: 700;">#{v_fifa}</p>
                    </div>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        # Bloco de Probabilidades
        st.markdown("### 🔮 Probabilidade de Resultado Final")
        
        # Obter bandeiras HTML
        flag_m_html = get_flag_html(mandante_nome, width=24)
        flag_v_html = get_flag_html(visitante_nome, width=24)
        
        st.markdown(clean_html(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.02); margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    {flag_m_html}
                    <span style="font-weight: 700; color: #2563eb; font-size: 15px;">Vitória do {mandante_nome}</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: center;">
                    <span style="font-weight: 600; color: #64748b; font-size: 14px;">Empate</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px; flex-direction: row-reverse;">
                    {flag_v_html}
                    <span style="font-weight: 700; color: #7c3aed; font-size: 15px;">Vitória do {visitante_nome}</span>
                </div>
            </div>
            
            <div style="height: 28px; border-radius: 14px; display: flex; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.06); background-color: #f1f5f9; border: 1px solid #e2e8f0; position: relative;">
                <div style="width: {p_mandante}%; background: linear-gradient(90deg, #2563eb, #3b82f6); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 13px; text-shadow: 0 1px 2px rgba(0,0,0,0.2); transition: width 0.5s ease; min-width: { '45px' if p_mandante >= 5 else '0px' };">
                    {f"{p_mandante:.0f}%" if p_mandante >= 5 else ""}
                </div>
                <div style="width: {p_empate}%; background: #94a3b8; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 13px; text-shadow: 0 1px 2px rgba(0,0,0,0.2); transition: width 0.5s ease; min-width: { '45px' if p_empate >= 5 else '0px' };">
                    {f"{p_empate:.0f}%" if p_empate >= 5 else ""}
                </div>
                <div style="width: {p_visitante}%; background: linear-gradient(90deg, #8b5cf6, #7c3aed); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 13px; text-shadow: 0 1px 2px rgba(0,0,0,0.2); transition: width 0.5s ease; min-width: { '45px' if p_visitante >= 5 else '0px' };">
                    {f"{p_visitante:.0f}%" if p_visitante >= 5 else ""}
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        # Detalhes de gols esperados e placar mais provável
        xg_m = pred["xG_mandante"]
        xg_v = pred["xG_visitante"]
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(clean_html(f"""
            <div class="glass-card" style="height: 100%;">
                <h3 style="margin-top: 0; color: #1e293b; font-family: 'Outfit', sans-serif;">Expectativa de Gols (xG)</h3>
                <div style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">
                    <div style="display: flex; align-items: center; justify-content: space-between; font-size: 14px;">
                        <span style="display: flex; align-items: center; gap: 8px;">
                            {get_flag_html(mandante_nome, width=20)} Gols Esperados do <strong>{mandante_nome}</strong>
                        </span>
                        <span style="color:#2563eb; font-size:20px; font-weight:700;">{xg_m:.2f}</span>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; font-size: 14px;">
                        <span style="display: flex; align-items: center; gap: 8px;">
                            {get_flag_html(visitante_nome, width=20)} Gols Esperados do <strong>{visitante_nome}</strong>
                        </span>
                        <span style="color:#7c3aed; font-size:20px; font-weight:700;">{xg_v:.2f}</span>
                    </div>
                </div>
                <small style="color:#64748b;">xG é calculado através da análise da força ofensiva do mandante contra a eficiência defensiva do visitante, ajustado pelo diferencial de ELO.</small>
            </div>
            """), unsafe_allow_html=True)
            
        with col_res2:
            st.markdown(clean_html(f"""
            <div class="glass-card glow-gold" style="background: linear-gradient(180deg, #ffffff 0%, #fffbeb 100%); height: 100%;">
                <h3 style="text-align: center; color: #f59e0b !important; margin-top: 0; font-family: 'Outfit', sans-serif;">🎯 Placar Mais Provável</h3>
                <div style="font-size: 44px; font-weight: 800; color: #d97706; text-align:center; padding: 8px 0; font-family: 'Outfit', sans-serif; display: flex; align-items: center; justify-content: center; gap: 12px;">
                    {get_flag_html(mandante_nome, width=32)}
                    <span>{placar_prov[0]} x {placar_prov[1]}</span>
                    {get_flag_html(visitante_nome, width=32)}
                </div>
                <p style="text-align:center; margin: 0; color: #475569; font-family: 'Outfit', sans-serif; font-size: 14px;">Probabilidade de ocorrência exata: <strong>{placar_prov[2]*100:.1f}%</strong></p>
            </div>
            """), unsafe_allow_html=True)

        st.markdown(format_explanation_markdown(explanation))

            
        # Matriz de Probabilidade de Placares (Heatmap)
        st.markdown("### 📊 Análise Detalhada de Probabilidades")
        
        # 1. Preparar dados para o Heatmap
        matriz_np = np.array(pred["matriz_placar"]) * 100
        gols_label = [str(x) for x in pred["gols_range"]]
        max_gols = max(pred["gols_range"])
        max_m_idx = placar_prov[0]
        max_v_idx = placar_prov[1]
        
        # Criar anotações de texto manuais para acessibilidade, contraste e destaque do mais provável
        annotations = []
        for m_idx in range(max_gols + 1):
            for v_idx in range(max_gols + 1):
                val = matriz_np[m_idx][v_idx]
                if val >= 0.5:
                    # Contraste inteligente: fundo escuro (val >= 4.0%) recebe texto branco, fundo claro recebe texto escuro
                    text_color = "#ffffff" if val >= 4.0 else "#1e293b"
                    
                    is_max = (m_idx == max_m_idx and v_idx == max_v_idx)
                    text_str = f"<b>{val:.1f}%</b>" if is_max else f"{val:.1f}%"
                    
                    # Forçar texto claro/escuro apropriado no placar mais provável
                    if is_max:
                        text_color = "#ffffff" if val >= 3.0 else "#1e293b"
                    
                    annotations.append(dict(
                        x=v_idx,
                        y=m_idx,
                        text=text_str,
                        showarrow=False,
                        font=dict(family="Outfit, sans-serif", size=10, color=text_color)
                    ))
                    
        # Adicionar legendas de áreas no mapa de calor (para ajudar a entender as vitórias de cada lado)
        annotations.append(dict(
            x=0.8,
            y=max_gols - 0.8,
            text=f"<b>◀ VITÓRIA DO<br>{mandante_nome.upper()}</b>",
            showarrow=False,
            font=dict(family="Outfit, sans-serif", size=11, color="rgba(37, 99, 235, 0.3)"),
            align="center"
        ))
        annotations.append(dict(
            x=max_gols - 0.8,
            y=0.8,
            text=f"<b>VITÓRIA DO<br>{visitante_nome.upper()} ▶</b>",
            showarrow=False,
            font=dict(family="Outfit, sans-serif", size=11, color="rgba(124, 58, 237, 0.3)"),
            align="center"
        ))
        
        # Mapa de cores personalizado (de azul claro para azul escuro profundo)
        custom_blues = [
            [0.0, '#f8fafc'],   # Slate/cinza bem claro
            [0.05, '#eff6ff'],  # Azul muito claro (BG)
            [0.2, '#dbeafe'],   # Azul claro
            [0.5, '#60a5fa'],   # Azul médio
            [0.8, '#2563eb'],   # Royal Blue
            [1.0, '#1d4ed8']    # Dark Blue
        ]
        
        fig_heat = go.Figure()
        
        # Trace do Heatmap
        fig_heat.add_trace(go.Heatmap(
            z=matriz_np,
            x=gols_label,
            y=gols_label,
            colorscale=custom_blues,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate=f'Placar: {mandante_nome} %{{y}} x %{{x}} {visitante_nome}<br>Chance: %{{z:.2f}}%<extra></extra>'
        ))
        
        # Trace da linha diagonal tracejada de Empate
        fig_heat.add_trace(go.Scatter(
            x=[-0.5, max_gols + 0.5],
            y=[-0.5, max_gols + 0.5],
            mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.6)", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Highlight na célula do placar mais provável com borda dourada
        fig_heat.add_shape(
            type="rect",
            x0=max_v_idx - 0.5, y0=max_m_idx - 0.5,
            x1=max_v_idx + 0.5, y1=max_m_idx + 0.5,
            line=dict(color="#f59e0b", width=3.5, dash="solid"),
            fillcolor="rgba(0,0,0,0)",
            layer="above"
        )
        
        fig_heat.update_layout(
            xaxis=dict(
                title=dict(
                    text=f"Gols do Visitante ({visitante_nome}) ➡️",
                    font=dict(size=12, family="Outfit", color="#475569")
                ),
                tickfont=dict(size=12, family="Outfit"),
                showgrid=False,
                range=[-0.5, max_gols + 0.5],
                dtick=1
            ),
            yaxis=dict(
                title=dict(
                    text=f"⬅️ Gols do Mandante ({mandante_nome})",
                    font=dict(size=12, family="Outfit", color="#475569")
                ),
                tickfont=dict(size=12, family="Outfit"),
                showgrid=False,
                range=[max_gols + 0.5, -0.5], # Reverte o eixo y mantendo limites quadrados
                dtick=1
            ),
            height=480,
            margin=dict(l=60, r=20, t=20, b=60),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            template="plotly_white",
            annotations=annotations
        )
        
        # 2. Calcular Estatísticas e Mercados Auxiliares
        matriz_raw = pred["matriz_placar"]
        gols_range = pred["gols_range"]
        
        prob_btts = 0.0
        prob_over_2_5 = 0.0
        prob_over_1_5 = 0.0
        
        score_probs = {}
        for m_idx in range(len(gols_range)):
            for v_idx in range(len(gols_range)):
                prob = matriz_raw[m_idx][v_idx]
                score_probs[(m_idx, v_idx)] = prob
                
                # Ambas Marcam
                if m_idx > 0 and v_idx > 0:
                    prob_btts += prob
                # Mais de 2.5 Gols
                if m_idx + v_idx > 2:
                    prob_over_2_5 += prob
                # Mais de 1.5 Gols
                if m_idx + v_idx > 1:
                    prob_over_1_5 += prob
                    
        # Ordenar os top 5 resultados
        sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
        top_5_placares = []
        for score, prob in sorted_scores[:5]:
            top_5_placares.append({
                "placar": f"{score[0]} x {score[1]}",
                "probabilidade": prob * 100
            })
            
        # Layout de duas colunas
        col_heat, col_analysis = st.columns([5, 4])
        
        with col_heat:
            st.write("#### 🗺️ Mapa de Calor de Placares Exatos")
            st.write("A linha pontilhada diagonal indica o **empate**. A célula com borda dourada destaca o **placar mais provável**.")
            st.plotly_chart(fig_heat, width='stretch')
            
        with col_analysis:
            st.write("#### 🎯 Insights de Probabilidade do Confronto")
            
            # Bloco dos top 5 placares exatos com estilo premium polido
            top5_html = """
            <div class="glass-card" style="margin-bottom: 16px; padding: 18px; border: 1px solid #e2e8f0; background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
                <h4 style="margin: 0 0 12px 0; font-family: 'Outfit', sans-serif; font-size: 14px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">
                    🎯 Top 5 Placares Mais Prováveis
                </h4>
                <div style="display: flex; flex-direction: column; gap: 8px;">
            """
            for idx, item in enumerate(top_5_placares):
                placar = item["placar"]
                prob = item["probabilidade"]
                
                parts = placar.split(" x ")
                if len(parts) == 2:
                    placar_html = f'{get_flag_html(mandante_nome, width=18)} <span style="margin: 0 4px; font-weight: 700;">{parts[0]}</span> x <span style="margin: 0 4px; font-weight: 700;">{parts[1]}</span> {get_flag_html(visitante_nome, width=18)}'
                else:
                    placar_html = placar
                
                if idx == 0:
                    # Mais provável em ouro polido
                    top5_html += f"""
                    <div style="border: 1px solid #fde047; background: linear-gradient(135deg, #fffdf0 0%, #fffbeb 100%); padding: 8px 12px; border-radius: 8px; display: flex; flex-direction: column; gap: 4px; box-shadow: 0 2px 6px rgba(234, 179, 8, 0.08);">
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 13px;">
                            <span style="font-weight: 700; color: #78350f; display: flex; align-items: center; gap: 6px;">{placar_html}</span>
                            <strong style="color: #b45309;">{prob:.1f}%</strong>
                        </div>
                        <div style="width: 100%; background: #fef9c3; height: 5px; border-radius: 3px; overflow: hidden; border: 1px solid #fef08a;">
                            <div style="width: {prob:.1f}%; background: #eab308; height: 100%; border-radius: 3px;"></div>
                        </div>
                    </div>
                    """
                else:
                    top5_html += f"""
                    <div style="border: 1px solid #e2e8f0; background: #f8fafc; padding: 8px 12px; border-radius: 8px; display: flex; flex-direction: column; gap: 4px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 13px;">
                            <span style="font-weight: 600; color: #1e293b; display: flex; align-items: center; gap: 6px;">{placar_html}</span>
                            <strong style="color: #2563eb;">{prob:.1f}%</strong>
                        </div>
                        <div style="width: 100%; background: #e2e8f0; height: 5px; border-radius: 3px; overflow: hidden;">
                            <div style="width: {prob:.1f}%; background: linear-gradient(90deg, #3b82f6, #6366f1); height: 100%; border-radius: 3px;"></div>
                        </div>
                    </div>
                    """
            top5_html += "</div></div>"
            st.markdown(clean_html(top5_html), unsafe_allow_html=True)
            
            # Bloco de Probabilidade de Mercados com visualizações inline
            st.markdown(clean_html(f"""
            <div class="glass-card" style="padding: 18px; border: 1px solid #e2e8f0; background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
                <h4 style="margin: 0 0 16px 0; font-family: 'Outfit', sans-serif; font-size: 14px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px;">
                    📈 Mercados & Estatísticas de Gols
                </h4>
                <div style="display: flex; flex-direction: column; gap: 16px; font-size: 13px; font-family: 'Outfit', sans-serif;">
                    <!-- Ambas Marcam -->
                    <div style="border-bottom: 1px solid #f1f5f9; padding-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="color: #1e293b; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                                ⚽ Ambas as Equipes Marcam
                                <span style="display: inline-flex; gap: 2px; align-items: center;">
                                    ({get_flag_html(mandante_nome, width=14)} + {get_flag_html(visitante_nome, width=14)})
                                </span>
                            </span>
                            <div>
                                <span style="color: #10b981; font-weight: 700;">Sim {prob_btts * 100:.1f}%</span>
                                <span style="color: #94a3b8; margin: 0 4px;">|</span>
                                <span style="color: #ef4444; font-weight: 700;">{(1.0 - prob_btts) * 100:.1f}% Não</span>
                            </div>
                        </div>
                        <div style="width: 100%; background: #fee2e2; height: 6px; border-radius: 3px; overflow: hidden; display: flex;">
                            <div style="width: {prob_btts * 100}%; background: #10b981; height: 100%;"></div>
                        </div>
                        <small style="color: #64748b; font-size: 11px; display: block; margin-top: 4px;">Chance de que ambas as seleções façam gols na partida.</small>
                    </div>
                    
                    <!-- Over Under 1.5 -->
                    <div style="border-bottom: 1px solid #f1f5f9; padding-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="color: #1e293b; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                                ⚡ Mais de 1.5 Gols (Total)
                            </span>
                            <div>
                                <span style="color: #7c3aed; font-weight: 700;">Over {prob_over_1_5 * 100:.1f}%</span>
                                <span style="color: #94a3b8; margin: 0 4px;">|</span>
                                <span style="color: #64748b; font-weight: 700;">{(1.0 - prob_over_1_5) * 100:.1f}% Under</span>
                            </div>
                        </div>
                        <div style="width: 100%; background: #e2e8f0; height: 6px; border-radius: 3px; overflow: hidden; display: flex;">
                            <div style="width: {prob_over_1_5 * 100}%; background: #7c3aed; height: 100%;"></div>
                        </div>
                        <small style="color: #64748b; font-size: 11px; display: block; margin-top: 4px;">Chance de que a partida termine com **2 ou mais gols** no placar somado.</small>
                    </div>
                    
                    <!-- Over Under 2.5 -->
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="color: #1e293b; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                                📊 Mais de 2.5 Gols (Total)
                            </span>
                            <div>
                                <span style="color: #2563eb; font-weight: 700;">Over {prob_over_2_5 * 100:.1f}%</span>
                                <span style="color: #94a3b8; margin: 0 4px;">|</span>
                                <span style="color: #64748b; font-weight: 700;">{(1.0 - prob_over_2_5) * 100:.1f}% Under</span>
                            </div>
                        </div>
                        <div style="width: 100%; background: #e2e8f0; height: 6px; border-radius: 3px; overflow: hidden; display: flex;">
                            <div style="width: {prob_over_2_5 * 100}%; background: #2563eb; height: 100%;"></div>
                        </div>
                        <small style="color: #64748b; font-size: 11px; display: block; margin-top: 4px;">Chance de que a partida termine com **3 ou mais gols** no placar somado (linha padrão do mercado).</small>
                    </div>
                </div>
            </div>
            """), unsafe_allow_html=True)
