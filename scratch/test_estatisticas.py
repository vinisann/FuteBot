import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Importações locais do projeto
from src.database import load_historical_matches, load_all_teams, load_2026_matches
from src.ML_models import calculate_team_strengths, run_tournament_simulation
from src.styles import inject_css
from src.utils import get_flag

# Configuração da página
st.set_page_config(page_title="Estatísticas - FuteBot", page_icon="📊", layout="wide")

# CSS compartilhado (tema branco)
inject_css()

st.title("📊 Estatísticas e Simulação")
st.write("Explore o desempenho das seleções, analise os coeficientes de ataque/defesa e acompanhe as probabilidades de avanço dinâmicas.")

# Carregar dados
df_matches = load_historical_matches()
df_teams = load_all_teams()

# Função de simulação cacheada
@st.cache_data
def get_cached_simulation(matches_len, elo_sum):
    df_m = load_historical_matches()
    df_t = load_all_teams()
    df_2026 = load_2026_matches()
    return run_tournament_simulation(df_m, df_t, df_2026, iterations=150)

if df_matches.empty:
    st.warning("Não há partidas finalizadas no banco de dados para calcular estatísticas.")
else:
    # Filtro de Edição
    edicao = st.selectbox(
        "Selecione a Edição para visualizar:",
        ["Copa do Mundo 2026", "Histórico Completo (2018-2026)"]
    )
    
    if edicao == "Copa do Mundo 2026":
        df_matches_filtered = df_matches[df_matches["ano_copa"] == 2026]
    else:
        df_matches_filtered = df_matches

    # 1. Calcular Forças pelo modelo de Poisson
    ataque, defesa, avg_m, avg_v = calculate_team_strengths(df_matches_filtered)
    
    # Criar tabela de estatísticas
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
        gols_sofridos = m_matches["gols_visitante"].sum() + m_matches["gols_mandante"].sum() # Wait, gols sofridos de mandante é gols_visitante!
        # Correção no gols sofridos:
        gols_sofridos = m_matches["gols_visitante"].sum() + v_matches["gols_mandante"].sum()
        
        vitorias = (
            len(m_matches[m_matches["gols_mandante"] > m_matches["gols_visitante"]]) + 
            len(v_matches[v_matches["gols_visitante"] > v_matches["gols_mandante"]])
        )
        empates = (
            len(m_matches[m_matches["gols_mandante"] == m_matches["gols_visitante"]]) + 
            len(v_matches[v_matches["gols_visitante"] == v_matches["gols_mandante"]])
        )
        derrotas = total_jogos - vitorias - empates
        
        stats_rows.append({
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
            "Defesa (Fator)": defesa.get(team_name, 1.0)
        })
        
    df_stats = pd.DataFrame(stats_rows)
    
    # Insights de IA
    if not df_stats.empty:
        # Best attack: highest Attack factor, tie-breaker: ELO rating
        best_att_row = df_stats.sort_values(by=["Ataque (Fator)", "ELO Rating"], ascending=[False, False]).iloc[0]
        # Worst attack: lowest Attack factor, tie-breaker: ELO rating (lowest ELO)
        worst_att_row = df_stats.sort_values(by=["Ataque (Fator)", "ELO Rating"], ascending=[True, True]).iloc[0]
        # Best defense: lowest Defense factor, tie-breaker: ELO rating (highest ELO)
        best_def_row = df_stats.sort_values(by=["Defesa (Fator)", "ELO Rating"], ascending=[True, False]).iloc[0]
        # Worst defense: highest Defense factor, tie-breaker: ELO rating (lowest ELO)
        worst_def_row = df_stats.sort_values(by=["Defesa (Fator)", "ELO Rating"], ascending=[False, True]).iloc[0]
        
        st.markdown("### 🔮 Insights da Inteligência Artificial")
        st.write(f"Coeficientes de força e desempenho calculados com base em {edicao.lower()}.")
        
        col_ins1, col_ins2, col_ins3, col_ins4 = st.columns(4)
        
        with col_ins1:
            st.markdown(f"""
            <div class="metric-card-premium" style="border-left: 4px solid #10b981; background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%);">
                <span style="font-size: 11px; color: #047857; font-weight: 700; text-transform: uppercase;">🔥 Melhor Ataque</span>
                <h3 style="margin: 4px 0 2px 0; font-size: 20px; color: #1e293b;">{get_flag(best_att_row["Seleção"])} {best_att_row["Seleção"]}</h3>
                <p style="margin: 0; font-size: 13px; color: #047857; font-weight: 600;">Coeficiente: {best_att_row["Ataque (Fator)"]:.2f}x</p>
                <small style="color: #64748b; font-size: 11px;">Gols marcados: {best_att_row["Gols Pró"]}</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col_ins2:
            st.markdown(f"""
            <div class="metric-card-premium" style="border-left: 4px solid #ef4444; background: linear-gradient(180deg, #ffffff 0%, #fef2f2 100%);">
                <span style="font-size: 11px; color: #b91c1c; font-weight: 700; text-transform: uppercase;">❄️ Pior Ataque</span>
                <h3 style="margin: 4px 0 2px 0; font-size: 20px; color: #1e293b;">{get_flag(worst_att_row["Seleção"])} {worst_att_row["Seleção"]}</h3>
                <p style="margin: 0; font-size: 13px; color: #b91c1c; font-weight: 600;">Coeficiente: {worst_att_row["Ataque (Fator)"]:.2f}x</p>
                <small style="color: #64748b; font-size: 11px;">Gols marcados: {worst_att_row["Gols Pró"]}</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col_ins3:
            st.markdown(f"""
            <div class="metric-card-premium" style="border-left: 4px solid #2563eb; background: linear-gradient(180deg, #ffffff 0%, #eff6ff 100%);">
                <span style="font-size: 11px; color: #1d4ed8; font-weight: 700; text-transform: uppercase;">🛡️ Melhor Defesa</span>
                <h3 style="margin: 4px 0 2px 0; font-size: 20px; color: #1e293b;">{get_flag(best_def_row["Seleção"])} {best_def_row["Seleção"]}</h3>
                <p style="margin: 0; font-size: 13px; color: #1d4ed8; font-weight: 600;">Coeficiente: {best_def_row["Defesa (Fator)"]:.2f}x</p>
                <small style="color: #64748b; font-size: 11px;">Gols sofridos: {best_def_row["Gols Contra"]}</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col_ins4:
            st.markdown(f"""
            <div class="metric-card-premium" style="border-left: 4px solid #d97706; background: linear-gradient(180deg, #ffffff 0%, #fffbeb 100%);">
                <span style="font-size: 11px; color: #b45309; font-weight: 700; text-transform: uppercase;">⚠️ Pior Defesa</span>
                <h3 style="margin: 4px 0 2px 0; font-size: 20px; color: #1e293b;">{get_flag(worst_def_row["Seleção"])} {worst_def_row["Seleção"]}</h3>
                <p style="margin: 0; font-size: 13px; color: #b45309; font-weight: 600;">Coeficiente: {worst_def_row["Defesa (Fator)"]:.2f}x</p>
                <small style="color: #64748b; font-size: 11px;">Gols sofridos: {worst_def_row["Gols Contra"]}</small>
            </div>
            """, unsafe_allow_html=True)

    # Seletor de time para análise individual
    selected_team = st.sidebar.selectbox("Escolha uma Seleção para análise detalhada:", sorted(df_stats["Seleção"].unique()))
    
    # Abas
    tab1, tab2, tab3 = st.tabs([
        "🌎 Classificação Geral", 
        "📈 Chances de Avanço (Monte Carlo)", 
        "🔍 Análise por Seleção"
    ])
    
    with tab1:
        st.subheader("Desempenho Geral das Seleções")
        st.write("Esta tabela resume o poder ofensivo, defensivo e saldo de gols de cada equipe. Valores de Ataque acima de 1.0 indicam um ataque melhor que a média; valores de Defesa abaixo de 1.0 indicam uma defesa mais sólida que a média.")
        
        # Formatar tabela para exibição elegante
        df_display = df_stats.copy()
        df_display["Seleção"] = df_display["Seleção"].apply(lambda name: f"{get_flag(name)} {name}")
        df_display["Ataque (Fator)"] = df_display["Ataque (Fator)"].map('{:.2f}'.format)
        df_display["Defesa (Fator)"] = df_display["Defesa (Fator)"].map('{:.2f}'.format)
        
        st.dataframe(
            df_display.sort_values("ELO Rating", ascending=False),
            column_config={
                "ELO Rating": st.column_config.NumberColumn(format="%.0f"),
                "Ranking FIFA": st.column_config.NumberColumn(format="%d"),
            },
            hide_index=True,
            width="stretch"
        )
        
        # Gráficos de comparação
        col1, col2 = st.columns(2)
        with col1:
            fig_elo = px.bar(
                df_stats.sort_values("ELO Rating", ascending=False).head(15),
                x="Seleção", y="ELO Rating",
                title="Top 15 Seleções por ELO Rating",
                color="ELO Rating",
                color_continuous_scale="Viridis",
                template="plotly_white"
            )
            fig_elo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_elo, width="stretch")
            
        with col2:
            fig_xg = px.scatter(
                df_stats,
                x="Ataque (Fator)", y="Defesa (Fator)",
                hover_name="Seleção",
                text="Sigla",
                title="Mapeamento de Estilo de Jogo (Ataque vs Defesa)",
                template="plotly_white"
            )
            fig_xg.update_traces(textposition='top center')
            fig_xg.update_layout(
                xaxis_title="Força de Ataque (Maior = Melhor)",
                yaxis_title="Força de Defesa (Menor = Melhor)",
                yaxis=dict(autorange="reversed"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_xg, width="stretch")
            
    with tab2:
        st.subheader("🔮 Probabilidades de Avanço da Copa do Mundo 2026")
        st.write("A Inteligência Artificial simulou o restante da Copa do Mundo 2026 (fase de grupos e mata-mata) por 150 vezes, considerando o ELO Rating atual e as forças de ataque e defesa de cada time. Os resultados abaixo mostram as probabilidades estimadas de alcançar cada fase do torneio.")
        
        # Obter simulação cacheada
        m2026 = df_matches[df_matches["ano_copa"] == 2026]
        matches_len = len(m2026[m2026["status"] == "FINISHED"])
        elo_sum = sum(df_teams["elo_rating"])
        
        df_probs = get_cached_simulation(matches_len, elo_sum)
        
        # Formatar exibição
        df_probs_display = df_probs.copy()
        df_probs_display["Seleção"] = df_probs_display["Seleção"].apply(lambda name: f"{get_flag(name)} {name}")
        df_probs_display["Quartas (%)"] = df_probs_display["Quartas (%)"].map('{:.1f}%'.format)
        df_probs_display["Semis (%)"] = df_probs_display["Semis (%)"].map('{:.1f}%'.format)
        df_probs_display["Finais (%)"] = df_probs_display["Finais (%)"].map('{:.1f}%'.format)
        df_probs_display["Campeão (%)"] = df_probs_display["Campeão (%)"].map('{:.1f}%'.format)
        
        st.dataframe(
            df_probs_display.sort_values("Campeão (%)", ascending=False),
            hide_index=True,
            width="stretch"
        )
        
    with tab3:
        st.subheader(f"Análise de Desempenho: {get_flag(selected_team)} {selected_team}")
        
        team_stats = df_stats[df_stats["Seleção"] == selected_team].iloc[0]
        
        # Layout de métricas principais
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card-premium">
                <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">Ranking FIFA</span>
                <h3 style="margin: 5px 0 0 0; font-size: 28px; color: #1e293b;">#{int(team_stats["Ranking FIFA"])}</h3>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card-premium">
                <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">ELO Rating</span>
                <h3 style="margin: 5px 0 0 0; font-size: 28px; color: #2563eb;">{team_stats['ELO Rating']:.0f}</h3>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card-premium">
                <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">Força de Ataque</span>
                <h3 style="margin: 5px 0 0 0; font-size: 28px; color: #10b981;">{team_stats['Ataque (Fator)']:.2f}x</h3>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="metric-card-premium">
                <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">Força de Defesa</span>
                <h3 style="margin: 5px 0 0 0; font-size: 28px; color: #ef4444;">{team_stats['Defesa (Fator)']:.2f}x</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Detalhes de jogos da equipe
        st.markdown("### Histórico Recente de Partidas")
        team_matches = df_matches_filtered[
            (df_matches_filtered["mandante_nome"] == selected_team) | 
            (df_matches_filtered["visitante_nome"] == selected_team)
        ].sort_values("data_hora", ascending=False)
        
        formatted_matches = []
        for _, match in team_matches.iterrows():
            is_mandante = match["mandante_nome"] == selected_team
            oponente = match["visitante_nome"] if is_mandante else match["mandante_nome"]
            gols_pro = match["gols_mandante"] if is_mandante else match["gols_visitante"]
            gols_contra = match["gols_visitante"] if is_mandante else match["gols_mandante"]
            
            if gols_pro > gols_contra:
                resultado = "✅ Vitória"
            elif gols_pro == gols_contra:
                resultado = "➖ Empate"
            else:
                resultado = "❌ Derrota"
                
            formatted_matches.append({
                "Data": match["data_hora"],
                "Competição / Fase": match["fase"],
                "Oponente": f"{get_flag(oponente)} {oponente}",
                "Gols Marcados": gols_pro,
                "Gols Sofridos": gols_contra,
                "Resultado": resultado
            })
            
        st.table(pd.DataFrame(formatted_matches))
