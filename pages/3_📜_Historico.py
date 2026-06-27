import streamlit as st
import pandas as pd

# Importações locais do projeto
from src.database import init_db, load_historical_matches
from src.styles import inject_css
from src.utils import get_flag_html, format_fase, format_fase_option

# Configuração da página
st.set_page_config(page_title="Histórico de Copas - FuteBot", page_icon="📜", layout="wide")

init_db()

# CSS compartilhado (tema branco)
inject_css()

st.title("📜 Histórico de Copas do Mundo")
st.write("Navegue por edições passadas da Copa do Mundo, filtre partidas por fase e consulte estatísticas consolidadas de gols e placares históricos.")

# Carregar dados
df_matches = load_historical_matches()

if df_matches.empty:
    st.info("Nenhum dado de histórico de partidas encontrado no banco SQLite.")
else:
    # Sidebar filtros
    edicoes = sorted(list(df_matches["ano_copa"].unique()), reverse=True)
    edicao_selecionada = st.sidebar.selectbox("Selecione a Edição da Copa:", edicoes)
    
    # Filtrar partidas da edição (usando .copy() para evitar SettingWithCopyWarning)
    df_edicao = df_matches[df_matches["ano_copa"] == edicao_selecionada].copy()
    
    fases = ["Todas"] + list(df_edicao["fase"].unique())
    fase_selecionada = st.sidebar.selectbox("Filtrar por Fase:", fases, format_func=format_fase_option)
    
    if fase_selecionada != "Todas":
        df_edicao = df_edicao[df_edicao["fase"] == fase_selecionada].copy()
        
    # Estatísticas rápidas da copa selecionada
    st.subheader(f"Estatísticas Gerais: Copa do Mundo de {edicao_selecionada}")
    
    total_jogos = len(df_edicao)
    total_gols = df_edicao["gols_mandante"].sum() + df_edicao["gols_visitante"].sum()
    media_gols = total_gols / total_jogos if total_jogos > 0 else 0
    
    # Encontrar maior placar
    df_edicao["total_gols_partida"] = df_edicao["gols_mandante"] + df_edicao["gols_visitante"]
    maior_jogo = df_edicao.sort_values("total_gols_partida", ascending=False)
    
    maior_placar_html = """
    <div class="glass-card" style="height: 100%; min-height: 118px; margin-bottom: 0; padding: 14px 16px; text-align: center; display: flex; flex-direction: column; justify-content: center; gap: 8px;">
        <div style="font-size: 13px; color: #64748b; font-weight: 700;">Partida com Mais Gols</div>
        <div style="font-size: 15px; color: #94a3b8;">N/A</div>
    </div>
    """
    if not maior_jogo.empty:
        mj = maior_jogo.iloc[0]
        maior_placar_html = f"""
        <div class="glass-card" style="height: 100%; min-height: 118px; margin-bottom: 0; padding: 14px 16px; text-align: center; display: flex; flex-direction: column; justify-content: center; gap: 8px;">
            <div style="font-size: 13px; color: #64748b; font-weight: 700;">Partida com Mais Gols</div>
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px; flex-wrap: wrap; line-height: 1.25;">
                <span style="display: inline-flex; align-items: center; gap: 6px; font-weight: 700; color: #1e293b; white-space: nowrap;">
                    {get_flag_html(mj['mandante_nome'], width=24)} {mj['mandante_nome']}
                </span>
                <span style="background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.18); color: #2563eb; border-radius: 999px; padding: 4px 12px; font-size: 18px; font-weight: 800; white-space: nowrap;">
                    {mj['gols_mandante']} x {mj['gols_visitante']}
                </span>
                <span style="display: inline-flex; align-items: center; gap: 6px; font-weight: 700; color: #1e293b; white-space: nowrap;">
                    {get_flag_html(mj['visitante_nome'], width=24)} {mj['visitante_nome']}
                </span>
            </div>
        </div>
        """
        
    # Layout de métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Partidas", total_jogos)
    col2.metric("Total de Gols Marcados", total_gols)
    col3.metric("Média de Gols / Jogo", f"{media_gols:.2f}")
    col4.markdown(maior_placar_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Exibir Partidas Filtradas
    st.subheader(f"Lista de Jogos ({fase_selecionada if fase_selecionada != 'Todas' else 'Todas as Fases'})")
    
    if df_edicao.empty:
        st.write("Nenhum jogo encontrado para os filtros selecionados.")
    else:
        for _, match in df_edicao.sort_values("data_hora").iterrows():
            gm = match['gols_mandante']
            gv = match['gols_visitante']
            if gm is not None and gv is not None:
                if gm > gv:
                    border_class = "border-win-m"
                elif gm < gv:
                    border_class = "border-win-v"
                else:
                    border_class = "border-draw"
            else:
                border_class = ""
                
            fase_exibicao = format_fase(match['fase'], match.get('grupo'))
            mandante_flag = get_flag_html(match['mandante_nome'], width=28)
            visitante_flag = get_flag_html(match['visitante_nome'], width=28)
            st.markdown(f"""
            <div class="glass-card {border_class}" style="padding: 12px 24px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; color:#64748b; font-size:12px; margin-bottom:6px;">
                    <span>{fase_exibicao} | {match['data_hora']}</span>
                    <span>Copa do Mundo {match['ano_copa']}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:1; display:flex; justify-content:flex-end; align-items:center; gap:8px; text-align:right; font-weight:600; font-size:18px; color:#1e293b; min-width:0;">
                        <span style="display:inline-flex; align-items:center;">{mandante_flag}</span>
                        <span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{match['mandante_nome']}</span>
                        <span style="color:#64748b; font-size:14px; font-weight:400;">({match['mandante_sigla']})</span>
                    </div>
                    <div style="background:rgba(37,99,235,0.08); padding:6px 20px; border-radius:30px; font-weight:800; font-size:20px; color:#2563eb; margin:0 20px; border: 1px solid rgba(37,99,235,0.2);">
                        {match['gols_mandante']} - {match['gols_visitante']}
                    </div>
                    <div style="flex:1; display:flex; justify-content:flex-start; align-items:center; gap:8px; text-align:left; font-weight:600; font-size:18px; color:#1e293b; min-width:0;">
                        <span style="color:#64748b; font-size:14px; font-weight:400;">({match['visitante_sigla']})</span>
                        <span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{match['visitante_nome']}</span>
                        <span style="display:inline-flex; align-items:center;">{visitante_flag}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
