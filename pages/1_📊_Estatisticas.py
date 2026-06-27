import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Importações locais do projeto
from src.config import get_api_key
from src.database import init_db, load_historical_matches, load_all_teams, load_2026_matches, sync_openfootball_finished_matches, evaluate_finished_predictions, load_prediction_evaluations
from src.ML_models import calculate_team_strengths, run_tournament_simulation, simulate_single_bracket, predict_match_probabilities
from src.model_calibration import build_model_calibration
from src.styles import inject_css
from src.statistics import build_team_stats
from src.utils import get_flag, format_fase, TEAM_FLAGS, get_flag_html, get_player_photo_url
from src.scraper import fetch_news_rss, fetch_weather_forecast, calculate_match_odds, get_probable_lineup, get_match_venue, fetch_match_specific_news
from datetime import datetime
import html
import functools

# Mapeamento para as fases reais do mata-mata da Copa de 2026
PHASE_MAPPING = {
    "R32": ["Fase de 32", "Dezesseis-avos", "Round of 32"],
    "R16": ["Oitavas de Final", "Oitavas", "Round of 16"],
    "QF": ["Quartas de Final", "Quartas", "Quarter-finals"],
    "SF": ["Semifinais", "Semifinal", "Semi-finals"],
    "F": ["Final"]
}

@st.cache_data(show_spinner=False)
def get_cached_player_photo(name):
    return get_player_photo_url(name)


def format_match_date(date_str):
    """Formata datas do banco para o padrão brasileiro do Google (ex: Sáb., 04/07, 18:00)."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return date_str
            
    weekdays = ["Seg.", "Ter.", "Qua.", "Qui.", "Sex.", "Sáb.", "Dom."]
    wday = weekdays[dt.weekday()]
    return f"{wday}, {dt.strftime('%d/%m, %H:%M')}"

def calculate_real_group_standings(df_2026, team_elo):
    """Computa os dados da tabela de classificação real para a Copa de 2026."""
    df_group = df_2026[df_2026["fase"].isin(["1", "2", "3"])].copy()
    
    # Identificar times em cada grupo
    group_teams = {}
    for _, row in df_group.iterrows():
        g = row["grupo"]
        if not g:
            continue
        if g not in group_teams:
            group_teams[g] = set()
        group_teams[g].add(row["mandante_nome"])
        group_teams[g].add(row["visitante_nome"])
        
    def compare_teams_real(a, b, df_g):
        # 1. Pts
        if a["Pts"] != b["Pts"]:
            return a["Pts"] - b["Pts"]
        # 2. SG
        if a["SG"] != b["SG"]:
            return a["SG"] - b["SG"]
        # 3. GM
        if a["GM"] != b["GM"]:
            return a["GM"] - b["GM"]
            
        # 4. Confronto Direto (H2H)
        h2h_pts_a = 0
        h2h_pts_b = 0
        h2h_gf_a = 0
        h2h_gf_b = 0
        
        h2h_matches = df_g[
            ((df_g["mandante_nome"] == a["team"]) & (df_g["visitante_nome"] == b["team"])) |
            ((df_g["mandante_nome"] == b["team"]) & (df_g["visitante_nome"] == a["team"]))
        ]
        
        for _, match in h2h_matches.iterrows():
            if match["status"] != "FINISHED":
                continue
            m = match["mandante_nome"]
            v = match["visitante_nome"]
            gm = match["gols_mandante"]
            gv = match["gols_visitante"]
            if gm is None or gv is None or pd.isna(gm) or pd.isna(gv):
                continue
            
            gm = int(gm)
            gv = int(gv)
            
            if m == a["team"] and v == b["team"]:
                h2h_gf_a += gm
                h2h_gf_b += gv
                if gm > gv:
                    h2h_pts_a += 3
                elif gv > gm:
                    h2h_pts_b += 3
                else:
                    h2h_pts_a += 1
                    h2h_pts_b += 1
            elif m == b["team"] and v == a["team"]:
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
            
        # 5. Fallback ELO
        if a["elo"] != b["elo"]:
            return 1 if a["elo"] > b["elo"] else -1
        return 0

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
                
                # Tratar valores nulos (jogos SCHEDULED)
                if pd.isna(gols_pro): gols_pro = 0
                if pd.isna(gols_concedidos): gols_concedidos = 0
                
                if status == "FINISHED":
                    pj += 1
                    gm += int(gols_pro)
                    gc += int(gols_concedidos)
                    
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
            
        group_rows.sort(key=functools.cmp_to_key(lambda x, y: compare_teams_real(x, y, df_group)), reverse=True)
        standings[g] = group_rows
        
    return standings

def get_knockout_stage_matches_real(stage, df_2026, team_elo, df_teams):
    """Busca as partidas do mata-mata real no banco de dados, ou retorna placeholders oficiais com datas reais."""
    phases = PHASE_MAPPING[stage]
    df_stage = df_2026[df_2026["fase"].isin(phases)].copy()
    df_stage = df_stage.sort_values("data_hora")
    
    team_id_to_name = {row["id"]: row["nome"] for _, row in df_teams.iterrows()}
    
    if len(df_stage) > 0:
        results = []
        for _, row in df_stage.iterrows():
            t1 = row["mandante_nome"]
            t2 = row["visitante_nome"]
            status = row["status"]
            g_m = row["gols_mandante"]
            g_v = row["gols_visitante"]
            
            score_str = ""
            winner = None
            if status == "FINISHED":
                score_str = f"{int(g_m)} - {int(g_v)}" if not pd.isna(g_m) and not pd.isna(g_v) else ""
                
                w_id = row.get("vencedor_id")
                if pd.notna(w_id) and w_id is not None:
                    winner = team_id_to_name.get(w_id)
                
                if not winner:
                    if g_m > g_v:
                        winner = t1
                    elif g_v > g_m:
                        winner = t2
                    else:
                        winner = t1 if team_elo.get(t1, 0) > team_elo.get(t2, 0) else t2
            
            results.append({
                "t1": t1,
                "t2": t2,
                "score": score_str,
                "w": winner,
                "data_hora": row["data_hora"]
            })
        return results
        
    placeholders = {
        "R32": [
            ("2º Grupo A", "2º Grupo B", "2026-06-28 16:00"),
            ("1º Grupo E", "3º Grupo A/B/C/D", "2026-06-28 20:00"),
            ("1º Grupo F", "2º Grupo C", "2026-06-29 14:00"),
            ("1º Grupo C", "2º Grupo F", "2026-06-29 18:00"),
            ("1º Grupo I", "3º Grupo E/F/G/H", "2026-06-29 22:00"),
            ("2º Grupo E", "2º Grupo I", "2026-06-30 14:00"),
            ("1º Grupo A", "3º Grupo I/J/K/L", "2026-06-30 18:00"),
            ("1º Grupo L", "3º Grupo A/B/C/D", "2026-06-30 22:00"),
            ("1º Grupo D", "3º Grupo E/F/G/H", "2026-07-01 14:00"),
            ("1º Grupo G", "3º Grupo I/J/K/L", "2026-07-01 18:00"),
            ("2º Grupo K", "2º Grupo L", "2026-07-01 22:00"),
            ("1º Grupo H", "2º Grupo J", "2026-07-02 14:00"),
            ("1º Grupo B", "3º Grupo E/F/G/H", "2026-07-02 18:00"),
            ("1º Grupo J", "2º Grupo H", "2026-07-02 22:00"),
            ("1º Grupo K", "3º Grupo A/B/C/D", "2026-07-03 14:00"),
            ("2º Grupo D", "2º Grupo G", "2026-07-03 18:00")
        ],
        "R16": [
            ("Venc. 74", "Venc. 77", "2026-07-04 14:00"),
            ("Venc. 73", "Venc. 75", "2026-07-04 18:00"),
            ("Venc. 76", "Venc. 78", "2026-07-05 14:00"),
            ("Venc. 79", "Venc. 80", "2026-07-05 18:00"),
            ("Venc. 83", "Venc. 84", "2026-07-06 14:00"),
            ("Venc. 81", "Venc. 82", "2026-07-06 18:00"),
            ("Venc. 86", "Venc. 88", "2026-07-07 14:00"),
            ("Venc. 85", "Venc. 87", "2026-07-07 18:00")
        ],
        "QF": [
            ("Venc. 89", "Venc. 90", "2026-07-09 14:00"),
            ("Venc. 93", "Venc. 94", "2026-07-09 18:00"),
            ("Venc. 91", "Venc. 92", "2026-07-10 14:00"),
            ("Venc. 95", "Venc. 96", "2026-07-10 18:00")
        ],
        "SF": [
            ("Venc. 97", "Venc. 98", "2026-07-14 20:00"),
            ("Venc. 99", "Venc. 100", "2026-07-15 20:00")
        ],
        "F": [
            ("Venc. 101", "Venc. 102", "2026-07-19 16:00")
        ]
    }
    
    results = []
    for t1, t2, date_str in placeholders[stage]:
        results.append({
            "t1": t1,
            "t2": t2,
            "score": "",
            "w": None,
            "data_hora": date_str
        })
    return results

def clean_html(html_str):
    """Remove a indentação inicial de todas as linhas de uma string HTML/CSS para evitar renderização como bloco de código no Streamlit."""
    return "\n".join(line.strip() for line in html_str.splitlines())

def format_bracket_team_html(team_name, is_winner, score_val, is_finished):
    """Renderiza uma linha de equipe no mata-mata real no padrão do Google."""
    if not team_name:
        team_name = "A definir"
        
    is_real = team_name in TEAM_FLAGS
    
    row_class = ""
    if is_finished:
        if is_winner:
            row_class = "winner"
        else:
            row_class = "loser"
            
    score_html = ""
    if is_finished and score_val is not None:
        score_class = "loser" if not is_winner else ""
        score_html = f'<span class="google-match-score {score_class}">{score_val}</span>'
        
    if is_real:
        flag_html = get_flag_html(team_name, width=18)
        team_content = f'{flag_html} <span>{team_name}</span>'
    else:
        team_content = f'<span style="color: #94a3b8; display: flex; align-items: center; gap: 6px;">🛡️ {team_name}</span>'
        
    return f"""
    <div class="google-match-team-row {row_class}">
        <div class="google-match-team-info">
            {team_content}
        </div>
        {score_html}
    </div>
    """

def build_real_round_col_html(title, matches, is_final=False):
    """Gera a estrutura HTML de uma coluna da árvore de mata-mata real."""
    html = f'<div class="google-bracket-col"><div class="google-bracket-header">{title}</div>'
    for m in matches:
        t1 = m["t1"]
        t2 = m["t2"]
        w = m["w"]
        score_str = m["score"]
        date_str = m["data_hora"]
        
        s1, s2 = None, None
        if score_str:
            parts = score_str.split(" - ")
            if len(parts) == 2:
                s1, s2 = parts[0], parts[1]
                
        is_finished = score_str != ""
        t1_winner = w == t1 if w else False
        t2_winner = w == t2 if w else False
        
        formatted_date = format_match_date(date_str)
        
        card_style = ""
        if is_final:
            card_style = 'style="border: 2px solid #f59e0b; box-shadow: 0 4px 12px rgba(245,158,11,0.15);"'
            
        t1_html = format_bracket_team_html(t1, t1_winner, s1, is_finished)
        t2_html = format_bracket_team_html(t2, t2_winner, s2, is_finished)
        
        html += f"""
        <div class="google-match-card-wrapper">
            <div class="google-match-time">{formatted_date}</div>
            <div class="google-match-card" {card_style}>
                {t1_html}
                {t2_html}
            </div>
        </div>
        """
    html += '</div>'
    return clean_html(html)

def build_group_standings_html(group_name, standings_rows):
    """Gera a tabela HTML de classificação de um grupo no padrão Google Dark."""
    html = f"""
    <div class="google-standings-card">
        <div class="google-standings-title">Grupo {group_name}</div>
        <table class="google-standings-table">
            <thead>
                <tr>
                    <th style="width: 20px;">#</th>
                    <th>Equipe</th>
                    <th style="text-align: center; width: 30px;">Pts</th>
                    <th style="text-align: center; width: 30px;">PJ</th>
                    <th style="text-align: center; width: 30px;">VIT</th>
                    <th style="text-align: center; width: 30px;">E</th>
                    <th style="text-align: center; width: 30px;">DER</th>
                    <th style="text-align: center; width: 30px;">GM</th>
                    <th style="text-align: center; width: 30px;">GC</th>
                    <th style="text-align: center; width: 30px;">SG</th>
                    <th style="text-align: center; width: 110px;">Últimas 5</th>
                </tr>
            </thead>
            <tbody>
    """
    for rank, row in enumerate(standings_rows, 1):
        team = row["team"]
        pts = int(row["Pts"])
        pj = int(row["PJ"])
        vit = int(row["VIT"])
        e = int(row["E"])
        der = int(row["DER"])
        gm = int(row["GM"])
        gc = int(row["GC"])
        sg = int(row["SG"])
        form = row["form"]
        
        flag_html = get_flag_html(team, width=18)
        
        form_html = '<div class="form-container">'
        for f in form:
            if f == "V":
                form_html += '<div class="form-circle win">✓</div>'
            elif f == "D":
                form_html += '<div class="form-circle loss">✗</div>'
            elif f == "E":
                form_html += '<div class="form-circle draw">−</div>'
            else:
                form_html += '<div class="form-circle empty"></div>'
        form_html += '</div>'
        
        sg_str = f"+{sg}" if sg > 0 else f"{sg}"
        
        html += f"""
        <tr>
            <td><span class="rank-num">{rank}</span></td>
            <td>
                <div class="team-cell">
                    {flag_html}
                    <span>{team}</span>
                </div>
            </td>
            <td style="text-align: center;" class="pts-col">{pts}</td>
            <td style="text-align: center;">{pj}</td>
            <td style="text-align: center;">{vit}</td>
            <td style="text-align: center;">{e}</td>
            <td style="text-align: center;">{der}</td>
            <td style="text-align: center;">{gm}</td>
            <td style="text-align: center;">{gc}</td>
            <td style="text-align: center;">{sg_str}</td>
            <td>{form_html}</td>
        </tr>
        """
    html += """
            </tbody>
        </table>
    </div>
    """
    return clean_html(html)


# Configuração da página
st.set_page_config(page_title="Estatísticas - FuteBot", page_icon="📊", layout="wide")

init_db()
if "openfootball_sync_2026" not in st.session_state:
    st.session_state["openfootball_sync_2026"] = sync_openfootball_finished_matches(2026)
    evaluate_finished_predictions()

# CSS compartilhado (tema branco)
inject_css()

# Configurações na Sidebar
st.sidebar.markdown("### ⚙️ Configurações")
api_key = st.sidebar.text_input(
    "🔑 Football-Data.org API Token:",
    value=get_api_key(st),
    key="api_key",
    type="password",
    help="Cole sua chave gratuita do site football-data.org para ver jogos oficiais da Copa 2026 em tempo real."
)

if api_key:
    from src.api_client import fetch_live_matches_from_api
    from src.database import sync_api_match_to_db
    # Sincronizar dados silenciosamente
    live_games_api, status_api = fetch_live_matches_from_api(api_key)
    if live_games_api:
        if "db_synced_once" not in st.session_state:
            st.session_state["db_synced_once"] = True
            
        for game in live_games_api:
            sync_api_match_to_db(game)
        evaluate_finished_predictions()
            
    # Exibir status de conexão na sidebar
    if "✅" in status_api:
        st.sidebar.success(status_api)
    elif "⚠️" in status_api:
        st.sidebar.warning(status_api)
    else:
        st.sidebar.info(status_api)

st.title("📊 Estatísticas e Simulação")
st.write("Explore o desempenho das seleções, analise os coeficientes de ataque/defesa e acompanhe as probabilidades de avanço dinâmicas.")

# Carregar dados
df_matches = load_historical_matches(include_seed_2026=True)
df_teams = load_all_teams()
df_prediction_evaluations = load_prediction_evaluations()
model_calibration = build_model_calibration(load_historical_matches(), df_prediction_evaluations)

# Função de simulação cacheada
@st.cache_data
def get_cached_simulation(matches_len, elo_sum, evaluated_predictions):
    df_m = load_historical_matches()
    df_t = load_all_teams()
    df_2026 = load_2026_matches()
    calibration = build_model_calibration(df_m, load_prediction_evaluations())
    return run_tournament_simulation(df_m, df_t, df_2026, iterations=150, calibration=calibration)

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

    # 1. Calcular Forças pelo modelo de Poisson (com suavização Laplaciana para estabilidade)
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
        
    df_stats = build_team_stats(df_matches_filtered, df_teams, ataque, defesa)
    
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
            <div class="ai-insight-card best-attack">
                <div>
                    <div style="display: flex; justify-content: center;">
                        <span class="ai-insight-badge best-attack">🔥 Melhor Ataque</span>
                    </div>
                    <div class="ai-insight-team">
                        {get_flag_html(best_att_row["Seleção"], width=22)}
                        <span>{best_att_row["Seleção"]}</span>
                    </div>
                </div>
                <div class="ai-insight-details">
                    <div class="ai-insight-row">
                        <span>Coeficiente:</span>
                        <strong style="color: #10b981;">{best_att_row["Ataque (Fator)"]:.2f}x</strong>
                    </div>
                    <div class="ai-insight-row">
                        <span>Gols marcados:</span>
                        <strong style="color: #1e293b;">{best_att_row["Gols Pró"]}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_ins2:
            st.markdown(f"""
            <div class="ai-insight-card worst-attack">
                <div>
                    <div style="display: flex; justify-content: center;">
                        <span class="ai-insight-badge worst-attack">❄️ Pior Ataque</span>
                    </div>
                    <div class="ai-insight-team">
                        {get_flag_html(worst_att_row["Seleção"], width=22)}
                        <span>{worst_att_row["Seleção"]}</span>
                    </div>
                </div>
                <div class="ai-insight-details">
                    <div class="ai-insight-row">
                        <span>Coeficiente:</span>
                        <strong style="color: #ef4444;">{worst_att_row["Ataque (Fator)"]:.2f}x</strong>
                    </div>
                    <div class="ai-insight-row">
                        <span>Gols marcados:</span>
                        <strong style="color: #1e293b;">{worst_att_row["Gols Pró"]}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_ins3:
            st.markdown(f"""
            <div class="ai-insight-card best-defense">
                <div>
                    <div style="display: flex; justify-content: center;">
                        <span class="ai-insight-badge best-defense">🛡️ Melhor Defesa</span>
                    </div>
                    <div class="ai-insight-team">
                        {get_flag_html(best_def_row["Seleção"], width=22)}
                        <span>{best_def_row["Seleção"]}</span>
                    </div>
                </div>
                <div class="ai-insight-details">
                    <div class="ai-insight-row">
                        <span>Coeficiente:</span>
                        <strong style="color: #3b82f6;">{best_def_row["Defesa (Fator)"]:.2f}x</strong>
                    </div>
                    <div class="ai-insight-row">
                        <span>Gols sofridos:</span>
                        <strong style="color: #1e293b;">{best_def_row["Gols Contra"]}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_ins4:
            st.markdown(f"""
            <div class="ai-insight-card worst-defense">
                <div>
                    <div style="display: flex; justify-content: center;">
                        <span class="ai-insight-badge worst-defense">⚠️ Pior Defesa</span>
                    </div>
                    <div class="ai-insight-team">
                        {get_flag_html(worst_def_row["Seleção"], width=22)}
                        <span>{worst_def_row["Seleção"]}</span>
                    </div>
                </div>
                <div class="ai-insight-details">
                    <div class="ai-insight-row">
                        <span>Coeficiente:</span>
                        <strong style="color: #f59e0b;">{worst_def_row["Defesa (Fator)"]:.2f}x</strong>
                    </div>
                    <div class="ai-insight-row">
                        <span>Gols sofridos:</span>
                        <strong style="color: #1e293b;">{worst_def_row["Gols Contra"]}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Seletor de time para análise individual
    selected_team = st.sidebar.selectbox("Escolha uma Seleção para análise detalhada:", sorted(df_stats["Seleção"].unique()))
    
    # Estilos CSS estilo Google Dark injetados no Streamlit
    st.markdown(clean_html("""
    <style>
        /* AI Insights Premium Cards */
        .ai-insight-card {
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            background: #ffffff;
            padding: 18px 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
            text-align: center;
            font-family: 'Outfit', sans-serif;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            height: 100%;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        }
        .ai-insight-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 20px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
            border-color: #cbd5e1;
        }
        .ai-insight-card.best-attack { border-top: 4px solid #10b981; }
        .ai-insight-card.worst-attack { border-top: 4px solid #ef4444; }
        .ai-insight-card.best-defense { border-top: 4px solid #3b82f6; }
        .ai-insight-card.worst-defense { border-top: 4px solid #f59e0b; }

        .ai-insight-badge {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 4px 10px;
            border-radius: 9999px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            margin-bottom: 12px;
        }
        .ai-insight-badge.best-attack { color: #065f46; background: #d1fae5; }
        .ai-insight-badge.worst-attack { color: #991b1b; background: #fee2e2; }
        .ai-insight-badge.best-defense { color: #1e40af; background: #dbeafe; }
        .ai-insight-badge.worst-defense { color: #92400e; background: #fef3c7; }

        .ai-insight-team {
            font-size: 16px;
            font-weight: 700;
            color: #1e293b;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            margin: 8px 0;
        }
        .ai-insight-details {
            margin-top: auto;
            padding-top: 10px;
            border-top: 1px solid #f1f5f9;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .ai-insight-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
        }
        .ai-insight-row span {
            color: #64748b;
        }
        .ai-insight-row strong {
            font-weight: 700;
        }

        /* Estilos Classificação Estilo Google Light */
        .google-standings-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            font-family: 'Outfit', sans-serif;
        }
        .google-standings-title {
            color: #1e293b;
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
        }
        .google-standings-table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }
        .google-standings-table th {
            color: #64748b;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            padding: 8px 4px;
            border-bottom: 1px solid #e2e8f0;
        }
        .google-standings-table td {
            color: #334155;
            font-size: 13px;
            padding: 10px 4px;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: middle;
        }
        .google-standings-table tr:last-child td {
            border-bottom: none;
        }
        .google-standings-table tr:hover td {
            background-color: #f8fafc;
        }
        .team-cell {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #1e293b;
            font-weight: 500;
        }
        .team-flag {
            font-size: 16px;
        }
        .rank-num {
            color: #64748b;
            font-size: 12px;
            font-weight: 600;
            width: 16px;
            display: inline-block;
            text-align: center;
        }
        .pts-col {
            font-weight: 700;
            color: #1e293b;
        }
        .form-container {
            display: flex;
            gap: 3px;
            align-items: center;
        }
        .form-circle {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: bold;
            color: white;
        }
        .form-circle.win {
            background-color: #10b981;
        }
        .form-circle.loss {
            background-color: #ef4444;
        }
        .form-circle.draw {
            background-color: #64748b;
        }
        .form-circle.empty {
            border: 1px solid #cbd5e1;
            background-color: transparent;
            width: 14px;
            height: 14px;
        }

        /* Estilos Mata-Mata Estilo Google Light */
        .google-bracket-container {
            display: flex;
            flex-direction: row;
            gap: 20px;
            overflow-x: auto;
            padding: 24px;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            align-items: stretch;
            min-height: 950px;
            margin-bottom: 20px;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.02);
        }
        .google-bracket-col {
            display: flex;
            flex-direction: column;
            justify-content: space-around;
            min-width: 240px;
            flex: 1;
            gap: 12px;
        }
        .google-bracket-header {
            font-size: 12px;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            text-align: center;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
            margin-bottom: 12px;
            letter-spacing: 0.5px;
        }
        .google-match-card-wrapper {
            margin: 6px 0;
        }
        .google-match-time {
            font-size: 11px;
            color: #64748b;
            margin-bottom: 4px;
            padding-left: 4px;
            font-weight: 500;
        }
        .google-match-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        }
        .google-match-team-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0;
            font-size: 13px;
            color: #475569;
        }
        .google-match-team-row.winner {
            font-weight: 700;
            color: #1e293b;
        }
        .google-match-team-row.loser {
            color: #94a3b8;
        }
        .google-match-team-info {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .google-match-team-flag {
            font-size: 16px;
        }
        .google-match-team-placeholder {
            color: #94a3b8;
            font-style: italic;
        }
        .google-match-score {
            font-weight: 700;
            font-size: 13px;
            color: #1e293b;
            font-family: monospace;
        }
        .google-match-score.loser {
            color: #94a3b8;
        }
    </style>
    """), unsafe_allow_html=True)

    # Abas reestruturadas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Fase de Grupos", 
        "🏆 Mata-Mata Real",
        "🌎 Força das Seleções (ELO)",
        "📈 Chances de Avanço (Monte Carlo)", 
        "🔍 Análise por Seleção"
    ])
    
    with tab1:
        st.subheader("Classificação da Fase de Grupos - Copa 2026")
        st.write("Acompanhe a tabela de classificação oficial calculada dinamicamente com base nas partidas finalizadas no banco de dados.")
        
        team_elo = {row["nome"]: row["elo_rating"] for _, row in df_teams.iterrows()}
        df_2026 = load_2026_matches()
        standings = calculate_real_group_standings(df_2026, team_elo)
        
        col_g1, col_g2 = st.columns(2)
        groups_list = sorted(list(standings.keys()))
        for idx, g_name in enumerate(groups_list):
            g_html = build_group_standings_html(g_name, standings[g_name])
            if idx % 2 == 0:
                with col_g1:
                    st.markdown(g_html, unsafe_allow_html=True)
            else:
                with col_g2:
                    st.markdown(g_html, unsafe_allow_html=True)
                    
    with tab2:
        st.subheader("Chaveamento do Mata-Mata Real")
        st.write("Acompanhe o chaveamento real da fase eliminatória da Copa do Mundo 2026. Slots em cinza representam posições ainda a serem definidas com base na conclusão da fase de grupos.")
        
        team_elo = {row["nome"]: row["elo_rating"] for _, row in df_teams.iterrows()}
        df_2026 = load_2026_matches()
        
        # Obter confrontos de cada fase
        r32_matches = get_knockout_stage_matches_real("R32", df_2026, team_elo, df_teams)
        r16_matches = get_knockout_stage_matches_real("R16", df_2026, team_elo, df_teams)
        qf_matches = get_knockout_stage_matches_real("QF", df_2026, team_elo, df_teams)
        sf_matches = get_knockout_stage_matches_real("SF", df_2026, team_elo, df_teams)
        f_matches = get_knockout_stage_matches_real("F", df_2026, team_elo, df_teams)
        
        # Gerar colunas HTML
        r32_col_html = build_real_round_col_html("Segunda rodada", r32_matches)
        r16_col_html = build_real_round_col_html("Oitavas de final", r16_matches)
        qf_col_html = build_real_round_col_html("Quartas de final", qf_matches)
        sf_col_html = build_real_round_col_html("Semifinais", sf_matches)
        f_col_html = build_real_round_col_html("Grande Final", f_matches, is_final=True)
        
        # Renderizar container da árvore de mata-mata
        st.markdown(clean_html(f"""
        <div class="google-bracket-container">
            {r32_col_html}
            {r16_col_html}
            {qf_col_html}
            {sf_col_html}
            {f_col_html}
        </div>
        """), unsafe_allow_html=True)

    with tab3:
        st.subheader("Desempenho Geral das Seleções (Força ELO)")
        st.write("Esta tabela resume o poder ofensivo, defensivo e saldo de gols de cada equipe. Valores de Ataque acima de 1.0 indicam um ataque melhor que a média; valores de Defesa abaixo de 1.0 indicam uma defesa mais sólida que a média.")
        
        # Formatar tabela para exibição elegante com bandeiras oficiais via ImageColumn
        df_display = df_stats.copy()
        
        from src.utils import TEAM_CODES
        df_display["Bandeira"] = df_display["Seleção"].apply(
            lambda name: f"https://flagcdn.com/w40/{TEAM_CODES[name]}.png" if name in TEAM_CODES else None
        )
        
        # Reordenar colunas para colocar a bandeira na frente de tudo
        cols = ["Bandeira"] + [col for col in df_display.columns if col != "Bandeira"]
        df_display = df_display[cols]
        
        st.dataframe(
            df_display.sort_values("ELO Rating", ascending=False),
            column_config={
                "Bandeira": st.column_config.ImageColumn(
                    "Bandeira",
                    help="Bandeira oficial da seleção.",
                    width="small"
                ),
                "Seleção": st.column_config.TextColumn(
                    "Seleção",
                    help="Nome da seleção nacional."
                ),
                "ELO Rating": st.column_config.ProgressColumn(
                    "ELO Rating",
                    help="Pontuação de força ELO baseada em resultados históricos e recentes.",
                    format="%.0f",
                    min_value=1300,
                    max_value=2200,
                ),
                "Ranking FIFA": st.column_config.NumberColumn(
                    "Rank FIFA",
                    help="Classificação oficial da FIFA.",
                    format="%d",
                ),
                "Ataque (Fator)": st.column_config.NumberColumn(
                    "Poder de Ataque",
                    help="Fator de gols marcados em relação à média mundial (gols esperados por partida).",
                    format="%.2f",
                ),
                "Defesa (Fator)": st.column_config.NumberColumn(
                    "Vulnerabilidade de Defesa",
                    help="Fator de gols sofridos em relação à média mundial (quanto menor, mais sólida).",
                    format="%.2f",
                ),
                "Saldo de Gols": st.column_config.NumberColumn(
                    "Saldo",
                    format="%+d",
                ),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Gráficos de comparação
        col1, col2 = st.columns(2)
        with col1:
            fig_elo = px.bar(
                df_stats.sort_values("ELO Rating", ascending=False).head(15),
                x="ELO Rating", y="Seleção",
                orientation="h",
                title="Top 15 Seleções por ELO Rating",
                color="ELO Rating",
                color_continuous_scale=["#3b82f6", "#7c3aed"], # Degradê de azul a violeta
                template="plotly_white"
            )
            fig_elo.update_layout(
                xaxis_title="ELO Rating",
                yaxis_title="",
                coloraxis_showscale=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Outfit", size=12),
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis=dict(categoryorder="total ascending")
            )
            st.plotly_chart(fig_elo, width="stretch")
            
        with col2:
            # Gráfico de barras agrupadas: Poder de Fogo: Ataque vs Defesa
            df_chart = df_stats.sort_values("ELO Rating", ascending=False).head(12).copy()
            df_chart["Força da Defesa"] = 1.0 / df_chart["Defesa (Fator)"]
            df_chart = df_chart.rename(columns={
                "Ataque (Fator)": "Força do Ataque"
            })
            
            df_melted = df_chart.melt(
                id_vars=["Seleção"],
                value_vars=["Força do Ataque", "Força da Defesa"],
                var_name="Métrica",
                value_name="Poder (Fator)"
            )
            
            fig_xg = px.bar(
                df_melted,
                y="Seleção",
                x="Poder (Fator)",
                color="Métrica",
                barmode="group",
                orientation="h",
                title="Poder de Fogo: Ataque vs Defesa (Top 12 Seleções)",
                color_discrete_map={
                    "Força do Ataque": "#10b981", # Verde Esmeralda
                    "Força da Defesa": "#3b82f6"  # Azul
                },
                template="plotly_white"
            )
            
            team_order = df_chart["Seleção"].tolist()[::-1]
            fig_xg.update_layout(
                xaxis_title="Fator de Poder (1.0 = Média Mundial)",
                yaxis_title="",
                legend_title="Métrica",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Outfit", size=12),
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis=dict(categoryorder="array", categoryarray=team_order)
            )
            st.plotly_chart(fig_xg, width="stretch")
            
    with tab4:
        st.subheader("🔮 Probabilidades de Avanço da Copa do Mundo 2026")
        st.write("A Inteligência Artificial simulou o restante da Copa do Mundo 2026 (fase de grupos e mata-mata) por 150 vezes, considerando o ELO Rating atual e as forças de ataque e defesa de cada time. Os resultados abaixo mostram as probabilidades estimadas de alcançar cada fase do torneio.")
        
        # Obter simulação cacheada
        m2026 = df_matches[df_matches["ano_copa"] == 2026]
        matches_len = len(m2026[m2026["status"] == "FINISHED"])
        elo_sum = sum(df_teams["elo_rating"])
        
        df_probs = get_cached_simulation(matches_len, elo_sum, len(df_prediction_evaluations))
        
        # Formatar exibição com ordenação correta e bandeiras oficiais
        df_probs_display = df_probs.copy()
        df_probs_display = df_probs_display.sort_values("Campeão (%)", ascending=False)
        
        from src.utils import TEAM_CODES
        df_probs_display["Bandeira"] = df_probs_display["Seleção"].apply(
            lambda name: f"https://flagcdn.com/w40/{TEAM_CODES[name]}.png" if name in TEAM_CODES else None
        )
        
        # Reordenar colunas para colocar a bandeira na frente
        cols = ["Bandeira"] + [col for col in df_probs_display.columns if col != "Bandeira"]
        df_probs_display = df_probs_display[cols]
        
        st.dataframe(
            df_probs_display,
            column_config={
                "Bandeira": st.column_config.ImageColumn(
                    "Bandeira",
                    help="Bandeira oficial da seleção.",
                    width="small"
                ),
                "Seleção": st.column_config.TextColumn(
                    "Seleção",
                    help="Nome da seleção nacional."
                ),
                "Quartas (%)": st.column_config.NumberColumn(
                    "Quartas de Final",
                    help="Probabilidade estimada de chegar às Quartas de Final.",
                    format="%.1f%%"
                ),
                "Semis (%)": st.column_config.NumberColumn(
                    "Semifinais",
                    help="Probabilidade estimada de chegar às Semifinais.",
                    format="%.1f%%"
                ),
                "Finais (%)": st.column_config.NumberColumn(
                    "Final",
                    help="Probabilidade estimada de chegar à Grande Final.",
                    format="%.1f%%"
                ),
                "Campeão (%)": st.column_config.ProgressColumn(
                    "Campeão da Copa",
                    help="Probabilidade estimada de vencer a Copa do Mundo 2026.",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0
                ),
            },
            hide_index=True,
            use_container_width=True
        )
        
    with tab5:
        if selected_team and not df_stats.empty and (df_stats["Seleção"] == selected_team).any():
            team_stats = df_stats[df_stats["Seleção"] == selected_team].iloc[0]
        else:
            team_stats = None
            
        if team_stats is not None:
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
        else:
            st.info(f"Dados estatísticos para {selected_team} indisponíveis no filtro atual.")
        
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
                "Competição / Fase": format_fase(match["fase"], match.get("grupo")),
                "Oponente_Nome": oponente,
                "Gols Marcados": gols_pro,
                "Gols Sofridos": gols_contra,
                "Resultado": resultado
            })
            
        if not formatted_matches:
            st.info("Nenhuma partida recente registrada.")
        else:
            table_rows = []
            for m in formatted_matches:
                res = m["Resultado"]
                if "Vitória" in res:
                    res_badge = '<span style="background-color: #d1fae5; color: #059669; padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; display: inline-block;">✅ Vitória</span>'
                elif "Empate" in res:
                    res_badge = '<span style="background-color: #f1f5f9; color: #475569; padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; display: inline-block;">➖ Empate</span>'
                else:
                    res_badge = '<span style="background-color: #fef2f2; color: #ef4444; padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; display: inline-block;">❌ Derrota</span>'

                table_rows.append(f"""
                <tr style="border-bottom: 1px solid #e2e8f0; font-size: 13px;">
                    <td style="padding: 12px 10px; color: #64748b; font-weight: 500;">{m['Data']}</td>
                    <td style="padding: 12px 10px; color: #1e293b; font-weight: 500;">{m['Competição / Fase']}</td>
                    <td style="padding: 12px 10px; color: #1e293b; font-weight: 600;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            {get_flag_html(m['Oponente_Nome'], width=20)} 
                            <span>{m['Oponente_Nome']}</span>
                        </div>
                    </td>
                    <td style="padding: 12px 10px; text-align: center; color: #1e293b; font-weight: 700;">{m['Gols Marcados']}</td>
                    <td style="padding: 12px 10px; text-align: center; color: #64748b; font-weight: 500;">{m['Gols Sofridos']}</td>
                    <td style="padding: 12px 10px;">{res_badge}</td>
                </tr>
                """)

            html_table = f"""
            <div style="overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.02); background: white;">
                <table style="width: 100%; border-collapse: collapse; text-align: left; font-family: 'Outfit', sans-serif;">
                    <thead>
                        <tr style="background-color: #f8fafc; border-bottom: 2px solid #e2e8f0; font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">
                            <th style="padding: 12px 10px;">Data</th>
                            <th style="padding: 12px 10px;">Competição / Fase</th>
                            <th style="padding: 12px 10px;">Oponente</th>
                            <th style="padding: 12px 10px; text-align: center;">Gols Pró</th>
                            <th style="padding: 12px 10px; text-align: center;">Gols Contra</th>
                            <th style="padding: 12px 10px;">Resultado</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(table_rows)}
                    </tbody>
                </table>
            </div>
            """
            st.markdown(clean_html(html_table), unsafe_allow_html=True)


        # ============================================================
        # SEÇÃO: ANÁLISE PRÉ-JOGO & INTELIGÊNCIA DE MERCADO
        # ============================================================
        st.markdown("---")
        st.markdown("### 🔮 Análise Pré-Jogo & Inteligência de Mercado")

        # Buscar a próxima partida SCHEDULED da seleção selecionada
        df_2026_all = load_2026_matches()
        next_matches = df_2026_all[
            ((df_2026_all["mandante_nome"] == selected_team) | (df_2026_all["visitante_nome"] == selected_team))
            & (df_2026_all["status"] == "SCHEDULED")
        ].sort_values("data_hora")

        if next_matches.empty:
            st.info(f"Não há partidas agendadas para {get_flag(selected_team)} {selected_team} no momento.")
        else:
            next_match = next_matches.iloc[0]
            is_mandante = next_match["mandante_nome"] == selected_team
            oponente = next_match["visitante_nome"] if is_mandante else next_match["mandante_nome"]
            match_date_str = next_match["data_hora"]
            match_grupo = next_match.get("grupo", None)
            match_fase = next_match.get("fase", None)

            # Cabeçalho do confronto
            st.markdown(clean_html(f"""
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; margin-bottom: 20px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.01);">
                <div style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px;">
                    {format_fase(match_fase, match_grupo)} • {format_match_date(match_date_str)}
                </div>
                <div style="font-size: 28px; font-weight: 800; color: #1e293b; display: flex; align-items: center; justify-content: center; gap: 8px;">
                    {get_flag_html(next_match['mandante_nome'], width=36)}
                    <span style="color: #1e293b !important;">{next_match['mandante_nome']}</span>
                    <span style="color: #94a3b8 !important; font-weight: 400; margin: 0 12px;">vs</span>
                    <span style="color: #1e293b !important;">{next_match['visitante_nome']}</span>
                    {get_flag_html(next_match['visitante_nome'], width=36)}
                </div>
            </div>
            """), unsafe_allow_html=True)

            # --- LINHA 1: Clima + Odds ---
            col_weather, col_odds = st.columns([1, 2])

            with col_weather:
                st.markdown("#### 🌤️ Previsão do Tempo")
                venue = get_match_venue(grupo=match_grupo, fase=match_fase)
                weather = fetch_weather_forecast(venue["lat"], venue["lon"], match_date_str, venue.get("tz", "America/New_York"))

                if weather and weather.get("temperatura_c") is not None:
                    st.markdown(clean_html(f"""
                    <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 12px; padding: 20px;">
                        <div style="font-size: 13px; color: #0369a1; font-weight: 600; margin-bottom: 8px;">📍 {venue['cidade']}</div>
                        <div style="font-size: 36px; font-weight: 800; color: #0c4a6e; margin-bottom: 4px;">{weather['temperatura_c']}°C</div>
                        <div style="font-size: 14px; color: #0369a1; margin-bottom: 12px;">{weather['descricao']}</div>
                        <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                            <div style="font-size: 12px; color: #475569;">💧 Umidade: <b>{weather['umidade_pct']}%</b></div>
                            <div style="font-size: 12px; color: #475569;">🌧️ Chuva: <b>{weather['precipitacao_pct']}%</b></div>
                            <div style="font-size: 12px; color: #475569;">💨 Vento: <b>{weather['vento_kmh']} km/h</b></div>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)
                else:
                    st.caption("Dados climáticos indisponíveis (fora do alcance de previsão ou sem conexão).")

            with col_odds:
                st.markdown("#### 📊 Odds de Apostas (Modelo IA)")
                # Calcular probabilidades via modelo Poisson/ELO
                mandante_elo = next_match["mandante_elo"]
                visitante_elo = next_match["visitante_elo"]
                pred = predict_match_probabilities(
                    next_match["mandante_nome"], next_match["visitante_nome"],
                    mandante_elo, visitante_elo, df_matches, calibration=model_calibration
                )
                odds_data = calculate_match_odds(
                    pred["prob_vitoria_mandante"], pred["prob_empate"], pred["prob_vitoria_visitante"]
                )

                if odds_data:
                    # Probabilidades
                    st.markdown(clean_html(f"""
                    <div style="display: flex; gap: 12px; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <!-- Mandante -->
                        <div class="metric-card-premium" style="flex: 1; border-left: 4px solid #2563eb;">
                            <div style="font-size: 13px; color: #64748b; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px; margin-bottom: 4px;">
                                {get_flag_html(next_match['mandante_nome'], width=18)} Vitória
                            </div>
                            <div style="font-size: 28px; font-weight: 800; color: #1e293b;">{odds_data['prob_mandante']}%</div>
                        </div>
                        <!-- Empate -->
                        <div class="metric-card-premium" style="flex: 1; border-left: 4px solid #94a3b8;">
                            <div style="font-size: 13px; color: #64748b; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px; margin-bottom: 4px;">
                                🤝 Empate
                            </div>
                            <div style="font-size: 28px; font-weight: 800; color: #1e293b;">{odds_data['prob_empate']}%</div>
                        </div>
                        <!-- Visitante -->
                        <div class="metric-card-premium" style="flex: 1; border-left: 4px solid #7c3aed;">
                            <div style="font-size: 13px; color: #64748b; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px; margin-bottom: 4px; flex-direction: row-reverse;">
                                {get_flag_html(next_match['visitante_nome'], width=18)} Vitória
                            </div>
                            <div style="font-size: 28px; font-weight: 800; color: #1e293b;">{odds_data['prob_visitante']}%</div>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)


                    # Tabela de odds por casa
                    odds_rows = []
                    for house_name, vals in odds_data["houses"].items():
                        odds_rows.append({
                            "Casa": house_name,
                            f"{next_match['mandante_nome']}": f"{vals['mandante']:.2f}",
                            "Empate": f"{vals['empate']:.2f}",
                            f"{next_match['visitante_nome']}": f"{vals['visitante']:.2f}"
                        })
                    st.dataframe(pd.DataFrame(odds_rows), hide_index=True, use_container_width=True)

                    # Placar mais provável
                    pmp = pred["placar_mais_provavel"]
                    st.caption(f"⚽ Placar mais provável: **{next_match['mandante_nome']} {pmp[0]} x {pmp[1]} {next_match['visitante_nome']}** ({pmp[2]*100:.1f}%)")

            # --- LINHA 2: Escalações ---
            st.markdown("---")
            st.markdown("#### ⚽ Escalações Prováveis")
            col_lineup1, col_lineup2 = st.columns(2)

            for col_l, team_l in [(col_lineup1, next_match["mandante_nome"]), (col_lineup2, next_match["visitante_nome"])]:
                with col_l:
                    lineup = get_probable_lineup(team_l)
                    players_html = ""
                    for i, p in enumerate(lineup["titulares"], 1):
                        bg = "#f0fdf4" if i == 1 else ("#f8fafc" if i % 2 == 0 else "#ffffff")
                        color = "#047857" if i == 1 else "#1e293b"
                        
                        # Buscar foto cacheada do jogador (Wikipedia com fallback initials)
                        photo_url = get_cached_player_photo(p)
                        
                        players_html += f"""
                        <div style="display: flex; align-items: center; padding: 8px 16px; background: {bg}; border-bottom: 1px solid #e2e8f0; gap: 12px;">
                            <span style="font-weight: 700; color: #94a3b8; font-size: 12px; width: 16px; text-align: center;">{i}</span>
                            <img src="{photo_url}" width="32" height="32" style="border-radius: 50%; object-fit: cover; border: 1px solid #cbd5e1; background: #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                            <span style="font-size: 13.5px; font-weight: 600; color: {color}; font-family: 'Outfit', sans-serif;">{p}</span>
                        </div>
                        """

                    st.markdown(clean_html(f"""
                    <div style="border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; margin-bottom: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);">
                        <div style="background: linear-gradient(135deg, #f8fafc, #f1f5f9); border-bottom: 1px solid #e2e8f0; padding: 14px 16px; display: flex; justify-content: space-between; align-items: center;">
                            <div style="color: #1e293b !important; font-weight: 700; font-size: 15px; display: flex; align-items: center; gap: 4px;">
                                {get_flag_html(team_l, width=22)}
                                <span style="color: #1e293b !important;">{team_l}</span>
                            </div>
                            <div style="background: #e2e8f0; color: #475569; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;">{lineup['formacao']}</div>
                        </div>
                        <div style="padding: 0;">
                            {players_html}
                        </div>
                        <div style="padding: 10px 16px; background: #f1f5f9; font-size: 12px; color: #64748b;">
                            🎩 Técnico: <b>{lineup['tecnico']}</b>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)

            # --- LINHA 3: Notícias ---
            st.markdown("---")
            st.markdown("#### 📰 Últimas Notícias")
            
            # Notícias específicas do confronto
            st.markdown("##### 🔍 Destaques do Confronto & Escalações da Mídia")
            match_news = fetch_match_specific_news(next_match["mandante_nome"], next_match["visitante_nome"], max_results=3)
            
            if match_news:
                for mn in match_news:
                    lineup_html = ""
                    if mn.get("parsed_lineup"):
                        # Converter quebras de linha para HTML e estilizar
                        esc_lineup_text = html.escape(mn["parsed_lineup"]).replace("\n", "<br>")
                        lineup_html = f"""
                        <div style="margin-top: 8px; padding: 12px; background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 6px; font-size: 13px; color: #78350f; font-family: 'Outfit', sans-serif;">
                            <strong>📋 Informação de Escalação Extraída da Mídia:</strong><br>
                            <span style="font-weight: 500;">{esc_lineup_text}</span>
                        </div>
                        """
                    
                    esc_link = html.escape(mn['link'])
                    esc_title = html.escape(mn['title'])
                    esc_source = html.escape(mn['source'])
                    esc_pub_date = html.escape(mn['pub_date'])
                    
                    st.markdown(clean_html(f"""
                    <div style="padding: 14px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                            <span style="font-size: 11px; font-weight: 700; color: #2563eb; text-transform: uppercase;">Notícia do Confronto</span>
                            <span style="font-size: 11px; color: #94a3b8;">{esc_pub_date}</span>
                        </div>
                        <a href="{esc_link}" target="_blank" style="font-size: 14px; font-weight: 700; color: #1e293b; text-decoration: none; font-family: 'Outfit', sans-serif;">
                            {esc_title}
                        </a>
                        <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Fonte: <b>{esc_source}</b></div>
                        {lineup_html}
                    </div>
                    """), unsafe_allow_html=True)
            else:
                st.caption("Nenhuma notícia específica do confronto encontrada no momento.")
            
            st.markdown("---")
            st.markdown("##### 📢 Notícias Gerais por Seleção")
            col_news1, col_news2 = st.columns(2)

            for col_n, team_n in [(col_news1, selected_team), (col_news2, oponente)]:
                with col_n:
                    st.markdown(clean_html(f"""
                    <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 15px; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">
                        {get_flag_html(team_n, width=22)} 
                        <span>{team_n}</span>
                    </div>
                    """), unsafe_allow_html=True)
                    news = fetch_news_rss(team_n, max_results=3)
                    if news:
                        for n in news:
                            esc_n_link = html.escape(n['link'])
                            esc_n_title = html.escape(n['title'])
                            esc_n_source = html.escape(n['source'])
                            esc_n_pub_date = html.escape(n['pub_date'])
                            
                            st.markdown(clean_html(f"""
                            <div style="padding: 10px 14px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 8px;">
                                <a href="{esc_n_link}" target="_blank" style="font-size: 13px; font-weight: 600; color: #1e293b; text-decoration: none;">{esc_n_title}</a>
                                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">{esc_n_source} • {esc_n_pub_date}</div>
                            </div>
                            """), unsafe_allow_html=True)
                    else:
                        st.caption("Nenhuma notícia encontrada no momento.")
