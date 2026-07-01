import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import html

# Adiciona o diretório raiz ao path para importação dos módulos src
sys.path.append(os.path.abspath('.'))

from src.accuracy import build_prediction_history
from src.database import get_connection, init_db, load_historical_matches, load_all_teams, sync_openfootball_finished_matches, evaluate_finished_predictions, load_prediction_evaluations
from src.ML_models import predict_match_probabilities
from src.model_evaluation import evaluate_model_variants, build_calibration_buckets
from src.styles import inject_css
from src.utils import get_flag_html, get_flag, format_fase

def clean_html(html_str):
    """Remove recuos e espaços vazios por linha para evitar falso-positivo de bloco de código no Streamlit Markdown."""
    return "\n".join(line.strip() for line in html_str.split("\n"))

# Configuração da página (deve ser a primeira chamada st)
st.set_page_config(
    page_title="Acurácia das Previsões - FuteBot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()
if "openfootball_sync_2026" not in st.session_state:
    st.session_state["openfootball_sync_2026"] = sync_openfootball_finished_matches(2026)
    evaluate_finished_predictions()

# Injetar CSS compartilhado (tema branco)
inject_css()

# Função para carregar as partidas concluídas joined com as forças ELO atuais das seleções
def load_finished_matches_with_elo():
    conn = get_connection()
    query = """
    SELECT 
        p.id,
        p.ano_copa,
        p.data_hora,
        p.fase,
        p.grupo,
        p.origem_dados,
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
      AND p.gols_mandante IS NOT NULL
      AND p.gols_visitante IS NOT NULL
      AND NOT (p.ano_copa = 2026 AND p.origem_dados = 'seed')
    ORDER BY p.data_hora DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Obter previsões retrospectivas com cache para manter a responsividade da página
@st.cache_data
def get_retroactive_predictions():
    df_finished = load_finished_matches_with_elo()
    df_matches = load_historical_matches(include_seed_2026=False)
    
    predictions = []
    for _, row in df_finished.iterrows():
        m_name = row["mandante_nome"]
        v_name = row["visitante_nome"]
        m_elo = row["mandante_elo"]
        v_elo = row["visitante_elo"]
        g_m = row["gols_mandante"]
        g_v = row["gols_visitante"]
        
        # Calcular previsão usando o modelo do projeto
        prediction_history = build_prediction_history(df_matches, row)
        if prediction_history.empty:
            continue

        pred = predict_match_probabilities(m_name, v_name, m_elo, v_elo, prediction_history)
        p_m = pred["prob_vitoria_mandante"]
        p_e = pred["prob_empate"]
        p_v = pred["prob_vitoria_visitante"]
        placar_prov = pred["placar_mais_provavel"]
        
        # Resultado Real (1X2)
        if g_m > g_v:
            actual_outcome = "M" # Mandante
        elif g_m == g_v:
            actual_outcome = "E" # Empate
        else:
            actual_outcome = "V" # Visitante
            
        # Resultado Previsto (Maior Probabilidade)
        probs = [p_m, p_e, p_v]
        pred_idx = np.argmax(probs)
        if pred_idx == 0:
            pred_outcome = "M"
        elif pred_idx == 1:
            pred_outcome = "E"
        else:
            pred_outcome = "V"
            
        # Placar exato
        actual_score = (int(g_m), int(g_v))
        predicted_score = (int(placar_prov[0]), int(placar_prov[1]))
        
        is_outcome_correct = (pred_outcome == actual_outcome)
        is_score_correct = (predicted_score == actual_score)
        
        # Erro Absoluto de Gols (MAE)
        goal_error = abs(g_m - placar_prov[0]) + abs(g_v - placar_prov[1])
        
        predictions.append({
            "ano_copa": row["ano_copa"],
            "data_hora": row["data_hora"],
            "fase": row["fase"],
            "grupo": row.get("grupo"),
            "origem_dados": row.get("origem_dados"),
            "mandante_nome": m_name,
            "visitante_nome": v_name,
            "gols_mandante": g_m,
            "gols_visitante": g_v,
            "prob_mandante": p_m * 100,
            "prob_empate": p_e * 100,
            "prob_visitante": p_v * 100,
            "prev_gols_m": int(placar_prov[0]),
            "prev_gols_v": int(placar_prov[1]),
            "is_outcome_correct": is_outcome_correct,
            "is_score_correct": is_score_correct,
            "goal_error": goal_error
        })
    return pd.DataFrame(predictions)

@st.cache_data
def get_model_variant_backtest():
    df_matches = load_historical_matches(include_seed_2026=False)
    df_teams = load_all_teams()
    df_predictions = load_prediction_evaluations()
    return evaluate_model_variants(df_matches, df_teams, df_predictions)


# Título / Hero Banner
st.markdown(clean_html("""
<div class="hero-banner" style="background: linear-gradient(135deg, #1e1b4b 0%, #1e3a8a 50%, #581c87 100%);">
    <!-- Decorative glow balls -->
    <div style="position: absolute; top: -50px; right: -50px; width: 180px; height: 180px; background: rgba(255, 255, 255, 0.08); border-radius: 50%; filter: blur(30px);"></div>
    <div style="position: absolute; bottom: -30px; left: 10%; width: 120px; height: 120px; background: rgba(124, 58, 237, 0.15); border-radius: 50%; filter: blur(20px);"></div>
    
    <div style="display: flex; align-items: center; gap: 20px; position: relative; z-index: 1;">
        <div style="font-size: 56px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.3)); line-height: 1; user-select: none;">🎯</div>
        <div>
            <div class="hero-banner-title">
                Acurácia das Previsões
            </div>
            <div class="hero-banner-subtitle">
                🌍 Análise retrospectiva do modelo Poisson-ELO comparando as previsões da nossa IA com os resultados reais das copas.
            </div>
        </div>
    </div>
</div>
"""), unsafe_allow_html=True)

# Buscar todas as previsões da base
df_pred = get_retroactive_predictions()
df_calibrated_eval = load_prediction_evaluations()
df_variant_backtest = get_model_variant_backtest()
if not df_calibrated_eval.empty:
    df_calibrated_eval = df_calibrated_eval.dropna(subset=["evaluated_at"]).copy()

if df_pred.empty:
    st.info("Nenhuma partida finalizada no banco de dados para calcular a acurácia no momento.")
    st.stop()

# Configurações na sidebar
st.sidebar.markdown("### ⚙️ Filtros do Relatório")

# Filtro por Copa do Mundo
copas_disponiveis = ["Todas"] + [str(y) for y in sorted(df_pred["ano_copa"].unique(), reverse=True)]
year_filter = st.sidebar.selectbox("Filtrar por Copa do Mundo:", copas_disponiveis)

# Filtro por Seleção
todas_selecoes = set(df_pred["mandante_nome"].unique()) | set(df_pred["visitante_nome"].unique())
selecoes_list = ["Todas"] + sorted(list(todas_selecoes))
selected_team = st.sidebar.selectbox("Filtrar por Seleção:", selecoes_list)

# Filtragem dos dados
df_filtered = df_pred.copy()
if year_filter != "Todas":
    df_filtered = df_filtered[df_filtered["ano_copa"] == int(year_filter)]
if selected_team != "Todas":
    df_filtered = df_filtered[
        (df_filtered["mandante_nome"] == selected_team) | 
        (df_filtered["visitante_nome"] == selected_team)
    ]

df_calibrated_filtered = df_calibrated_eval.copy()
if not df_calibrated_filtered.empty:
    if year_filter != "Todas":
        df_calibrated_filtered = df_calibrated_filtered[df_calibrated_filtered["ano_copa"] == int(year_filter)]
    if selected_team != "Todas":
        df_calibrated_filtered = df_calibrated_filtered[
            (df_calibrated_filtered["mandante_nome"] == selected_team)
            | (df_calibrated_filtered["visitante_nome"] == selected_team)
        ]

df_variant_filtered = df_variant_backtest.copy()
if not df_variant_filtered.empty:
    if year_filter != "Todas":
        df_variant_filtered = df_variant_filtered[df_variant_filtered["ano_copa"] == int(year_filter)]
    if selected_team != "Todas":
        df_variant_filtered = df_variant_filtered[
            (df_variant_filtered["mandante_nome"] == selected_team)
            | (df_variant_filtered["visitante_nome"] == selected_team)
        ]

# Métricas Calculadas
total_jogos = len(df_filtered)

if total_jogos == 0:
    st.warning("Nenhum jogo finalizado encontrado para os filtros selecionados.")
    st.stop()

vencedor_acertos = df_filtered["is_outcome_correct"].sum()
placar_acertos = df_filtered["is_score_correct"].sum()
mae_gols = df_filtered["goal_error"].mean()

acc_vencedor = (vencedor_acertos / total_jogos) * 100
acc_placar = (placar_acertos / total_jogos) * 100

# Renderização de Métricas Globais com visual premium
st.markdown(clean_html(f"""
<div style="display: flex; gap: 20px; justify-content: space-between; align-items: center; margin-bottom: 28px; flex-wrap: wrap;">
    <!-- Total Jogos -->
    <div class="glass-card" style="flex: 1; min-width: 220px; border-left: 5px solid #64748b; margin-bottom: 0; padding: 20px; text-align: center;">
        <div style="font-size: 13px; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; font-family: 'Outfit', sans-serif;">
            🏃 Jogos Concluídos
        </div>
        <div style="font-size: 36px; font-weight: 800; color: #1e293b; font-family: 'Outfit', sans-serif;">{total_jogos}</div>
        <small style="color: #94a3b8; font-size: 11px; font-family: 'Outfit', sans-serif;">Total de partidas sob análise</small>
    </div>
    
    <!-- Outcome Acc (1X2) -->
    <div class="glass-card" style="flex: 1; min-width: 220px; border-left: 5px solid #10b981; margin-bottom: 0; padding: 20px; text-align: center;">
        <div style="font-size: 13px; color: #059669; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; font-family: 'Outfit', sans-serif;">
            📈 Vencedor Correto (1X2)
        </div>
        <div style="font-size: 36px; font-weight: 800; color: #10b981; font-family: 'Outfit', sans-serif;">{acc_vencedor:.1f}%</div>
        <small style="color: #64748b; font-size: 11px; font-family: 'Outfit', sans-serif; font-weight: 500;">
            {vencedor_acertos} de {total_jogos} resultados previstos
        </small>
    </div>
    
    <!-- Score Acc -->
    <div class="glass-card" style="flex: 1; min-width: 220px; border-left: 5px solid #fbbf24; margin-bottom: 0; padding: 20px; text-align: center;">
        <div style="font-size: 13px; color: #d97706; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; font-family: 'Outfit', sans-serif;">
            🎯 Placar Exato
        </div>
        <div style="font-size: 36px; font-weight: 800; color: #fbbf24; font-family: 'Outfit', sans-serif;">{acc_placar:.1f}%</div>
        <small style="color: #64748b; font-size: 11px; font-family: 'Outfit', sans-serif; font-weight: 500;">
            {placar_acertos} de {total_jogos} placares exatos cravados
        </small>
    </div>
    
    <!-- Goal Error MAE -->
    <div class="glass-card" style="flex: 1; min-width: 220px; border-left: 5px solid #3b82f6; margin-bottom: 0; padding: 20px; text-align: center;">
        <div style="font-size: 13px; color: #2563eb; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; font-family: 'Outfit', sans-serif;">
            ⚽ Erro de Gols (MAE)
        </div>
        <div style="font-size: 36px; font-weight: 800; color: #2563eb; font-family: 'Outfit', sans-serif;">{mae_gols:.2f}</div>
        <small style="color: #64748b; font-size: 11px; font-family: 'Outfit', sans-serif; font-weight: 500;">
            Gols de margem média de desvio por jogo
        </small>
    </div>
</div>
"""), unsafe_allow_html=True)

st.markdown("### Comparativo do Modelo Calibrado")
if df_calibrated_filtered.empty:
    st.info(
        "A calibração incremental ainda não tem previsões pré-jogo avaliadas para estes filtros. "
        "Depois que jogos agendados forem previstos e finalizados, esta seção compara o modelo calibrado com o backtest base."
    )
else:
    cal_total = len(df_calibrated_filtered)
    cal_outcome = int(df_calibrated_filtered["outcome_correct"].fillna(0).sum())
    cal_score = int(df_calibrated_filtered["score_exact"].fillna(0).sum())
    cal_mae = float(df_calibrated_filtered["goal_error"].mean())
    cal_brier = float(df_calibrated_filtered["brier_score"].mean())
    cal_acc = (cal_outcome / cal_total) * 100 if cal_total else 0.0
    cal_score_acc = (cal_score / cal_total) * 100 if cal_total else 0.0

    col_base, col_cal, col_sample = st.columns(3)
    with col_base:
        st.metric("Modelo base - 1X2", f"{acc_vencedor:.1f}%")
        st.caption("Backtest temporal sem usar a própria partida no histórico.")
    with col_cal:
        st.metric("Modelo calibrado - 1X2", f"{cal_acc:.1f}%", delta=f"{cal_acc - acc_vencedor:.1f} p.p.")
        st.caption(f"Placar exato: {cal_score_acc:.1f}% | MAE: {cal_mae:.2f}")
    with col_sample:
        st.metric("Previsões avaliadas", cal_total)
        st.caption(f"Brier Score médio: {cal_brier:.3f}")

st.markdown("### Backtesting avancado: ELO dinamico, forma recente, Dixon-Coles e contexto")
if df_variant_filtered.empty:
    st.info(
        "Ainda nao ha amostra suficiente para comparar as variantes do modelo com os filtros atuais."
    )
else:
    summary = df_variant_filtered.groupby("modelo").agg(
        jogos=("partida_id", "count"),
        acuracia_1x2=("is_outcome_correct", "mean"),
        placar_exato=("is_score_correct", "mean"),
        erro_gols=("goal_error", "mean"),
        brier_score=("brier_score", "mean"),
        log_loss=("log_loss", "mean"),
    ).reset_index()
    summary["Acuracia 1X2"] = (summary["acuracia_1x2"] * 100).round(1)
    summary["Placar exato"] = (summary["placar_exato"] * 100).round(1)
    summary["Erro medio gols"] = summary["erro_gols"].round(2)
    summary["Brier Score"] = summary["brier_score"].round(3)
    summary["Log Loss"] = summary["log_loss"].round(3)

    st.dataframe(
        summary[
            [
                "modelo",
                "jogos",
                "Acuracia 1X2",
                "Placar exato",
                "Erro medio gols",
                "Brier Score",
                "Log Loss",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    if int(summary["jogos"].max()) < 10:
        st.warning(
            "A amostra ainda e pequena. Use estes numeros como sinal inicial, nao como conclusao estatistica definitiva."
        )

    best_row = summary.sort_values(["brier_score", "log_loss"]).iloc[0]
    st.caption(
        f"Melhor calibracao probabilistica no filtro atual: {best_row['modelo']} "
        f"(Brier {best_row['Brier Score']:.3f}, Log Loss {best_row['Log Loss']:.3f}). "
        "O ensemble ponderado combina as variantes por desempenho historico, mantendo o modelo base como ancora quando a amostra ainda e pequena."
    )

    chart_data = df_variant_filtered.copy()
    chart_data["data"] = pd.to_datetime(chart_data["data_hora"], errors="coerce").dt.date
    chart_data = chart_data.dropna(subset=["data"])
    if not chart_data.empty:
        evolution = chart_data.groupby(["data", "modelo"])["is_outcome_correct"].mean().reset_index()
        evolution["Acuracia 1X2"] = evolution["is_outcome_correct"] * 100
        st.line_chart(
            evolution.pivot(index="data", columns="modelo", values="Acuracia 1X2"),
            use_container_width=True,
        )

    preferred = df_variant_filtered[
        df_variant_filtered["modelo"] == "Ensemble ponderado"
    ]
    buckets = build_calibration_buckets(preferred)
    if not buckets.empty:
        st.markdown("#### Curva de calibracao por confianca")
        buckets_display = buckets.copy()
        buckets_display["confianca_media"] = (buckets_display["confianca_media"] * 100).round(1)
        buckets_display["acuracia"] = (buckets_display["acuracia"] * 100).round(1)
        st.dataframe(
            buckets_display.rename(
                columns={
                    "faixa_confianca": "Faixa de confianca",
                    "previsoes": "Previsoes",
                    "confianca_media": "Confianca media (%)",
                    "acuracia": "Acuracia real (%)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

# Tabela detalhada
st.markdown("### 📋 Log Completo de Previsões vs Realidade")
st.write("Abaixo está a lista detalhada de jogos concluídos e a análise individual de acertos da inteligência artificial:")

# Conversão das previsões para exibição em tabela HTML
table_rows = []
for idx, m in df_filtered.iterrows():
    # Identificar cor do background da linha baseado no resultado da previsão
    if m["is_score_correct"]:
        bg_style = 'background-color: #fffdf0; border-left: 4px solid #fbbf24;' # Destaque dourado para placar exato
        badge = '<span style="background-color: #fef3c7; color: #d97706; padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-transform: uppercase; font-family: \'Outfit\', sans-serif;">🎯 Placar Exato</span>'
    elif m["is_outcome_correct"]:
        bg_style = 'background-color: #f8fdf9; border-left: 4px solid #10b981;' # Destaque verde claro para vencedor correto
        badge = '<span style="background-color: #d1fae5; color: #059669; padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-transform: uppercase; font-family: \'Outfit\', sans-serif;">✅ Vencedor</span>'
    else:
        bg_style = 'border-left: 4px solid #e2e8f0;'
        badge = '<span style="background-color: #f1f5f9; color: #64748b; padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 11px; display: inline-block; text-transform: uppercase; font-family: \'Outfit\', sans-serif;">❌ Erro</span>'

    # Formatar as strings de probabilidade
    prob_text = f"""
    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">
        <span style="color:#2563eb; font-weight:600;">{m['prob_mandante']:.0f}%</span> / 
        <span>{m['prob_empate']:.0f}%</span> / 
        <span style="color:#7c3aed; font-weight:600;">{m['prob_visitante']:.0f}%</span>
    </div>
    """

    table_rows.append(f"""
    <tr style="{bg_style} border-bottom: 1px solid #e2e8f0; font-size: 13px;">
        <td style="padding: 12px 10px; color: #64748b; font-weight: 500;">
            🏆 Copa {m['ano_copa']}<br>
            <span style="font-size: 11px; color: #94a3b8;">{format_fase(m['fase'], m.get('grupo'))}</span>
        </td>
        <td style="padding: 12px 10px; color: #1e293b; font-weight: 600;">
            <div style="display: flex; align-items: center; gap: 8px;">
                {get_flag_html(m['mandante_nome'], width=20)} 
                <span>{m['mandante_nome']}</span>
            </div>
        </td>
        <td style="padding: 12px 10px; text-align: center; font-size: 15px; color: #1e293b; font-weight: 800; white-space: nowrap;">
            {m['gols_mandante']} - {m['gols_visitante']}
        </td>
        <td style="padding: 12px 10px; color: #1e293b; font-weight: 600;">
            <div style="display: flex; align-items: center; gap: 8px; flex-direction: row;">
                {get_flag_html(m['visitante_nome'], width=20)} 
                <span>{m['visitante_nome']}</span>
            </div>
        </td>
        <td style="padding: 12px 10px;">
            <div style="font-weight: 700; color: #475569; font-size: 13px;">{m['prev_gols_m']} x {m['prev_gols_v']}</div>
            {prob_text}
        </td>
        <td style="padding: 12px 10px; text-align: right;">{badge}</td>
    </tr>
    """)

html_table = f"""
<div style="overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 12px; margin-top: 15px; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.02); background: white;">
    <table style="width: 100%; border-collapse: collapse; text-align: left; font-family: 'Outfit', sans-serif;">
        <thead>
            <tr style="background-color: #f8fafc; border-bottom: 2px solid #e2e8f0; font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">
                <th style="padding: 12px 10px;">Copa / Fase</th>
                <th style="padding: 12px 10px;">Mandante</th>
                <th style="padding: 12px 10px; text-align: center;">Placar Real</th>
                <th style="padding: 12px 10px;">Visitante</th>
                <th style="padding: 12px 10px;">Previsão IA (Placar/Odds)</th>
                <th style="padding: 12px 10px; text-align: right;">Status</th>
            </tr>
        </thead>
        <tbody>
            {"".join(table_rows)}
        </tbody>
    </table>
</div>
"""
st.markdown(clean_html(html_table), unsafe_allow_html=True)
