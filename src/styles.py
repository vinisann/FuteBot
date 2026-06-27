"""
Módulo de estilos CSS compartilhado do FuteBot.
Centraliza todo o CSS em um único lugar para evitar duplicação.
Tema: Fundo branco com acentos azul/violeta.
"""

FUTEBOT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* ============ RESET & BASE ============ */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background-color: #ffffff;
        color: #1e293b;
    }



    /* ============ SIDEBAR ============ */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] label {
        color: #1e293b !important;
    }

    h1, h2, h3 {
        font-weight: 800 !important;
        color: #1e293b !important;
        background: none !important;
        -webkit-text-fill-color: #1e293b !important;
        margin-bottom: 15px !important;
    }

    p, span, div, label {
        color: #334155;
    }

    /* ============ GLASSMORPHISM CARDS ============ */
    div.glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.04);
        color: #1e293b;
        transition: box-shadow 0.3s ease, transform 0.2s ease;
    }
    div.glass-card:hover {
        box-shadow: 0 8px 30px 0 rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }

    .neon-border-live {
        border-left: 5px solid #ef4444 !important;
    }
    .neon-border-scheduled {
        border-left: 5px solid #3b82f6 !important;
    }
    .neon-border-finished {
        border-left: 5px solid #10b981 !important;
    }

    /* ============ BOTÕES ============ */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.15) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button, 
    .stButton>button p, 
    .stButton>button span, 
    .stButton>button div {
        color: #ffffff !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.3) !important;
    }
    .stButton>button:hover, 
    .stButton>button:hover p, 
    .stButton>button:hover span, 
    .stButton>button:hover div,
    .stButton>button:active, 
    .stButton>button:active p, 
    .stButton>button:active span, 
    .stButton>button:active div,
    .stButton>button:focus,
    .stButton>button:focus p,
    .stButton>button:focus span,
    .stButton>button:focus div {
        color: #ffffff !important;
    }

    /* ============ SCOREBOARD ============ */
    .scoreboard {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        background: #f8fafc;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 16px 0;
    }
    .team-name {
        font-size: 22px;
        font-weight: 600;
        min-width: 140px;
        text-align: center;
        color: #1e293b;
    }
    .score {
        font-size: 48px;
        font-weight: 800;
        color: #2563eb;
        font-family: 'Outfit', monospace;
    }
    .score-pending {
        font-size: 36px;
        font-weight: 700;
        color: #94a3b8;
        font-family: 'Outfit', monospace;
    }

    /* ============ BADGES ============ */
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-live {
        background-color: #ef4444;
        color: white;
        animation: pulse-live 1.5s ease-in-out infinite;
    }
    .badge-scheduled {
        background-color: #e0e7ff;
        color: #3b82f6;
    }
    .badge-finished {
        background-color: #d1fae5;
        color: #059669;
    }

    @keyframes pulse-live {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ============ SECTION HEADER ============ */
    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }
    .section-header-icon {
        font-size: 20px;
    }
    .section-header-text {
        font-size: 16px;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ============ MÉTRICAS ============ */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }
    [data-testid="stMetricValue"] {
        color: #1e293b !important;
    }

    /* ============ TABS ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }

    /* ============ DATAFRAME ============ */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ============ INFO BAR ============ */
    .info-bar {
        background: linear-gradient(135deg, #eff6ff, #f0f0ff);
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #1e40af;
        font-size: 14px;
        font-weight: 500;
    }

    /* ============ TIMELINE & EVENTOS SIMULADOS ============ */
    .timeline-container {
        position: relative;
        padding-left: 24px;
        margin: 15px 0;
        border-left: 2px solid #e2e8f0;
    }
    .timeline-event {
        position: relative;
        margin-bottom: 12px;
        font-size: 14px;
        color: #334155;
    }
    .timeline-event::before {
        content: '';
        position: absolute;
        left: -31px;
        top: 5px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #3b82f6;
        border: 2px solid #ffffff;
        box-shadow: 0 0 0 2px #dbeafe;
    }
    .timeline-event-goal::before {
        background: #10b981;
        box-shadow: 0 0 0 2px #d1fae5;
    }
    .timeline-event-card::before {
        background: #f59e0b;
        box-shadow: 0 0 0 2px #fef3c7;
    }

    /* ============ BADGES VS E DETALHES ============ */
    .vs-badge {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white !important;
        font-weight: 800;
        font-size: 18px;
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        margin: 0 auto;
    }

    .glow-gold {
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
        border: 2px solid #f59e0b !important;
    }

    /* ============ BORDAS DE RESULTADO HISTÓRICO ============ */
    .border-win-m {
        border-left: 6px solid #10b981 !important;
    }
    .border-win-v {
        border-left: 6px solid #ef4444 !important;
    }
    .border-draw {
        border-left: 6px solid #94a3b8 !important;
    }

    /* ============ METRICAS E COMPARADORES ============ */
    .metric-card-premium {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.01);
        transition: transform 0.2s ease;
    }
    .metric-card-premium:hover {
        transform: translateY(-2px);
        border-color: #cbd5e1;
    }

    /* ============ HERO BANNER ============ */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #7c3aed 100%) !important;
        border: none !important;
        padding: 35px 25px !important;
        border-radius: 20px !important;
        margin-bottom: 30px !important;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.15) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .hero-banner-title {
        margin: 0 0 8px 0 !important;
        color: #ffffff !important;
        font-size: 32px !important;
        font-weight: 800 !important;
        font-family: 'Outfit', sans-serif !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    .hero-banner-subtitle {
        margin: 0 !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        font-family: 'Outfit', sans-serif !important;
        line-height: 1.4 !important;
        -webkit-text-fill-color: rgba(255, 255, 255, 0.9) !important;
    }
</style>

"""


def inject_css():
    """Injeta o CSS global do FuteBot na página Streamlit atual."""
    import streamlit as st
    st.markdown(FUTEBOT_CSS, unsafe_allow_html=True)
