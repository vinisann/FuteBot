"""
Cliente de API para o Football-Data.org (v4).
Consome partidas em tempo real da Copa do Mundo 2026 com cache e fallback.
"""

import time
import requests
from datetime import datetime, timezone

# URL Base para a API Football-Data.org (plano gratuito disponível em football-data.org)
API_BASE_URL = "https://api.football-data.org/v4"

# Cache em memória para evitar estourar o limite de 10 chamadas/minuto do Free Tier
_api_cache = {
    "data": None,
    "status": None,
    "timestamp": 0,
    "api_key": None
}
CACHE_TTL_SECONDS = 120  # Cache de 2 minutos (máximo 0.5 chamadas por minuto)

# Mapeamento completo de todos os status possíveis da API
STATUS_MAP = {
    "TIMED": "SCHEDULED",
    "SCHEDULED": "SCHEDULED",
    "IN_PLAY": "LIVE",
    "PAUSED": "LIVE",          # Intervalo (halftime)
    "LIVE": "LIVE",            # Pseudo-status da API
    "EXTRA_TIME": "LIVE",      # Prorrogação (mata-mata)
    "PENALTY_SHOOTOUT": "LIVE",# Pênaltis
    "FINISHED": "FINISHED",
    "SUSPENDED": "SUSPENDED",
    "POSTPONED": "POSTPONED",
    "CANCELLED": "CANCELLED",
    "AWARDED": "FINISHED",
}

# Status que indicam jogo ainda não começou (score deve ser None)
STATUS_NO_SCORE = {"TIMED", "SCHEDULED", "POSTPONED", "CANCELLED"}


def fetch_live_matches_from_api(api_key=None):
    """
    Consome partidas em tempo real/agendadas da API Football-Data.org com cache de 2 minutos.
    Se api_key for None ou a chamada falhar, retorna lista vazia com status informativo.
    """
    global _api_cache
    now = time.time()

    # Se o cache ainda for válido e a chave for a mesma, retornar os dados em cache diretamente
    if _api_cache["data"] is not None and (now - _api_cache["timestamp"]) < CACHE_TTL_SECONDS and _api_cache["api_key"] == api_key:
        return _api_cache["data"], f"{_api_cache['status']} (Cache)"

    if not api_key:
        # Sem API Key → retornar lista vazia com mensagem clara
        _api_cache["data"] = []
        _api_cache["status"] = "⚠️ SEM API KEY — Insira seu token do football-data.org na barra lateral"
        _api_cache["timestamp"] = now
        _api_cache["api_key"] = api_key
        return _api_cache["data"], _api_cache["status"]

    headers = {"X-Auth-Token": api_key}

    try:
        url = f"{API_BASE_URL}/competitions/WC/matches"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            try:
                data = response.json()
                matches = data.get("matches", [])
                formatted_matches = _parse_api_matches(matches)

                if not formatted_matches:
                    _api_cache["data"] = _api_cache.get("data") or []
                    _api_cache["status"] = "✅ CONECTADO — Nenhum jogo da Copa 2026 encontrado na API"
                else:
                    _api_cache["data"] = formatted_matches
                    _api_cache["status"] = f"✅ CONECTADO — {len(formatted_matches)} jogos carregados"
            except ValueError:
                if _api_cache["data"] is not None:
                    _api_cache["status"] = "❌ ERRO DA API — Resposta inválida (JSONDecodeError) (Usando cache)"
                else:
                    _api_cache["data"] = []
                    _api_cache["status"] = "❌ ERRO DA API — Resposta inválida (JSONDecodeError)"

        elif response.status_code == 429:
            # Rate limit excedido - estende o tempo de cache por 3 minutos
            _api_cache["timestamp"] = now + 180
            if _api_cache["data"] is not None:
                _api_cache["status"] = "⏳ RATE LIMIT — Usando dados do cache anterior"
            else:
                _api_cache["data"] = []
                _api_cache["status"] = "⏳ RATE LIMIT — Aguarde 1 minuto e tente novamente"
        elif response.status_code == 403:
            if _api_cache["data"] is None:
                _api_cache["data"] = []
            _api_cache["status"] = "🔑 API KEY INVÁLIDA — Verifique seu token em football-data.org"
        else:
            if _api_cache["data"] is not None:
                _api_cache["status"] = f"❌ ERRO DA API — HTTP {response.status_code} (Usando cache)"
            else:
                _api_cache["data"] = []
                _api_cache["status"] = f"❌ ERRO DA API — HTTP {response.status_code}"

    except requests.exceptions.Timeout:
        _api_cache["data"] = _api_cache.get("data") or []
        _api_cache["status"] = "⏱️ TIMEOUT — A API demorou para responder"
    except requests.exceptions.ConnectionError:
        _api_cache["data"] = _api_cache.get("data") or []
        _api_cache["status"] = "📡 SEM CONEXÃO — Verifique sua internet"
    except Exception as e:
        _api_cache["data"] = _api_cache.get("data") or []
        _api_cache["status"] = f"❌ ERRO — {str(e)[:80]}"

    _api_cache["timestamp"] = now
    _api_cache["api_key"] = api_key
    return _api_cache["data"], _api_cache["status"]


def _parse_api_matches(matches):
    """
    Converte a lista de partidas da API para o formato interno do FuteBot.
    Trata corretamente os scores baseado no status do jogo.
    """
    formatted = []

    for m in matches:
        api_status = m.get("status", "SCHEDULED")
        mapped_status = STATUS_MAP.get(api_status, "SCHEDULED")

        # --- Leitura correta do score baseado no status ---
        score_obj = m.get("score", {}) or {}

        if api_status in STATUS_NO_SCORE:
            # Jogo não começou → sem placar
            gols_m = None
            gols_v = None
        else:
            # Jogo em andamento ou finalizado → ler fullTime (funciona como running score)
            full_time = score_obj.get("fullTime", {}) or {}
            gols_m = full_time.get("home")
            gols_v = full_time.get("away")

            # Se fullTime for None mas o jogo está IN_PLAY, inicializar com 0
            if mapped_status == "LIVE":
                gols_m = gols_m if gols_m is not None else 0
                gols_v = gols_v if gols_v is not None else 0

        # --- Extrair minuto do jogo (quando disponível) ---
        minuto = m.get("minute")  # Campo fornecido pela API para jogos ao vivo

        # --- Extrair informação de data e hora ---
        utc_date = m.get("utcDate", "")
        data_hora = _format_api_date(utc_date)

        # --- Mapear fase e grupo ---
        stage = m.get("stage", "")
        api_group = m.get("group")
        grupo = None
        if api_group:
            grupo = api_group.replace("GROUP_", "").replace("Group ", "").strip()

        if stage == "GROUP_STAGE":
            fase = str(m.get("matchday", "1"))
        else:
            fase = _translate_stage(stage)

        from src.database import TRANSLATE_TEAM_NAME
        home_team_obj = m.get("homeTeam") or {}
        away_team_obj = m.get("awayTeam") or {}
        raw_home = home_team_obj.get("name")
        raw_away = away_team_obj.get("name")
        
        mandante_nome = TRANSLATE_TEAM_NAME.get(raw_home, raw_home) if raw_home else "TBD"
        visitante_nome = TRANSLATE_TEAM_NAME.get(raw_away, raw_away) if raw_away else "TBD"

        winner_api = score_obj.get("winner")
        vencedor_nome = None
        if winner_api == "HOME_TEAM":
            vencedor_nome = mandante_nome
        elif winner_api == "AWAY_TEAM":
            vencedor_nome = visitante_nome

        formatted.append({
            "id": m.get("id"),
            "ano_copa": 2026,
            "data_hora": data_hora,
            "utc_date": utc_date,
            "mandante_nome": mandante_nome,
            "mandante_sigla": home_team_obj.get("tla", "TBD"),
            "visitante_nome": visitante_nome,
            "visitante_sigla": away_team_obj.get("tla", "TBD"),
            "gols_mandante": gols_m,
            "gols_visitante": gols_v,
            "fase": fase,
            "grupo": grupo,
            "status": mapped_status,
            "minuto": minuto,
            "vencedor_nome": vencedor_nome,
        })

    return formatted


def _format_api_date(date_str):
    """Formata datas ISO da API para formato legível (YYYY-MM-DD HH:MM)."""
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        # Converter UTC para horário local
        dt = dt.replace(tzinfo=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        try:
            # Formato alternativo com microsegundos
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            dt = dt.astimezone()
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return date_str


def _translate_stage(stage):
    """Traduz o nome da fase da API para português."""
    translations = {
        "GROUP_STAGE": "Fase de Grupos",
        "ROUND_OF_32": "Fase de 32",
        "LAST_32": "Fase de 32",
        "ROUND_OF_16": "Oitavas de Final",
        "LAST_16": "Oitavas de Final",
        "QUARTER_FINALS": "Quartas de Final",
        "SEMI_FINALS": "Semifinais",
        "THIRD_PLACE": "Disputa de 3º Lugar",
        "FINAL": "Final",
    }
    return translations.get(stage, stage.replace("_", " ").title() if stage else "Fase Eliminatória")


def calculate_match_minute(utc_date_str):
    """
    Calcula o minuto estimado de um jogo ao vivo baseado no horário de início.
    Útil quando a API não fornece o campo 'minute'.
    Desconta 15 minutos do intervalo entre os tempos.
    """
    if not utc_date_str:
        return 45  # fallback
    try:
        dt_start = datetime.strptime(utc_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        elapsed = (now - dt_start).total_seconds() / 60.0
        
        if elapsed <= 45:
            return max(0, int(elapsed))
        elif elapsed <= 60:
            return 45  # Intervalo
        else:
            return max(45, min(int(elapsed - 15), 120))
    except ValueError:
        return 45
