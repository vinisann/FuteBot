import pandas as pd


CONTEXT_MIN = 0.92
CONTEXT_MAX = 1.08
REST_MIN = 0.96
REST_MAX = 1.04

KNOCKOUT_PHASE_KEYWORDS = (
    "oitavas",
    "quartas",
    "semif",
    "final",
    "round of",
    "quarter",
    "semi",
)


def _neutral_context():
    return {
        "modelo_com_contexto": False,
        "fator_mandante": 1.0,
        "fator_visitante": 1.0,
        "fator_descanso_mandante": 1.0,
        "fator_descanso_visitante": 1.0,
        "fator_clima": 1.0,
        "fator_fase": 1.0,
        "contexto_resumo": "Contexto neutro",
    }


def _clip(value, min_value, max_value):
    return float(max(min_value, min(max_value, value)))


def _last_match_date(df_matches, team, target_date, target_match_id=None):
    if df_matches is None or df_matches.empty:
        return None
    df = df_matches.copy()
    if "data_hora" not in df.columns:
        return None

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    target_date = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target_date):
        return None

    df = df[df["data_hora"] < target_date]
    if target_match_id is not None and "id" in df.columns:
        df = df[df["id"] != target_match_id]
    if "status" in df.columns:
        df = df[df["status"].fillna("FINISHED").str.upper() == "FINISHED"]
    if "origem_dados" in df.columns:
        df = df[df["origem_dados"].fillna("").str.lower() != "seed"]

    df = df[(df["mandante_nome"] == team) | (df["visitante_nome"] == team)]
    if df.empty:
        return None
    return df["data_hora"].max()


def _rest_factor(df_matches, team, target_date, target_match_id=None):
    last_date = _last_match_date(df_matches, team, target_date, target_match_id)
    target_date = pd.to_datetime(target_date, errors="coerce")
    if last_date is None or pd.isna(target_date):
        return 1.0

    rest_days = max(0.0, (target_date - last_date).total_seconds() / 86400.0)
    if rest_days < 3:
        return REST_MIN
    if rest_days > 5:
        return REST_MAX
    return 1.0


def _phase_factor(phase):
    phase_text = str(phase or "").lower()
    if any(keyword in phase_text for keyword in KNOCKOUT_PHASE_KEYWORDS):
        return 0.98
    return 1.0


def _weather_factor(weather=None, venue=None):
    factor = 1.0
    if weather:
        temp = weather.get("temperatura_c")
        rain = weather.get("precipitacao_pct")
        wind = weather.get("vento_kmh")
        if temp is not None and float(temp) >= 32:
            factor *= 0.98
        if rain is not None and float(rain) >= 60:
            factor *= 0.99
        if wind is not None and float(wind) >= 25:
            factor *= 0.99

    if venue:
        altitude = venue.get("altitude_m")
        if altitude is not None and float(altitude) >= 1500:
            factor *= 0.99

    return _clip(factor, 0.95, 1.02)


def build_match_context(df_matches, match_row, weather=None, venue=None):
    context = _neutral_context()
    if not match_row:
        return context

    target_date = match_row.get("data_hora")
    mandante = match_row.get("mandante_nome")
    visitante = match_row.get("visitante_nome")
    target_match_id = match_row.get("id")

    rest_m = _rest_factor(df_matches, mandante, target_date, target_match_id) if mandante else 1.0
    rest_v = _rest_factor(df_matches, visitante, target_date, target_match_id) if visitante else 1.0
    phase = _phase_factor(match_row.get("fase"))
    weather_factor = _weather_factor(weather, venue)

    factor_m = _clip(rest_m * phase * weather_factor, CONTEXT_MIN, CONTEXT_MAX)
    factor_v = _clip(rest_v * phase * weather_factor, CONTEXT_MIN, CONTEXT_MAX)
    active = any(
        abs(value - 1.0) > 1e-12
        for value in (rest_m, rest_v, phase, weather_factor)
    )

    summary_parts = []
    if phase < 1.0:
        summary_parts.append("mata-mata reduz levemente o ritmo")
    if rest_m != 1.0 or rest_v != 1.0:
        summary_parts.append("descanso recente ajustado")
    if weather_factor != 1.0:
        summary_parts.append("clima/sede ajusta gols esperados")

    context.update(
        {
            "modelo_com_contexto": active,
            "fator_mandante": factor_m,
            "fator_visitante": factor_v,
            "fator_descanso_mandante": float(rest_m),
            "fator_descanso_visitante": float(rest_v),
            "fator_clima": float(weather_factor),
            "fator_fase": float(phase),
            "contexto_resumo": "; ".join(summary_parts) if summary_parts else "Contexto neutro",
        }
    )
    return context
