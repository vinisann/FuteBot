import re


INJURY_TERMS = (
    "lesao",
    "lesionado",
    "machucado",
    "desfalque",
    "suspensao",
    "suspenso",
    "fora",
)
PRESSURE_TERMS = (
    "pressao",
    "crise",
    "obrigacao",
    "cobrado",
    "criticas",
    "eliminacao",
)
STABILITY_TERMS = (
    "time definido",
    "titulares",
    "forca maxima",
    "deve repetir",
    "retorno",
    "recuperado",
)
LINEUP_TERMS = (
    "provavel escalacao",
    "escalacao",
    "deve ir a campo",
    "deve comecar",
)


def _normalize_text(value):
    return str(value or "").lower()


def _count_terms(text, terms):
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", text)) for term in terms)


def _as_text(news_items=None, lineup_text=None):
    items = []
    if isinstance(news_items, str):
        items.append(news_items)
    elif news_items:
        items.extend(str(item) for item in news_items if item)
    if lineup_text:
        items.append(str(lineup_text))
    return " ".join(items)


def _clip_adjustment(value):
    return round(max(0.95, min(1.05, float(value))), 4)


def extract_team_external_signals(team_name, news_items=None, lineup_text=None):
    text = _normalize_text(_as_text(news_items, lineup_text))
    injury_hits = _count_terms(text, INJURY_TERMS)
    pressure_hits = _count_terms(text, PRESSURE_TERMS)
    stability_hits = _count_terms(text, STABILITY_TERMS)
    lineup_hits = _count_terms(text, LINEUP_TERMS)

    if lineup_text and ";" in str(lineup_text):
        lineup_hits += 1

    injury_risk = min(1.0, injury_hits / 3.0)
    news_pressure = min(1.0, pressure_hits / 3.0)
    stability_score = min(1.0, stability_hits / 3.0)
    lineup_confidence = min(1.0, lineup_hits / 2.0)

    adjustment = 1.0
    adjustment -= injury_risk * 0.035
    adjustment -= news_pressure * 0.010
    adjustment += stability_score * 0.020
    adjustment += lineup_confidence * 0.025
    adjustment = _clip_adjustment(adjustment)

    reasons = []
    if injury_risk > 0:
        reasons.append(f"{team_name}: possivel desfalque ou lesao detectado em noticia recente.")
    if lineup_confidence > 0:
        reasons.append(f"{team_name}: escalacao provavel encontrada com boa confianca.")
    if stability_score > 0:
        reasons.append(f"{team_name}: sinais de estabilidade no time titular.")
    if news_pressure > 0:
        reasons.append(f"{team_name}: noticias indicam pressao competitiva adicional.")
    if not reasons:
        reasons.append(f"{team_name}: sem evidencias externas suficientes; ajuste neutro.")

    return {
        "team": team_name,
        "lineup_confidence": float(lineup_confidence),
        "injury_risk": float(injury_risk),
        "news_pressure": float(news_pressure),
        "stability_score": float(stability_score),
        "external_adjustment": float(adjustment),
        "reasons": reasons,
    }


def _items_for_team(team_name, news_items):
    if isinstance(news_items, dict):
        return news_items.get(team_name, [])
    return news_items or []


def _lineup_for_team(team_name, lineups):
    if isinstance(lineups, dict):
        return lineups.get(team_name)
    return None


def build_match_external_signals(mandante, visitante, news_items=None, lineups=None):
    mandante_signals = extract_team_external_signals(
        mandante,
        news_items=_items_for_team(mandante, news_items),
        lineup_text=_lineup_for_team(mandante, lineups),
    )
    visitante_signals = extract_team_external_signals(
        visitante,
        news_items=_items_for_team(visitante, news_items),
        lineup_text=_lineup_for_team(visitante, lineups),
    )
    reasons = mandante_signals["reasons"] + visitante_signals["reasons"]
    used = (
        mandante_signals["external_adjustment"] != 1.0
        or visitante_signals["external_adjustment"] != 1.0
    )
    return {
        "mandante": mandante_signals,
        "visitante": visitante_signals,
        "sinais_externos_usados": bool(used),
        "motivos_sinais_externos": reasons,
    }


def apply_external_signal_adjustment(lambda_m, lambda_v, mandante, visitante, external_signals=None):
    if not external_signals:
        return float(lambda_m), float(lambda_v), {
            "sinais_externos_usados": False,
            "ajuste_externo_mandante": 1.0,
            "ajuste_externo_visitante": 1.0,
            "motivos_sinais_externos": [],
        }

    mandante_data = external_signals.get("mandante", {})
    visitante_data = external_signals.get("visitante", {})
    ajuste_m = _clip_adjustment(mandante_data.get("external_adjustment", 1.0))
    ajuste_v = _clip_adjustment(visitante_data.get("external_adjustment", 1.0))
    motivos = list(external_signals.get("motivos_sinais_externos", []))
    if not motivos:
        motivos = list(mandante_data.get("reasons", [])) + list(visitante_data.get("reasons", []))

    return max(float(lambda_m) * ajuste_m, 0.1), max(float(lambda_v) * ajuste_v, 0.1), {
        "sinais_externos_usados": bool(external_signals.get("sinais_externos_usados", False)),
        "ajuste_externo_mandante": float(ajuste_m),
        "ajuste_externo_visitante": float(ajuste_v),
        "motivos_sinais_externos": motivos,
    }
