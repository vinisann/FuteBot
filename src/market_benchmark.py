OUTCOME_LABELS = {
    "mandante": "mandante",
    "empate": "empate",
    "visitante": "visitante",
}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def implied_probabilities_from_odds(odds):
    """Converte odds decimais em probabilidades implicitas normalizadas."""
    implied_raw = {}
    for outcome in OUTCOME_LABELS:
        odd = _safe_float(odds.get(outcome), 0.0) if odds else 0.0
        implied_raw[outcome] = 1.0 / odd if odd > 1.0 else 0.0

    total = sum(implied_raw.values())
    if total <= 0:
        return {outcome: 1.0 / 3.0 for outcome in OUTCOME_LABELS}
    return {outcome: implied_raw[outcome] / total for outcome in OUTCOME_LABELS}


def average_market_probabilities(houses):
    """Media probabilidades implicitas entre casas/proxies, removendo margem de cada uma."""
    if not houses:
        return {outcome: 1.0 / 3.0 for outcome in OUTCOME_LABELS}

    valid = []
    for odds in houses.values():
        if odds:
            valid.append(implied_probabilities_from_odds(odds))

    if not valid:
        return {outcome: 1.0 / 3.0 for outcome in OUTCOME_LABELS}

    averaged = {
        outcome: sum(row[outcome] for row in valid) / len(valid)
        for outcome in OUTCOME_LABELS
    }
    total = sum(averaged.values())
    return {outcome: averaged[outcome] / total for outcome in OUTCOME_LABELS}


def _normalize_model_probabilities(model_probabilities):
    values = {
        "mandante": _safe_float(
            model_probabilities.get("mandante", model_probabilities.get("prob_vitoria_mandante", 0.0))
        ),
        "empate": _safe_float(
            model_probabilities.get("empate", model_probabilities.get("prob_empate", 0.0))
        ),
        "visitante": _safe_float(
            model_probabilities.get("visitante", model_probabilities.get("prob_vitoria_visitante", 0.0))
        ),
    }
    total = sum(values.values())
    if total <= 0:
        return {outcome: 1.0 / 3.0 for outcome in OUTCOME_LABELS}
    return {outcome: values[outcome] / total for outcome in OUTCOME_LABELS}


def _classification(max_delta_pp):
    if max_delta_pp >= 12.0:
        return "divergencia_alta"
    if max_delta_pp >= 6.0:
        return "monitorar"
    return "alinhado"


def compare_model_to_market(model_probabilities, market_odds):
    """Compara probabilidades do modelo com odds de mercado/proxy sem retroalimentar o modelo."""
    model = _normalize_model_probabilities(model_probabilities or {})
    market = average_market_probabilities(market_odds or {})
    deltas = {
        outcome: (model[outcome] - market[outcome]) * 100.0
        for outcome in OUTCOME_LABELS
    }
    max_outcome = max(deltas, key=lambda outcome: abs(deltas[outcome]))
    max_delta = abs(deltas[max_outcome])
    classification = _classification(max_delta)

    readable = {
        "mandante": "vitoria do mandante",
        "empate": "empate",
        "visitante": "vitoria do visitante",
    }[max_outcome]
    resumo = (
        f"Benchmark de odds: maior diferenca em {readable}, "
        f"{max_delta:.1f} p.p. entre modelo e mercado/proxy."
    )

    return {
        "modelo": model,
        "mercado": market,
        "diferencas_pp": deltas,
        "maior_divergencia_resultado": max_outcome,
        "maior_divergencia_pp": float(round(max_delta, 3)),
        "classificacao": classification,
        "resumo": resumo,
        "usar_no_treino": False,
    }
