from datetime import datetime

import pandas as pd


MIN_EVALUATED_PREDICTIONS = 4
ATTACK_CAP = (0.90, 1.10)
DEFENSE_CAP = (0.90, 1.10)
MAX_UPSET_FACTOR = 0.04


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def _parse_date(value):
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def _empty_calibration():
    return {
        "active": False,
        "sample_size": 0,
        "team_adjustments": {},
        "upset_factor": 0.0,
    }


def build_model_calibration(df_matches, df_predictions):
    """
    Build conservative calibration multipliers from evaluated pre-match predictions.

    The historical matches frame is accepted for future extension and to keep the
    call site explicit, but v1 learns only from stored prediction errors.
    """
    if df_predictions is None or df_predictions.empty:
        return _empty_calibration()

    required = {
        "mandante_nome",
        "visitante_nome",
        "xg_mandante",
        "xg_visitante",
        "gols_mandante_real",
        "gols_visitante_real",
        "prob_mandante",
        "prob_empate",
        "prob_visitante",
        "evaluated_at",
    }
    if not required.issubset(df_predictions.columns):
        return _empty_calibration()

    df = df_predictions.dropna(
        subset=[
            "mandante_nome",
            "visitante_nome",
            "xg_mandante",
            "xg_visitante",
            "gols_mandante_real",
            "gols_visitante_real",
            "evaluated_at",
        ]
    ).copy()
    if len(df) < MIN_EVALUATED_PREDICTIONS:
        calibration = _empty_calibration()
        calibration["sample_size"] = len(df)
        return calibration

    parsed_dates = df["evaluated_at"].apply(_parse_date)
    latest = max([date for date in parsed_dates if date is not None], default=None)

    team_errors = {}
    upset_scores = []

    for idx, row in df.iterrows():
        evaluated_at = _parse_date(row.get("evaluated_at"))
        if latest and evaluated_at:
            days_old = max((latest - evaluated_at).days, 0)
            weight = 0.75 ** min(days_old, 8)
        else:
            weight = 1.0

        mandante = row["mandante_nome"]
        visitante = row["visitante_nome"]
        xg_m = float(row["xg_mandante"])
        xg_v = float(row["xg_visitante"])
        gols_m = float(row["gols_mandante_real"])
        gols_v = float(row["gols_visitante_real"])

        for team in (mandante, visitante):
            team_errors.setdefault(team, {"attack": [], "defense": []})

        team_errors[mandante]["attack"].append((gols_m - xg_m, weight))
        team_errors[mandante]["defense"].append((gols_v - xg_v, weight))
        team_errors[visitante]["attack"].append((gols_v - xg_v, weight))
        team_errors[visitante]["defense"].append((gols_m - xg_m, weight))

        probs = [
            float(row.get("prob_mandante", 0.0)),
            float(row.get("prob_empate", 0.0)),
            float(row.get("prob_visitante", 0.0)),
        ]
        actual_idx = 0 if gols_m > gols_v else 1 if gols_m == gols_v else 2
        favorite_idx = int(max(range(3), key=lambda pos: probs[pos]))
        favorite_prob = probs[favorite_idx]
        if favorite_idx != actual_idx and favorite_prob >= 0.60:
            upset_scores.append((favorite_prob - probs[actual_idx], weight))

    adjustments = {}
    for team, errors in team_errors.items():
        attack = _weighted_mean(errors["attack"])
        defense = _weighted_mean(errors["defense"])
        adjustments[team] = {
            "attack": _clip(1.0 + attack * 0.08, *ATTACK_CAP),
            "defense": _clip(1.0 + defense * 0.08, *DEFENSE_CAP),
        }

    raw_upset = _weighted_mean(upset_scores) if upset_scores else 0.0
    upset_factor = _clip(raw_upset * 0.10, 0.0, MAX_UPSET_FACTOR)

    return {
        "active": True,
        "sample_size": len(df),
        "team_adjustments": adjustments,
        "upset_factor": upset_factor,
    }


def _weighted_mean(items):
    if not items:
        return 0.0
    total_weight = sum(weight for _, weight in items)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in items) / total_weight


def apply_calibration_to_lambdas(lambda_m, lambda_v, mandante, visitante, calibration):
    if not calibration or not calibration.get("active"):
        return lambda_m, lambda_v, {
            "modelo_calibrado": False,
            "ajuste_mandante": 1.0,
            "ajuste_visitante": 1.0,
            "fator_zebra": 0.0,
        }

    adjustments = calibration.get("team_adjustments", {})
    mandante_adj = adjustments.get(mandante, {"attack": 1.0, "defense": 1.0})
    visitante_adj = adjustments.get(visitante, {"attack": 1.0, "defense": 1.0})

    ajuste_m = _clip(
        float(mandante_adj.get("attack", 1.0)) * float(visitante_adj.get("defense", 1.0)),
        0.90,
        1.10,
    )
    ajuste_v = _clip(
        float(visitante_adj.get("attack", 1.0)) * float(mandante_adj.get("defense", 1.0)),
        0.90,
        1.10,
    )

    return lambda_m * ajuste_m, lambda_v * ajuste_v, {
        "modelo_calibrado": True,
        "ajuste_mandante": ajuste_m,
        "ajuste_visitante": ajuste_v,
        "fator_zebra": float(calibration.get("upset_factor", 0.0)),
    }
