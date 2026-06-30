import numpy as np
import pandas as pd


DEFAULT_RHO = -0.08
RHO_MIN = -0.15
RHO_MAX = 0.05
MIN_RHO_SAMPLE = 3


def _metadata(active, rho, low_score_adjustment):
    return {
        "modelo_dixon_coles": bool(active),
        "rho_dixon_coles": float(rho),
        "ajuste_placares_baixos": bool(low_score_adjustment),
    }


def _tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    if home_goals == 0 and away_goals == 0:
        return 1.0 - (lambda_home * lambda_away * rho)
    if home_goals == 0 and away_goals == 1:
        return 1.0 + (lambda_home * rho)
    if home_goals == 1 and away_goals == 0:
        return 1.0 + (lambda_away * rho)
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def apply_dixon_coles_correction(score_matrix, lambda_home, lambda_away, rho=DEFAULT_RHO):
    matrix = np.array(score_matrix, dtype=float)
    total = matrix.sum()
    if total <= 0:
        return matrix, _metadata(False, 0.0, False)

    normalized = matrix / total
    try:
        rho = float(rho)
    except (TypeError, ValueError):
        rho = DEFAULT_RHO
    rho = float(np.clip(rho, RHO_MIN, RHO_MAX))

    if abs(rho) < 1e-12:
        return normalized, _metadata(False, 0.0, False)

    corrected = normalized.copy()
    max_home, max_away = corrected.shape
    for home_goals in range(min(2, max_home)):
        for away_goals in range(min(2, max_away)):
            corrected[home_goals, away_goals] *= max(
                0.01,
                _tau(home_goals, away_goals, float(lambda_home), float(lambda_away), rho),
            )

    corrected_total = corrected.sum()
    if corrected_total <= 0:
        return normalized, _metadata(False, 0.0, False)

    return corrected / corrected_total, _metadata(True, rho, True)


def _valid_matches(df_matches):
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()

    df = df_matches.copy()
    df = df[df["gols_mandante"].notna() & df["gols_visitante"].notna()]
    if "status" in df.columns:
        df = df[df["status"].fillna("FINISHED").str.upper() == "FINISHED"]
    if "origem_dados" in df.columns:
        df = df[df["origem_dados"].fillna("").str.lower() != "seed"]
    return df


def estimate_dixon_coles_rho(df_matches):
    df = _valid_matches(df_matches)
    if df.empty:
        return 0.0
    if len(df) < MIN_RHO_SAMPLE:
        return DEFAULT_RHO

    low_score = df[
        (df["gols_mandante"].astype(float) <= 1.0)
        & (df["gols_visitante"].astype(float) <= 1.0)
    ]
    observed_rate = len(low_score) / len(df)

    # Copas tendem a ter jogos mais travados em mata-mata e aberturas.
    # Um excesso de placares baixos puxa rho para baixo de forma conservadora.
    excess_low_score = observed_rate - 0.34
    rho = DEFAULT_RHO - (excess_low_score * 0.08)
    return float(np.clip(rho, RHO_MIN, RHO_MAX))
