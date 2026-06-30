import math

import pandas as pd


DEFAULT_ELO = 1500.0
DEFAULT_K_FACTOR = 28.0
FORM_WINDOW = 5
FORM_DECAY = 0.72
FORM_CAP_MIN = 0.90
FORM_CAP_MAX = 1.10


def _is_real_finished_match(row):
    status = str(row.get("status", "FINISHED")).upper()
    if status != "FINISHED":
        return False
    if pd.isna(row.get("gols_mandante")) or pd.isna(row.get("gols_visitante")):
        return False
    if str(row.get("origem_dados", "")).lower() == "seed":
        return False
    return True


def _clean_matches(df_matches):
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()

    df = df_matches.copy()
    df = df[df.apply(_is_real_finished_match, axis=1)]
    if df.empty:
        return df

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df = df.dropna(subset=["data_hora"])
    return df.sort_values(["data_hora", "id"]).reset_index(drop=True)


def _margin_multiplier(goal_diff):
    return max(1.0, math.log(abs(goal_diff) + 1.0))


def _expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def build_dynamic_elo_history(df_matches, initial_ratings=None):
    """Build pre/post-match ELO snapshots without looking ahead."""
    df = _clean_matches(df_matches)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "partida_id",
                "data_hora",
                "mandante_nome",
                "visitante_nome",
                "pre_elo_mandante",
                "pre_elo_visitante",
                "post_elo_mandante",
                "post_elo_visitante",
            ]
        )

    ratings = {str(team): float(value) for team, value in (initial_ratings or {}).items()}
    snapshots = []

    for _, row in df.iterrows():
        mandante = row["mandante_nome"]
        visitante = row["visitante_nome"]
        rating_m = float(ratings.get(mandante, DEFAULT_ELO))
        rating_v = float(ratings.get(visitante, DEFAULT_ELO))

        gols_m = int(row["gols_mandante"])
        gols_v = int(row["gols_visitante"])
        if gols_m > gols_v:
            actual_m = 1.0
        elif gols_m == gols_v:
            actual_m = 0.5
        else:
            actual_m = 0.0

        expected_m = _expected_score(rating_m, rating_v)
        multiplier = _margin_multiplier(gols_m - gols_v)
        delta = DEFAULT_K_FACTOR * multiplier * (actual_m - expected_m)

        post_m = rating_m + delta
        post_v = rating_v - delta
        ratings[mandante] = post_m
        ratings[visitante] = post_v

        snapshots.append(
            {
                "partida_id": row.get("id"),
                "data_hora": row["data_hora"],
                "mandante_nome": mandante,
                "visitante_nome": visitante,
                "pre_elo_mandante": float(rating_m),
                "pre_elo_visitante": float(rating_v),
                "post_elo_mandante": float(post_m),
                "post_elo_visitante": float(post_v),
            }
        )

    return pd.DataFrame(snapshots)


def get_pre_match_elo(elo_history, partida_id, mandante, visitante):
    if elo_history is None or elo_history.empty:
        return None, None

    rows = elo_history[elo_history["partida_id"] == partida_id]
    if rows.empty:
        return None, None

    row = rows.iloc[0]
    if row["mandante_nome"] == mandante and row["visitante_nome"] == visitante:
        return float(row["pre_elo_mandante"]), float(row["pre_elo_visitante"])
    if row["mandante_nome"] == visitante and row["visitante_nome"] == mandante:
        return float(row["pre_elo_visitante"]), float(row["pre_elo_mandante"])
    return None, None


def _team_recent_matches(df, team, target_date, target_match_id=None, window=FORM_WINDOW):
    target_date = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target_date):
        return pd.DataFrame()

    valid = _clean_matches(df)
    if valid.empty:
        return valid

    valid = valid[valid["data_hora"] < target_date]
    if target_match_id is not None and "id" in valid.columns:
        valid = valid[valid["id"] != target_match_id]

    valid = valid[
        (valid["mandante_nome"] == team)
        | (valid["visitante_nome"] == team)
    ].sort_values("data_hora", ascending=False)
    return valid.head(window).copy()


def _team_form_factor(df, team, target_date, target_match_id=None):
    recent = _team_recent_matches(df, team, target_date, target_match_id)
    if recent.empty:
        return 1.0, 0

    weighted_score = 0.0
    weight_total = 0.0
    for idx, (_, row) in enumerate(recent.iterrows()):
        weight = FORM_DECAY ** idx
        is_home = row["mandante_nome"] == team
        goals_for = float(row["gols_mandante"] if is_home else row["gols_visitante"])
        goals_against = float(row["gols_visitante"] if is_home else row["gols_mandante"])

        if goals_for > goals_against:
            points_component = 1.0
        elif goals_for == goals_against:
            points_component = 0.0
        else:
            points_component = -1.0

        goal_component = max(-1.5, min(1.5, (goals_for - goals_against) / 2.0))
        weighted_score += weight * ((0.65 * points_component) + (0.35 * goal_component))
        weight_total += weight

    normalized = weighted_score / weight_total if weight_total else 0.0
    factor = 1.0 + (normalized * 0.08)
    return float(max(FORM_CAP_MIN, min(FORM_CAP_MAX, factor))), int(len(recent))


def calculate_recent_form(df_matches, mandante, visitante, target_date, target_match_id=None):
    """Return conservative form multipliers for teams before a target match."""
    mandante_factor, mandante_games = _team_form_factor(
        df_matches, mandante, target_date, target_match_id
    )
    visitante_factor, visitante_games = _team_form_factor(
        df_matches, visitante, target_date, target_match_id
    )
    return {
        "mandante_factor": mandante_factor,
        "visitante_factor": visitante_factor,
        "mandante_games": mandante_games,
        "visitante_games": visitante_games,
    }
