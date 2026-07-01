import math

import numpy as np
import pandas as pd

from src.accuracy import build_prediction_history
from src.dynamic_elo import (
    build_dynamic_elo_history,
    calculate_recent_form,
    get_pre_match_elo,
)
from src.ML_models import predict_match_probabilities
from src.dixon_coles import estimate_dixon_coles_rho
from src.match_context import build_match_context
from src.model_calibration import build_model_calibration
from src.model_ensemble import calculate_ensemble_weights, combine_predictions


OUTCOME_LABELS = ("M", "E", "V")


def _valid_finished_matches(df_matches):
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()

    df = df_matches.copy()
    df = df[df["gols_mandante"].notna() & df["gols_visitante"].notna()]
    if "status" in df.columns:
        df = df[df["status"].fillna("FINISHED").str.upper() == "FINISHED"]
    if "origem_dados" in df.columns:
        df = df[df["origem_dados"].fillna("").str.lower() != "seed"]
    if "data_hora" in df.columns:
        df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
        df = df.dropna(subset=["data_hora"]).sort_values(["data_hora", "id"])
    return df.reset_index(drop=True)


def _team_elo_map(df_teams):
    if df_teams is None or df_teams.empty:
        return {}
    if "nome" not in df_teams.columns or "elo_rating" not in df_teams.columns:
        return {}
    return {
        row["nome"]: float(row["elo_rating"])
        for _, row in df_teams.dropna(subset=["nome"]).iterrows()
        if pd.notna(row.get("elo_rating"))
    }


def _actual_outcome(gols_m, gols_v):
    if gols_m > gols_v:
        return "M"
    if gols_m == gols_v:
        return "E"
    return "V"


def _prediction_probabilities(prediction):
    return {
        "M": float(prediction["prob_vitoria_mandante"]),
        "E": float(prediction["prob_empate"]),
        "V": float(prediction["prob_vitoria_visitante"]),
    }


def calculate_brier_score(probabilities, actual_outcome):
    return float(
        sum(
            (float(probabilities.get(label, 0.0)) - (1.0 if label == actual_outcome else 0.0)) ** 2
            for label in OUTCOME_LABELS
        )
    )


def calculate_log_loss(probabilities, actual_outcome):
    probability = float(probabilities.get(actual_outcome, 0.0))
    probability = max(1e-12, min(1.0, probability))
    return float(-math.log(probability))


def _variant_result(model_name, row, prediction):
    probabilities = _prediction_probabilities(prediction)
    actual = _actual_outcome(row["gols_mandante"], row["gols_visitante"])
    predicted = max(probabilities, key=probabilities.get)
    predicted_score = prediction["placar_mais_provavel"]

    return {
        "modelo": model_name,
        "partida_id": row.get("id"),
        "ano_copa": row.get("ano_copa"),
        "data_hora": row.get("data_hora"),
        "fase": row.get("fase"),
        "grupo": row.get("grupo"),
        "mandante_nome": row["mandante_nome"],
        "visitante_nome": row["visitante_nome"],
        "gols_mandante": int(row["gols_mandante"]),
        "gols_visitante": int(row["gols_visitante"]),
        "prev_gols_m": int(predicted_score[0]),
        "prev_gols_v": int(predicted_score[1]),
        "actual_outcome": actual,
        "predicted_outcome": predicted,
        "confidence": float(probabilities[predicted]),
        "prob_mandante": probabilities["M"],
        "prob_empate": probabilities["E"],
        "prob_visitante": probabilities["V"],
        "is_outcome_correct": predicted == actual,
        "is_score_correct": (int(predicted_score[0]), int(predicted_score[1]))
        == (int(row["gols_mandante"]), int(row["gols_visitante"])),
        "goal_error": abs(int(row["gols_mandante"]) - int(predicted_score[0]))
        + abs(int(row["gols_visitante"]) - int(predicted_score[1])),
        "brier_score": calculate_brier_score(probabilities, actual),
        "log_loss": calculate_log_loss(probabilities, actual),
        "elo_mandante_usado": prediction.get("elo_mandante_usado"),
        "elo_visitante_usado": prediction.get("elo_visitante_usado"),
        "fator_forma_mandante": prediction.get("fator_forma_mandante", 1.0),
        "fator_forma_visitante": prediction.get("fator_forma_visitante", 1.0),
        "modelo_calibrado": prediction.get("modelo_calibrado", False),
        "modelo_com_forma": prediction.get("modelo_com_forma", False),
        "modelo_dixon_coles": prediction.get("modelo_dixon_coles", False),
        "rho_dixon_coles": prediction.get("rho_dixon_coles", 0.0),
        "ajuste_placares_baixos": prediction.get("ajuste_placares_baixos", False),
        "modelo_com_contexto": prediction.get("modelo_com_contexto", False),
        "modelo_ensemble": prediction.get("modelo_ensemble", False),
        "pesos_ensemble": prediction.get("pesos_ensemble"),
        "fator_contexto_mandante": prediction.get("fator_contexto_mandante", 1.0),
        "fator_contexto_visitante": prediction.get("fator_contexto_visitante", 1.0),
        "fator_descanso_mandante": prediction.get("fator_descanso_mandante", 1.0),
        "fator_descanso_visitante": prediction.get("fator_descanso_visitante", 1.0),
        "fator_clima": prediction.get("fator_clima", 1.0),
        "fator_fase": prediction.get("fator_fase", 1.0),
        "contexto_resumo": prediction.get("contexto_resumo", "Contexto neutro"),
    }


def evaluate_model_variants(df_matches, df_teams, df_predictions=None):
    matches = _valid_finished_matches(df_matches)
    if matches.empty:
        return pd.DataFrame()

    team_elos = _team_elo_map(df_teams)
    dynamic_history = build_dynamic_elo_history(matches)
    calibration = build_model_calibration(matches, df_predictions if df_predictions is not None else pd.DataFrame())

    rows = []
    for _, row in matches.iterrows():
        history = build_prediction_history(matches, row)
        if history.empty:
            continue

        m_name = row["mandante_nome"]
        v_name = row["visitante_nome"]
        base_m_elo = team_elos.get(m_name, 1850.0)
        base_v_elo = team_elos.get(v_name, 1850.0)
        dynamic_elo = get_pre_match_elo(dynamic_history, row.get("id"), m_name, v_name)
        dynamic_m_elo = dynamic_elo[0] if dynamic_elo[0] is not None else 1500.0
        dynamic_v_elo = dynamic_elo[1] if dynamic_elo[1] is not None else 1500.0
        recent_form = calculate_recent_form(
            matches,
            m_name,
            v_name,
            row["data_hora"],
            target_match_id=row.get("id"),
        )
        dixon_coles_rho = estimate_dixon_coles_rho(history)
        match_context = build_match_context(history, row.to_dict())

        variants = [
            (
                "Base Poisson-ELO",
                predict_match_probabilities(m_name, v_name, base_m_elo, base_v_elo, history),
            ),
            (
                "ELO dinâmico",
                predict_match_probabilities(
                    m_name,
                    v_name,
                    dynamic_m_elo,
                    dynamic_v_elo,
                    history,
                    dynamic_elo=(dynamic_m_elo, dynamic_v_elo),
                ),
            ),
            (
                "ELO dinâmico + forma/calibração",
                predict_match_probabilities(
                    m_name,
                    v_name,
                    dynamic_m_elo,
                    dynamic_v_elo,
                    history,
                    calibration=calibration,
                    dynamic_elo=(dynamic_m_elo, dynamic_v_elo),
                    recent_form=recent_form,
                ),
            ),
            (
                "ELO dinâmico + forma/calibração + Dixon-Coles",
                predict_match_probabilities(
                    m_name,
                    v_name,
                    dynamic_m_elo,
                    dynamic_v_elo,
                    history,
                    calibration=calibration,
                    dynamic_elo=(dynamic_m_elo, dynamic_v_elo),
                    recent_form=recent_form,
                    score_correction={"method": "dixon_coles", "rho": dixon_coles_rho},
                ),
            ),
            (
                "ELO dinâmico + forma/calibração + Dixon-Coles + contexto",
                predict_match_probabilities(
                    m_name,
                    v_name,
                    dynamic_m_elo,
                    dynamic_v_elo,
                    history,
                    calibration=calibration,
                    dynamic_elo=(dynamic_m_elo, dynamic_v_elo),
                    recent_form=recent_form,
                    score_correction={"method": "dixon_coles", "rho": dixon_coles_rho},
                    match_context=match_context,
                ),
            ),
        ]

        predictions_by_model = {model_name: prediction for model_name, prediction in variants}
        prior_results = pd.DataFrame(rows)
        weights = calculate_ensemble_weights(prior_results)
        if not weights:
            weights = {model_name: 1.0 / len(predictions_by_model) for model_name in predictions_by_model}
        variants.append(
            (
                "Ensemble ponderado",
                combine_predictions(predictions_by_model, weights),
            )
        )

        for model_name, prediction in variants:
            rows.append(_variant_result(model_name, row, prediction))

    return pd.DataFrame(rows)


def build_calibration_buckets(df_results):
    if df_results is None or df_results.empty:
        return pd.DataFrame(
            columns=["faixa_confianca", "previsoes", "confianca_media", "acuracia"]
        )

    df = df_results.dropna(subset=["confidence", "is_outcome_correct"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=["faixa_confianca", "previsoes", "confianca_media", "acuracia"]
        )

    bins = np.linspace(0.0, 1.0, 6)
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    df["faixa_confianca"] = pd.cut(
        df["confidence"].clip(0, 1),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    grouped = df.groupby("faixa_confianca", observed=False).agg(
        previsoes=("confidence", "count"),
        confianca_media=("confidence", "mean"),
        acuracia=("is_outcome_correct", "mean"),
    )
    grouped = grouped[grouped["previsoes"] > 0].reset_index()
    return grouped
