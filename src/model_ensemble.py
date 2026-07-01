import numpy as np
import pandas as pd


def _normalize_weights(weights):
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0:
        count = max(1, len(weights))
        return {name: 1.0 / count for name in weights}
    return {name: max(0.0, float(value)) / total for name, value in weights.items()}


def calculate_ensemble_weights(df_results, min_sample=2):
    if df_results is None or df_results.empty or "modelo" not in df_results.columns:
        return {}

    df = df_results.dropna(subset=["modelo", "brier_score", "log_loss"]).copy()
    if df.empty:
        return {}

    samples = df.groupby("modelo")["partida_id"].nunique()
    model_names = sorted(samples.index.tolist())
    if not model_names:
        return {}

    if int(samples.min()) < int(min_sample):
        return {name: 1.0 / len(model_names) for name in model_names}

    metrics = df.groupby("modelo").agg(
        brier_score=("brier_score", "mean"),
        log_loss=("log_loss", "mean"),
    )
    raw_weights = {}
    for model_name, row in metrics.iterrows():
        score = float(row["brier_score"]) + float(row["log_loss"])
        raw_weights[model_name] = 1.0 / max(score, 1e-9)

    weights = _normalize_weights(raw_weights)
    floor = 0.08 / len(weights)
    weights = {name: max(floor, weight) for name, weight in weights.items()}
    return _normalize_weights(weights)


def combine_predictions(predictions_by_model, weights):
    valid_models = [
        model_name
        for model_name, prediction in predictions_by_model.items()
        if model_name in weights and prediction
    ]
    if not valid_models:
        raise ValueError("No valid predictions to combine")

    weights = _normalize_weights({name: weights[name] for name in valid_models})
    first_prediction = predictions_by_model[valid_models[0]]
    base_matrix = np.array(first_prediction["matriz_placar"], dtype=float)
    combined_matrix = np.zeros_like(base_matrix, dtype=float)
    prob_m = 0.0
    prob_e = 0.0
    prob_v = 0.0

    for model_name in valid_models:
        prediction = predictions_by_model[model_name]
        weight = weights[model_name]
        matrix = np.array(prediction["matriz_placar"], dtype=float)
        combined_matrix += matrix * weight
        prob_m += float(prediction["prob_vitoria_mandante"]) * weight
        prob_e += float(prediction["prob_empate"]) * weight
        prob_v += float(prediction["prob_vitoria_visitante"]) * weight

    matrix_total = combined_matrix.sum()
    if matrix_total > 0:
        combined_matrix = combined_matrix / matrix_total

    prob_total = prob_m + prob_e + prob_v
    if prob_total > 0:
        prob_m /= prob_total
        prob_e /= prob_total
        prob_v /= prob_total

    m_idx, v_idx = np.unravel_index(np.argmax(combined_matrix), combined_matrix.shape)
    return {
        "prob_vitoria_mandante": float(prob_m),
        "prob_empate": float(prob_e),
        "prob_vitoria_visitante": float(prob_v),
        "matriz_placar": combined_matrix.tolist(),
        "gols_range": first_prediction.get("gols_range", list(range(combined_matrix.shape[0]))),
        "placar_mais_provavel": (int(m_idx), int(v_idx), float(combined_matrix[m_idx, v_idx])),
        "modelo_ensemble": True,
        "pesos_ensemble": weights,
        "modelos_ensemble": valid_models,
        "xG_mandante": float(
            sum(float(predictions_by_model[name].get("xG_mandante", 0.0)) * weights[name] for name in valid_models)
        ),
        "xG_visitante": float(
            sum(float(predictions_by_model[name].get("xG_visitante", 0.0)) * weights[name] for name in valid_models)
        ),
    }
