import importlib
import math

import pandas as pd


def _prediction(home_prob, draw_prob, away_prob, score_prob=0.5):
    return {
        "prob_vitoria_mandante": home_prob,
        "prob_empate": draw_prob,
        "prob_vitoria_visitante": away_prob,
        "matriz_placar": [
            [score_prob, 0.0],
            [0.0, 1.0 - score_prob],
        ],
        "gols_range": [0, 1],
        "placar_mais_provavel": (0, 0, score_prob),
    }


def _matches():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "ano_copa": 2022,
                "data_hora": "2022-11-20 12:00",
                "fase": "Grupo",
                "grupo": "A",
                "mandante_nome": "Brasil",
                "visitante_nome": "Alemanha",
                "gols_mandante": 1,
                "gols_visitante": 0,
                "status": "FINISHED",
                "origem_dados": "api",
            },
            {
                "id": 2,
                "ano_copa": 2022,
                "data_hora": "2022-11-24 12:00",
                "fase": "Grupo",
                "grupo": "A",
                "mandante_nome": "Canadá",
                "visitante_nome": "Brasil",
                "gols_mandante": 0,
                "gols_visitante": 0,
                "status": "FINISHED",
                "origem_dados": "api",
            },
            {
                "id": 3,
                "ano_copa": 2022,
                "data_hora": "2022-11-29 12:00",
                "fase": "Oitavas de Final",
                "grupo": None,
                "mandante_nome": "Alemanha",
                "visitante_nome": "Canadá",
                "gols_mandante": 1,
                "gols_visitante": 1,
                "status": "FINISHED",
                "origem_dados": "manual",
            },
        ]
    )


def test_ensemble_weights_favor_lower_brier_and_log_loss():
    ensemble = importlib.import_module("src.model_ensemble")
    metrics = pd.DataFrame(
        [
            {"modelo": "Base", "brier_score": 0.42, "log_loss": 1.05, "partida_id": 1},
            {"modelo": "Base", "brier_score": 0.40, "log_loss": 1.00, "partida_id": 2},
            {"modelo": "Avancado", "brier_score": 0.20, "log_loss": 0.62, "partida_id": 1},
            {"modelo": "Avancado", "brier_score": 0.22, "log_loss": 0.66, "partida_id": 2},
        ]
    )

    weights = ensemble.calculate_ensemble_weights(metrics)

    assert weights["Avancado"] > weights["Base"]
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)


def test_ensemble_weights_are_conservative_with_small_sample():
    ensemble = importlib.import_module("src.model_ensemble")
    metrics = pd.DataFrame(
        [
            {"modelo": "Base", "brier_score": 0.9, "log_loss": 2.0, "partida_id": 1},
            {"modelo": "Avancado", "brier_score": 0.1, "log_loss": 0.2, "partida_id": 1},
        ]
    )

    weights = ensemble.calculate_ensemble_weights(metrics, min_sample=4)

    assert math.isclose(weights["Base"], 0.5, rel_tol=1e-9)
    assert math.isclose(weights["Avancado"], 0.5, rel_tol=1e-9)


def test_combine_predictions_normalizes_probabilities_and_matrix():
    ensemble = importlib.import_module("src.model_ensemble")
    combined = ensemble.combine_predictions(
        {
            "Base": _prediction(0.50, 0.30, 0.20, score_prob=0.7),
            "Avancado": _prediction(0.65, 0.20, 0.15, score_prob=0.4),
        },
        {"Base": 0.25, "Avancado": 0.75},
    )

    prob_total = (
        combined["prob_vitoria_mandante"]
        + combined["prob_empate"]
        + combined["prob_vitoria_visitante"]
    )
    matrix_total = sum(sum(row) for row in combined["matriz_placar"])

    assert math.isclose(prob_total, 1.0, rel_tol=1e-9)
    assert math.isclose(matrix_total, 1.0, rel_tol=1e-9)
    assert combined["modelo_ensemble"] is True
    assert combined["pesos_ensemble"]["Avancado"] == 0.75


def test_model_evaluation_includes_weighted_ensemble_variant():
    evaluation = importlib.import_module("src.model_evaluation")
    teams = pd.DataFrame(
        [
            {"nome": "Brasil", "elo_rating": 1900},
            {"nome": "Alemanha", "elo_rating": 1850},
            {"nome": "Canadá", "elo_rating": 1700},
        ]
    )

    results = evaluation.evaluate_model_variants(_matches(), teams)

    assert "Ensemble ponderado" in set(results["modelo"])
    ensemble_rows = results[results["modelo"] == "Ensemble ponderado"]
    assert not ensemble_rows.empty
    assert ensemble_rows["modelo_ensemble"].all()
