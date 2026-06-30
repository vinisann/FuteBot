import importlib
import math

import numpy as np
import pandas as pd


def _matches():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "ano_copa": 2018,
                "data_hora": "2018-06-10 12:00",
                "mandante_nome": "Brasil",
                "visitante_nome": "Alemanha",
                "gols_mandante": 1,
                "gols_visitante": 1,
                "status": "FINISHED",
                "origem_dados": "api",
            },
            {
                "id": 2,
                "ano_copa": 2018,
                "data_hora": "2018-06-15 12:00",
                "mandante_nome": "Canadá",
                "visitante_nome": "Brasil",
                "gols_mandante": 0,
                "gols_visitante": 1,
                "status": "FINISHED",
                "origem_dados": "api",
            },
            {
                "id": 3,
                "ano_copa": 2018,
                "data_hora": "2018-06-20 12:00",
                "mandante_nome": "Alemanha",
                "visitante_nome": "Canadá",
                "gols_mandante": 0,
                "gols_visitante": 0,
                "status": "FINISHED",
                "origem_dados": "manual",
            },
            {
                "id": 4,
                "ano_copa": 2022,
                "data_hora": "2022-06-20 12:00",
                "mandante_nome": "Brasil",
                "visitante_nome": "Canadá",
                "gols_mandante": 3,
                "gols_visitante": 0,
                "status": "FINISHED",
                "origem_dados": "seed",
            },
            {
                "id": 5,
                "ano_copa": 2022,
                "data_hora": "2022-06-25 12:00",
                "mandante_nome": "Brasil",
                "visitante_nome": "Alemanha",
                "gols_mandante": None,
                "gols_visitante": None,
                "status": "SCHEDULED",
                "origem_dados": "api",
            },
        ]
    )


def test_dixon_coles_rho_zero_keeps_matrix_equivalent():
    dc = importlib.import_module("src.dixon_coles")
    matrix = np.array([[0.20, 0.10], [0.30, 0.40]], dtype=float)

    adjusted, metadata = dc.apply_dixon_coles_correction(matrix, 1.2, 0.9, rho=0.0)

    assert np.allclose(adjusted, matrix / matrix.sum())
    assert metadata["modelo_dixon_coles"] is False
    assert metadata["rho_dixon_coles"] == 0.0


def test_dixon_coles_adjusts_low_scores_and_normalizes():
    dc = importlib.import_module("src.dixon_coles")
    matrix = np.full((4, 4), 1 / 16, dtype=float)

    adjusted, metadata = dc.apply_dixon_coles_correction(matrix, 1.1, 0.8, rho=-0.08)

    assert math.isclose(float(adjusted.sum()), 1.0, rel_tol=1e-9)
    assert not math.isclose(float(adjusted[0, 0]), float(matrix[0, 0]))
    assert not math.isclose(float(adjusted[1, 1]), float(matrix[1, 1]))
    assert math.isclose(float(adjusted[2, 2] / adjusted.sum()), float(adjusted[2, 2]), rel_tol=1e-9)
    assert metadata["modelo_dixon_coles"] is True
    assert metadata["ajuste_placares_baixos"] is True


def test_estimated_rho_ignores_seed_and_unfinished_matches():
    dc = importlib.import_module("src.dixon_coles")

    rho = dc.estimate_dixon_coles_rho(_matches())

    assert -0.15 <= rho <= 0.05
    assert rho < 0.0


def test_prediction_model_applies_optional_dixon_coles_metadata():
    models = importlib.import_module("src.ML_models")
    matches = _matches().iloc[:3].copy()

    base = models.predict_match_probabilities("Brasil", "Alemanha", 1800, 1750, matches)
    corrected = models.predict_match_probabilities(
        "Brasil",
        "Alemanha",
        1800,
        1750,
        matches,
        score_correction={"method": "dixon_coles", "rho": -0.08},
    )

    total_probability = (
        corrected["prob_vitoria_mandante"]
        + corrected["prob_empate"]
        + corrected["prob_vitoria_visitante"]
    )

    assert corrected["modelo_dixon_coles"] is True
    assert corrected["rho_dixon_coles"] == -0.08
    assert corrected["matriz_placar"] != base["matriz_placar"]
    assert math.isclose(total_probability, 1.0, rel_tol=1e-9)


def test_model_evaluation_includes_dixon_coles_variant():
    evaluation = importlib.import_module("src.model_evaluation")
    teams = pd.DataFrame(
        [
            {"nome": "Brasil", "elo_rating": 1900},
            {"nome": "Alemanha", "elo_rating": 1850},
            {"nome": "Canadá", "elo_rating": 1700},
        ]
    )

    results = evaluation.evaluate_model_variants(_matches().iloc[:3].copy(), teams)

    assert "ELO dinâmico + forma/calibração + Dixon-Coles" in set(results["modelo"])
    dixon_rows = results[results["modelo"] == "ELO dinâmico + forma/calibração + Dixon-Coles"]
    assert not dixon_rows.empty
    assert dixon_rows["modelo_dixon_coles"].all()
