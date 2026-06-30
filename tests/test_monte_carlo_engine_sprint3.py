import importlib

import numpy as np
import pandas as pd


def _history():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "ano_copa": 2022,
                "data_hora": "2022-11-20 12:00",
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
                "data_hora": "2022-11-25 12:00",
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
                "data_hora": "2022-11-30 12:00",
                "mandante_nome": "Alemanha",
                "visitante_nome": "Canadá",
                "gols_mandante": 2,
                "gols_visitante": 1,
                "status": "FINISHED",
                "origem_dados": "manual",
            },
        ]
    )


def _future_group():
    return pd.DataFrame(
        [
            {
                "id": 10,
                "ano_copa": 2026,
                "data_hora": "2026-06-10 12:00",
                "fase": "Grupo",
                "grupo": "A",
                "mandante_nome": "Brasil",
                "visitante_nome": "Alemanha",
                "gols_mandante": None,
                "gols_visitante": None,
                "status": "SCHEDULED",
            }
        ]
    )


def _predict_with_certain_score(home_goals, away_goals, calls):
    def fake_predict(*args, **kwargs):
        calls.append(kwargs)
        matrix = np.zeros((3, 3), dtype=float)
        matrix[home_goals, away_goals] = 1.0
        return {
            "matriz_placar": matrix.tolist(),
            "gols_range": [0, 1, 2],
            "xG_mandante": 1.0,
            "xG_visitante": 1.0,
            "prob_vitoria_mandante": 1.0 if home_goals > away_goals else 0.0,
            "prob_empate": 1.0 if home_goals == away_goals else 0.0,
            "prob_vitoria_visitante": 1.0 if home_goals < away_goals else 0.0,
            "placar_mais_provavel": (home_goals, away_goals, 1.0),
        }

    return fake_predict


def test_model_engine_samples_score_from_prediction_matrix(monkeypatch):
    models = importlib.import_module("src.ML_models")
    calls = []
    monkeypatch.setattr(models, "predict_match_probabilities", _predict_with_certain_score(2, 1, calls))

    score = models.simulate_match_score_from_model(
        "Brasil",
        "Alemanha",
        {"Brasil": 1900, "Alemanha": 1800},
        _history(),
        score_correction={"method": "dixon_coles", "rho": -0.08},
    )

    assert score == (2, 1)
    assert calls[0]["score_correction"] == {"method": "dixon_coles", "rho": -0.08}


def test_group_simulation_uses_model_engine_when_history_is_available(monkeypatch):
    models = importlib.import_module("src.ML_models")
    calls = []
    monkeypatch.setattr(models, "predict_match_probabilities", _predict_with_certain_score(1, 0, calls))

    simulated = models.simulate_group_matches(
        _future_group(),
        {},
        {},
        1.0,
        1.0,
        {"Brasil": 1900, "Alemanha": 1800},
        model_history=_history(),
        score_correction={"method": "dixon_coles", "rho": -0.08},
    )

    row = simulated.iloc[0]
    assert row["gols_mandante"] == 1
    assert row["gols_visitante"] == 0
    assert row["status"] == "FINISHED"
    assert calls[0]["score_correction"] == {"method": "dixon_coles", "rho": -0.08}


def test_knockout_simulation_uses_model_engine_and_preserves_penalties(monkeypatch):
    models = importlib.import_module("src.ML_models")
    calls = []
    monkeypatch.setattr(models, "predict_match_probabilities", _predict_with_certain_score(0, 1, calls))

    winner = models.simulate_knockout_match(
        "Brasil",
        "Alemanha",
        {},
        {},
        1.0,
        1.0,
        {"Brasil": 1900, "Alemanha": 1800},
        model_history=_history(),
        score_correction={"method": "dixon_coles", "rho": -0.08},
    )

    assert winner == "Alemanha"
    assert calls[0]["score_correction"] == {"method": "dixon_coles", "rho": -0.08}
