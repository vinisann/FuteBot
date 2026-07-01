import importlib
import math

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
                "gols_mandante": 2,
                "gols_visitante": 0,
                "status": "FINISHED",
                "origem_dados": "api",
            },
            {
                "id": 2,
                "ano_copa": 2018,
                "data_hora": "2018-06-15 12:00",
                "mandante_nome": "Brasil",
                "visitante_nome": "Canadá",
                "gols_mandante": 1,
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
                "gols_mandante": 3,
                "gols_visitante": 0,
                "status": "FINISHED",
                "origem_dados": "manual",
            },
            {
                "id": 4,
                "ano_copa": 2026,
                "data_hora": "2026-06-11 12:00",
                "mandante_nome": "Brasil",
                "visitante_nome": "Canadá",
                "gols_mandante": 4,
                "gols_visitante": 0,
                "status": "FINISHED",
                "origem_dados": "seed",
            },
            {
                "id": 5,
                "ano_copa": 2026,
                "data_hora": "2026-06-12 12:00",
                "mandante_nome": "Brasil",
                "visitante_nome": "Alemanha",
                "gols_mandante": None,
                "gols_visitante": None,
                "status": "SCHEDULED",
                "origem_dados": "api",
            },
            {
                "id": 6,
                "ano_copa": 2022,
                "data_hora": "2022-06-12 12:00",
                "mandante_nome": "Canadá",
                "visitante_nome": "Alemanha",
                "gols_mandante": 8,
                "gols_visitante": 0,
                "status": "CANCELLED",
                "origem_dados": "api",
            },
        ]
    )


def test_dynamic_elo_uses_only_prior_real_finished_matches():
    dynamic_elo = importlib.import_module("src.dynamic_elo")

    history = dynamic_elo.build_dynamic_elo_history(_matches(), initial_ratings={"Brasil": 1500, "Alemanha": 1500, "Canadá": 1500})

    pre_1 = dynamic_elo.get_pre_match_elo(history, 1, "Brasil", "Alemanha")
    pre_2 = dynamic_elo.get_pre_match_elo(history, 2, "Brasil", "Canadá")
    pre_seed = dynamic_elo.get_pre_match_elo(history, 4, "Brasil", "Canadá")

    assert pre_1 == (1500.0, 1500.0)
    assert pre_2[0] > 1500.0
    assert pre_2[1] == 1500.0
    assert pre_seed == (None, None)


def test_recent_form_excludes_target_seed_and_unfinished_matches():
    dynamic_elo = importlib.import_module("src.dynamic_elo")

    form = dynamic_elo.calculate_recent_form(
        _matches(),
        "Alemanha",
        "Canadá",
        target_date="2026-06-12 12:00",
        target_match_id=5,
    )

    assert 0.9 <= form["mandante_factor"] <= 1.1
    assert 0.9 <= form["visitante_factor"] <= 1.1
    assert form["mandante_games"] == 2
    assert form["visitante_games"] == 2
    assert form["mandante_factor"] > form["visitante_factor"]


def test_prediction_accepts_dynamic_elo_and_recent_form_metadata():
    models = importlib.import_module("src.ML_models")
    matches = _matches().iloc[:3].copy()

    base = models.predict_match_probabilities("Brasil", "Alemanha", 1500, 1500, matches)
    improved = models.predict_match_probabilities(
        "Brasil",
        "Alemanha",
        1500,
        1500,
        matches,
        dynamic_elo=(1560, 1480),
        recent_form={"mandante_factor": 1.08, "visitante_factor": 0.94},
    )

    total_probability = (
        improved["prob_vitoria_mandante"]
        + improved["prob_empate"]
        + improved["prob_vitoria_visitante"]
    )

    assert improved["elo_mandante_usado"] == 1560.0
    assert improved["elo_visitante_usado"] == 1480.0
    assert improved["fator_forma_mandante"] == 1.08
    assert improved["fator_forma_visitante"] == 0.94
    assert improved["modelo_com_forma"] is True
    assert not math.isclose(improved["xG_mandante"], base["xG_mandante"])
    assert math.isclose(total_probability, 1.0, rel_tol=1e-9)


def test_model_evaluation_metrics_are_stable_for_known_cases():
    evaluation = importlib.import_module("src.model_evaluation")

    probs = {"M": 0.7, "E": 0.2, "V": 0.1}

    assert math.isclose(evaluation.calculate_brier_score(probs, "M"), 0.14, rel_tol=1e-9)
    assert math.isclose(evaluation.calculate_log_loss(probs, "M"), -math.log(0.7), rel_tol=1e-9)


def test_model_evaluation_compares_base_dynamic_and_form_variants():
    evaluation = importlib.import_module("src.model_evaluation")
    teams = pd.DataFrame(
        [
            {"nome": "Brasil", "elo_rating": 1900},
            {"nome": "Alemanha", "elo_rating": 1850},
            {"nome": "Canadá", "elo_rating": 1700},
        ]
    )

    results = evaluation.evaluate_model_variants(_matches().iloc[:3].copy(), teams)

    assert set(results["modelo"].unique()) == {
        "Base Poisson-ELO",
        "ELO dinâmico",
        "ELO dinâmico + forma/calibração",
        "ELO dinâmico + forma/calibração + Dixon-Coles",
        "ELO dinâmico + forma/calibração + Dixon-Coles + contexto",
        "Ensemble ponderado",
    }
    assert {"brier_score", "log_loss", "is_outcome_correct", "goal_error"}.issubset(results.columns)
    assert not results["log_loss"].isna().any()


def test_calibration_buckets_group_predictions_by_confidence():
    evaluation = importlib.import_module("src.model_evaluation")
    df = pd.DataFrame(
        [
            {"confidence": 0.72, "is_outcome_correct": True},
            {"confidence": 0.68, "is_outcome_correct": False},
            {"confidence": 0.42, "is_outcome_correct": True},
        ]
    )

    buckets = evaluation.build_calibration_buckets(df)

    assert set(buckets.columns) == {"faixa_confianca", "previsoes", "confianca_media", "acuracia"}
    assert int(buckets["previsoes"].sum()) == 3
