import importlib

import pandas as pd


def _matches():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "ano_copa": 2018,
                "data_hora": "2018-06-10 12:00",
                "fase": "Grupo",
                "grupo": "A",
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
                "fase": "Grupo",
                "grupo": "A",
                "mandante_nome": "Brasil",
                "visitante_nome": "Canada",
                "gols_mandante": 1,
                "gols_visitante": 1,
                "status": "FINISHED",
                "origem_dados": "api",
            },
            {
                "id": 3,
                "ano_copa": 2018,
                "data_hora": "2018-06-20 12:00",
                "fase": "Oitavas de Final",
                "grupo": None,
                "mandante_nome": "Alemanha",
                "visitante_nome": "Canada",
                "gols_mandante": 3,
                "gols_visitante": 0,
                "status": "FINISHED",
                "origem_dados": "manual",
            },
            {
                "id": 4,
                "ano_copa": 2022,
                "data_hora": "2022-11-20 12:00",
                "fase": "Grupo",
                "grupo": "B",
                "mandante_nome": "Brasil",
                "visitante_nome": "Canada",
                "gols_mandante": 3,
                "gols_visitante": 1,
                "status": "FINISHED",
                "origem_dados": "api",
            },
            {
                "id": 5,
                "ano_copa": 2022,
                "data_hora": "2022-11-25 12:00",
                "fase": "Quartas de Final",
                "grupo": None,
                "mandante_nome": "Alemanha",
                "visitante_nome": "Brasil",
                "gols_mandante": 0,
                "gols_visitante": 1,
                "status": "FINISHED",
                "origem_dados": "api",
            },
        ]
    )


def _teams():
    return pd.DataFrame(
        [
            {"nome": "Brasil", "elo_rating": 1900},
            {"nome": "Alemanha", "elo_rating": 1840},
            {"nome": "Canada", "elo_rating": 1660},
        ]
    )


def test_evaluate_model_variants_includes_elo_baseline_and_deep_columns():
    evaluation = importlib.import_module("src.model_evaluation")

    results = evaluation.evaluate_model_variants(_matches(), _teams())

    assert "Baseline ELO simples" in set(results["modelo"])
    assert {
        "history_matches_used",
        "history_cutoff",
        "confidence_bucket",
        "favorite_side",
        "actual_is_upset",
        "prob_actual",
    }.issubset(results.columns)
    assert (results["history_matches_used"] >= 1).all()


def test_deep_backtest_summary_reports_deltas_and_reliability():
    evaluation = importlib.import_module("src.model_evaluation")
    results = evaluation.evaluate_model_variants(_matches(), _teams())

    summary = evaluation.build_deep_backtest_summary(results)

    assert {
        "modelo",
        "jogos",
        "acuracia_1x2",
        "brier_score",
        "log_loss",
        "delta_brier_vs_baseline",
        "delta_log_loss_vs_baseline",
        "calibration_error",
        "overconfidence_gap",
    }.issubset(summary.columns)
    assert not summary["delta_brier_vs_baseline"].isna().any()
    assert (summary["calibration_error"] >= 0).all()


def test_segment_performance_finds_weak_slices_by_phase_year_and_confidence():
    evaluation = importlib.import_module("src.model_evaluation")
    results = evaluation.evaluate_model_variants(_matches(), _teams())

    segments = evaluation.build_segment_performance(results, min_games=1)

    assert {"segmento_tipo", "segmento", "modelo", "jogos", "brier_score", "log_loss"}.issubset(
        segments.columns
    )
    assert {"fase", "ano_copa", "confidence_bucket"}.issubset(set(segments["segmento_tipo"]))
    assert (segments["jogos"] >= 1).all()


def test_backtest_diagnostics_detects_small_sample_and_temporal_coverage():
    evaluation = importlib.import_module("src.model_evaluation")
    results = evaluation.evaluate_model_variants(_matches(), _teams())

    diagnostics = evaluation.build_backtest_diagnostics(results, min_reliable_games=10)

    assert diagnostics["total_games"] == results["partida_id"].nunique()
    assert diagnostics["sample_warning"] is True
    assert diagnostics["date_start"] <= diagnostics["date_end"]
    assert diagnostics["models_evaluated"] >= 2
