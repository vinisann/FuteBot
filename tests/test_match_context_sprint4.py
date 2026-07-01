import importlib
import math

import pandas as pd


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
                "gols_mandante": 2,
                "gols_visitante": 0,
                "status": "FINISHED",
                "origem_dados": "api",
            },
            {
                "id": 2,
                "ano_copa": 2022,
                "data_hora": "2022-11-23 12:00",
                "fase": "Grupo",
                "grupo": "A",
                "mandante_nome": "Canadá",
                "visitante_nome": "Brasil",
                "gols_mandante": 1,
                "gols_visitante": 1,
                "status": "FINISHED",
                "origem_dados": "api",
            },
            {
                "id": 3,
                "ano_copa": 2022,
                "data_hora": "2022-11-28 12:00",
                "fase": "Oitavas de Final",
                "grupo": None,
                "mandante_nome": "Alemanha",
                "visitante_nome": "Canadá",
                "gols_mandante": 1,
                "gols_visitante": 0,
                "status": "FINISHED",
                "origem_dados": "manual",
            },
        ]
    )


def test_build_match_context_adjusts_rest_phase_and_weather_with_caps():
    context = importlib.import_module("src.match_context")
    match_context = context.build_match_context(
        _matches(),
        {
            "id": 4,
            "data_hora": "2022-12-01 12:00",
            "fase": "Quartas de Final",
            "mandante_nome": "Brasil",
            "visitante_nome": "Alemanha",
        },
        weather={"temperatura_c": 34, "precipitacao_pct": 70, "vento_kmh": 28},
        venue={"cidade": "Cidade do México", "altitude_m": 2240},
    )

    assert 0.92 <= match_context["fator_mandante"] <= 1.08
    assert 0.92 <= match_context["fator_visitante"] <= 1.08
    assert match_context["modelo_com_contexto"] is True
    assert match_context["fator_fase"] < 1.0
    assert match_context["fator_clima"] < 1.0
    assert "mata-mata" in match_context["contexto_resumo"].lower()


def test_match_context_is_neutral_without_contextual_inputs():
    context = importlib.import_module("src.match_context")
    match_context = context.build_match_context(
        pd.DataFrame(),
        {
            "id": 1,
            "data_hora": "2026-06-10 12:00",
            "fase": "Grupo",
            "mandante_nome": "Brasil",
            "visitante_nome": "Alemanha",
        },
    )

    assert match_context["fator_mandante"] == 1.0
    assert match_context["fator_visitante"] == 1.0
    assert match_context["modelo_com_contexto"] is False


def test_prediction_model_applies_match_context_metadata():
    models = importlib.import_module("src.ML_models")
    history = _matches()

    base = models.predict_match_probabilities("Brasil", "Alemanha", 1900, 1800, history)
    contextual = models.predict_match_probabilities(
        "Brasil",
        "Alemanha",
        1900,
        1800,
        history,
        match_context={
            "fator_mandante": 0.96,
            "fator_visitante": 0.98,
            "fator_descanso_mandante": 0.97,
            "fator_descanso_visitante": 1.0,
            "fator_clima": 0.99,
            "fator_fase": 0.98,
            "modelo_com_contexto": True,
            "contexto_resumo": "Teste contextual",
        },
    )

    total_probability = (
        contextual["prob_vitoria_mandante"]
        + contextual["prob_empate"]
        + contextual["prob_vitoria_visitante"]
    )

    assert contextual["modelo_com_contexto"] is True
    assert contextual["fator_contexto_mandante"] == 0.96
    assert contextual["fator_contexto_visitante"] == 0.98
    assert contextual["contexto_resumo"] == "Teste contextual"
    assert contextual["xG_mandante"] != base["xG_mandante"]
    assert math.isclose(total_probability, 1.0, rel_tol=1e-9)


def test_model_evaluation_includes_context_variant():
    evaluation = importlib.import_module("src.model_evaluation")
    teams = pd.DataFrame(
        [
            {"nome": "Brasil", "elo_rating": 1900},
            {"nome": "Alemanha", "elo_rating": 1850},
            {"nome": "Canadá", "elo_rating": 1700},
        ]
    )

    results = evaluation.evaluate_model_variants(_matches(), teams)

    assert "ELO dinâmico + forma/calibração + Dixon-Coles + contexto" in set(results["modelo"])
    context_rows = results[
        results["modelo"] == "ELO dinâmico + forma/calibração + Dixon-Coles + contexto"
    ]
    assert not context_rows.empty
    assert "modelo_com_contexto" in context_rows.columns
