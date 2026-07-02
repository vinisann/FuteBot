import importlib


def test_explain_prediction_describes_ensemble_weights_and_confidence():
    explainability = importlib.import_module("src.model_explainability")
    explanation = explainability.explain_prediction(
        {
            "mandante": "Brasil",
            "visitante": "Alemanha",
            "prob_vitoria_mandante": 0.62,
            "prob_empate": 0.23,
            "prob_vitoria_visitante": 0.15,
            "xG_mandante": 1.8,
            "xG_visitante": 0.9,
            "modelo_ensemble": True,
            "pesos_ensemble": {"Base": 0.25, "Avancado": 0.75},
            "elo_mandante_usado": 1900,
            "elo_visitante_usado": 1800,
        }
    )

    assert "Brasil" in explanation["resumo"]
    assert explanation["confianca"] == "alta"
    assert any(f["tipo"] == "ensemble" and f["impacto"] == "positivo" for f in explanation["fatores"])
    assert any("Avancado" in f["descricao"] for f in explanation["fatores"])


def test_explain_prediction_warns_when_calibration_is_inactive():
    explainability = importlib.import_module("src.model_explainability")
    explanation = explainability.explain_prediction(
        {
            "mandante": "Canadá",
            "visitante": "Brasil",
            "prob_vitoria_mandante": 0.20,
            "prob_empate": 0.25,
            "prob_vitoria_visitante": 0.55,
            "xG_mandante": 0.8,
            "xG_visitante": 1.5,
            "modelo_calibrado": False,
            "modelo_com_contexto": False,
        }
    )

    assert explanation["confianca"] == "media"
    assert any("Calibracao incremental inativa" in alert for alert in explanation["alertas"])
    assert any(f["tipo"] == "calibracao" and f["impacto"] == "neutro" for f in explanation["fatores"])


def test_explain_prediction_marks_neutral_context_and_zebra_factor():
    explainability = importlib.import_module("src.model_explainability")
    explanation = explainability.explain_prediction(
        {
            "mandante": "Favorito",
            "visitante": "Azarao",
            "prob_vitoria_mandante": 0.48,
            "prob_empate": 0.27,
            "prob_vitoria_visitante": 0.25,
            "xG_mandante": 1.3,
            "xG_visitante": 1.0,
            "modelo_com_contexto": False,
            "fator_zebra": 0.03,
        }
    )

    assert any("Contexto neutro" in alert for alert in explanation["alertas"])
    assert any(f["tipo"] == "zebra" and f["impacto"] == "alerta" for f in explanation["fatores"])


def test_explain_prediction_includes_player_impact_factor():
    explainability = importlib.import_module("src.model_explainability")
    explanation = explainability.explain_prediction(
        {
            "mandante": "Brasil",
            "visitante": "Alemanha",
            "prob_vitoria_mandante": 0.51,
            "prob_empate": 0.27,
            "prob_vitoria_visitante": 0.22,
            "xG_mandante": 1.5,
            "xG_visitante": 1.1,
            "modelo_com_jogadores": True,
            "fator_jogadores_mandante": 1.04,
            "fator_jogadores_visitante": 0.98,
            "motivos_jogadores": ["Brasil: titulares acima da media elevam levemente o potencial."],
        }
    )

    assert any(f["tipo"] == "jogadores" for f in explanation["fatores"])
    assert any("Brasil" in f["descricao"] for f in explanation["fatores"] if f["tipo"] == "jogadores")


def test_format_explanation_markdown_is_streamlit_friendly():
    explainability = importlib.import_module("src.model_explainability")
    explanation = explainability.explain_prediction(
        {
            "mandante": "Brasil",
            "visitante": "Alemanha",
            "prob_vitoria_mandante": 0.45,
            "prob_empate": 0.30,
            "prob_vitoria_visitante": 0.25,
            "xG_mandante": 1.2,
            "xG_visitante": 1.0,
        }
    )

    markdown = explainability.format_explanation_markdown(explanation)

    assert "Resumo" in markdown
    assert "Fatores" in markdown
    assert "Confiança" in markdown
