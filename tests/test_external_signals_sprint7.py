import math

import pandas as pd


def _sample_matches():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "mandante_nome": "Brasil",
                "visitante_nome": "Alemanha",
                "gols_mandante": 2,
                "gols_visitante": 1,
            },
            {
                "id": 2,
                "mandante_nome": "Alemanha",
                "visitante_nome": "Brasil",
                "gols_mandante": 1,
                "gols_visitante": 1,
            },
            {
                "id": 3,
                "mandante_nome": "Brasil",
                "visitante_nome": "Franca",
                "gols_mandante": 3,
                "gols_visitante": 0,
            },
            {
                "id": 4,
                "mandante_nome": "Alemanha",
                "visitante_nome": "Franca",
                "gols_mandante": 0,
                "gols_visitante": 2,
            },
        ]
    )


def test_external_signals_are_neutral_without_evidence():
    from src.external_signals import build_match_external_signals

    signals = build_match_external_signals("Brasil", "Alemanha")

    assert signals["mandante"]["external_adjustment"] == 1.0
    assert signals["visitante"]["external_adjustment"] == 1.0
    assert signals["sinais_externos_usados"] is False


def test_injury_text_reduces_team_adjustment_with_cap():
    from src.external_signals import extract_team_external_signals

    signals = extract_team_external_signals(
        "Brasil",
        news_items=[
            "Brasil tem lesao de titular e suspensao no meio-campo",
            "Novo desfalque preocupa a comissao tecnica antes da partida",
        ],
    )

    assert signals["injury_risk"] > 0.0
    assert 0.95 <= signals["external_adjustment"] < 1.0
    assert any("desfalque" in reason.lower() or "lesao" in reason.lower() for reason in signals["reasons"])


def test_probable_lineup_text_increases_confidence_with_cap():
    from src.external_signals import extract_team_external_signals

    signals = extract_team_external_signals(
        "Alemanha",
        lineup_text="Provavel escalacao: Neuer; Kimmich, Rudiger, Tah, Raum; Kroos, Gundogan; Musiala, Wirtz, Sane; Havertz.",
    )

    assert signals["lineup_confidence"] > 0.0
    assert 1.0 < signals["external_adjustment"] <= 1.05
    assert any("escalacao" in reason.lower() for reason in signals["reasons"])


def test_prediction_is_unchanged_without_external_signals():
    from src.ML_models import predict_match_probabilities

    base = predict_match_probabilities("Brasil", "Alemanha", 1950, 1900, _sample_matches())
    neutral = predict_match_probabilities(
        "Brasil", "Alemanha", 1950, 1900, _sample_matches(), external_signals=None
    )

    assert neutral["xG_mandante"] == base["xG_mandante"]
    assert neutral["xG_visitante"] == base["xG_visitante"]
    assert neutral["prob_vitoria_mandante"] == base["prob_vitoria_mandante"]
    assert neutral["sinais_externos_usados"] is False


def test_prediction_applies_external_adjustments_and_keeps_probabilities_normalized():
    from src.ML_models import predict_match_probabilities
    from src.external_signals import build_match_external_signals

    base = predict_match_probabilities("Brasil", "Alemanha", 1950, 1900, _sample_matches())
    external = build_match_external_signals(
        "Brasil",
        "Alemanha",
        news_items={
            "Brasil": ["Brasil deve repetir titulares e chega com time definido"],
            "Alemanha": [
                "Alemanha tem lesao de titular confirmada",
                "Suspensao deixa Alemanha com novo desfalque",
            ],
        },
        lineups={
            "Brasil": "Provavel escalacao: Alisson; Danilo, Marquinhos, Gabriel, Arana; Casemiro, Bruno; Raphinha, Neymar, Vini; Rodrygo."
        },
    )
    adjusted = predict_match_probabilities(
        "Brasil", "Alemanha", 1950, 1900, _sample_matches(), external_signals=external
    )

    total_prob = (
        adjusted["prob_vitoria_mandante"]
        + adjusted["prob_empate"]
        + adjusted["prob_vitoria_visitante"]
    )
    assert adjusted["xG_mandante"] > base["xG_mandante"]
    assert adjusted["xG_visitante"] < base["xG_visitante"]
    assert adjusted["ajuste_externo_mandante"] <= 1.05
    assert adjusted["ajuste_externo_visitante"] >= 0.95
    assert math.isclose(total_prob, 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert adjusted["sinais_externos_usados"] is True


def test_explainability_includes_external_signal_reasons():
    from src.model_explainability import explain_prediction

    explanation = explain_prediction(
        {
            "mandante": "Brasil",
            "visitante": "Alemanha",
            "xG_mandante": 1.6,
            "xG_visitante": 1.1,
            "prob_vitoria_mandante": 0.52,
            "prob_empate": 0.25,
            "prob_vitoria_visitante": 0.23,
            "sinais_externos_usados": True,
            "ajuste_externo_mandante": 1.03,
            "ajuste_externo_visitante": 0.97,
            "motivos_sinais_externos": [
                "Brasil: escalacao provavel encontrada com boa confianca.",
                "Alemanha: possivel desfalque detectado em noticia recente.",
            ],
        }
    )

    external_factors = [factor for factor in explanation["fatores"] if factor["tipo"] == "sinais_externos"]

    assert external_factors
    assert "Brasil" in external_factors[0]["descricao"]
