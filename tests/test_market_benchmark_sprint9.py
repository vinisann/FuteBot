import math


def test_implied_probabilities_remove_bookmaker_margin():
    from src.market_benchmark import implied_probabilities_from_odds

    implied = implied_probabilities_from_odds(
        {"mandante": 1.90, "empate": 3.40, "visitante": 4.50}
    )

    assert math.isclose(sum(implied.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert implied["mandante"] > implied["empate"] > implied["visitante"]


def test_average_market_probabilities_across_houses():
    from src.market_benchmark import average_market_probabilities

    market = average_market_probabilities(
        {
            "Casa A": {"mandante": 2.00, "empate": 3.50, "visitante": 4.00},
            "Casa B": {"mandante": 1.80, "empate": 3.60, "visitante": 4.80},
        }
    )

    assert math.isclose(sum(market.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert market["mandante"] > market["empate"]


def test_compare_model_to_market_flags_large_disagreement():
    from src.market_benchmark import compare_model_to_market

    benchmark = compare_model_to_market(
        model_probabilities={"mandante": 0.65, "empate": 0.22, "visitante": 0.13},
        market_odds={
            "Casa A": {"mandante": 2.60, "empate": 3.30, "visitante": 2.80},
            "Casa B": {"mandante": 2.50, "empate": 3.20, "visitante": 2.90},
        },
    )

    assert benchmark["classificacao"] == "divergencia_alta"
    assert benchmark["maior_divergencia_resultado"] == "mandante"
    assert benchmark["maior_divergencia_pp"] >= 20.0
    assert benchmark["usar_no_treino"] is False


def test_compare_model_to_market_marks_aligned_proxy():
    from src.market_benchmark import compare_model_to_market

    benchmark = compare_model_to_market(
        model_probabilities={"mandante": 0.50, "empate": 0.25, "visitante": 0.25},
        market_odds={
            "Proxy": {"mandante": 1.90, "empate": 3.80, "visitante": 3.80},
        },
    )

    assert benchmark["classificacao"] == "alinhado"
    assert benchmark["maior_divergencia_pp"] < 5.0
    assert "benchmark" in benchmark["resumo"].lower()


def test_explainability_includes_market_benchmark_factor():
    from src.model_explainability import explain_prediction

    explanation = explain_prediction(
        {
            "mandante": "Brasil",
            "visitante": "Alemanha",
            "xG_mandante": 1.4,
            "xG_visitante": 1.1,
            "prob_vitoria_mandante": 0.45,
            "prob_empate": 0.28,
            "prob_vitoria_visitante": 0.27,
            "market_benchmark": {
                "classificacao": "divergencia_alta",
                "resumo": "Benchmark de odds: maior diferenca em vitoria do mandante.",
            },
        }
    )

    factors = [factor for factor in explanation["fatores"] if factor["tipo"] == "odds_benchmark"]

    assert factors
    assert factors[0]["impacto"] == "alerta"
