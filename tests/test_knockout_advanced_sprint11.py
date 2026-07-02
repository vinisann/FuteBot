import importlib

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
                "gols_visitante": 1,
                "status": "FINISHED",
                "origem_dados": "api",
            }
        ]
    )


def test_knockout_match_uses_extra_time_before_penalties(monkeypatch):
    models = importlib.import_module("src.ML_models")
    penalty_calls = []

    monkeypatch.setattr(models, "simulate_match_score_from_model", lambda *args, **kwargs: (1, 1))
    monkeypatch.setattr(models, "simulate_extra_time_goals", lambda *args, **kwargs: (1, 0))
    monkeypatch.setattr(
        models,
        "simulate_penalty_shootout",
        lambda *args, **kwargs: penalty_calls.append(args) or (4, 5),
    )

    result = models.simulate_knockout_match(
        "Brasil",
        "Alemanha",
        {},
        {},
        1.0,
        1.0,
        {"Brasil": 1900, "Alemanha": 1800},
        model_history=_history(),
        return_details=True,
    )

    assert result["winner"] == "Brasil"
    assert result["resolution"] == "extra_time"
    assert result["extra_time_score"] == (1, 0)
    assert penalty_calls == []


def test_knockout_match_uses_penalties_after_extra_time_draw(monkeypatch):
    models = importlib.import_module("src.ML_models")

    monkeypatch.setattr(models, "simulate_match_score_from_model", lambda *args, **kwargs: (0, 0))
    monkeypatch.setattr(models, "simulate_extra_time_goals", lambda *args, **kwargs: (0, 0))
    monkeypatch.setattr(models, "simulate_penalty_shootout", lambda *args, **kwargs: (3, 5))

    result = models.simulate_knockout_match(
        "Brasil",
        "Alemanha",
        {},
        {},
        1.0,
        1.0,
        {"Brasil": 1900, "Alemanha": 1800},
        model_history=_history(),
        return_details=True,
    )

    assert result["winner"] == "Alemanha"
    assert result["resolution"] == "penalties"
    assert result["full_time_score"] == (0, 0)
    assert result["extra_time_score"] == (0, 0)
    assert result["penalty_score"] == (3, 5)


def test_format_knockout_score_shows_extra_time_and_penalties():
    models = importlib.import_module("src.ML_models")

    extra_time_text = models.format_knockout_score(
        {
            "winner": "Brasil",
            "full_time_score": (1, 1),
            "extra_time_score": (1, 0),
            "penalty_score": None,
            "resolution": "extra_time",
        }
    )
    penalties_text = models.format_knockout_score(
        {
            "winner": "Alemanha",
            "full_time_score": (0, 0),
            "extra_time_score": (0, 0),
            "penalty_score": (3, 5),
            "resolution": "penalties",
        }
    )

    assert extra_time_text == "2 - 1 a.p."
    assert penalties_text == "0 (3) - (5) 0"
