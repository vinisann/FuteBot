import math


def _lineup():
    return {
        "titulares": [
            "Alisson (GK)",
            "Danilo",
            "Marquinhos",
            "Gabriel",
            "Arana",
            "Casemiro",
            "Bruno Guimaraes",
            "Raphinha",
            "Neymar",
            "Vini Jr",
            "Rodrygo",
        ],
        "banco": ["Ederson (GK)", "Militao", "Paqueta", "Martinelli", "Endrick"],
    }


def _ratings():
    return {
        "Alisson": 89,
        "Danilo": 82,
        "Marquinhos": 86,
        "Gabriel": 84,
        "Arana": 79,
        "Casemiro": 86,
        "Bruno Guimaraes": 84,
        "Raphinha": 84,
        "Neymar": 90,
        "Vini Jr": 91,
        "Rodrygo": 85,
        "Ederson": 88,
        "Militao": 85,
        "Paqueta": 83,
        "Martinelli": 82,
        "Endrick": 80,
    }


def test_player_impact_is_neutral_without_lineup():
    from src.player_impact import calculate_team_player_impact

    impact = calculate_team_player_impact("Brasil")

    assert impact["player_adjustment"] == 1.0
    assert impact["has_player_data"] is False


def test_player_impact_rewards_strong_lineup_with_conservative_cap():
    from src.player_impact import calculate_team_player_impact

    impact = calculate_team_player_impact("Brasil", lineup=_lineup(), player_ratings=_ratings())

    assert impact["has_player_data"] is True
    assert impact["average_starter_rating"] > 84
    assert 1.0 < impact["player_adjustment"] <= 1.06
    assert impact["bench_depth"] > 0.0


def test_absence_of_key_forward_reduces_adjustment_without_overreacting():
    from src.player_impact import calculate_team_player_impact

    impact = calculate_team_player_impact(
        "Brasil",
        lineup=_lineup(),
        player_ratings=_ratings(),
        unavailable_players=["Vini Jr", "Neymar"],
    )

    assert impact["absence_penalty"] > 0.0
    assert 0.94 <= impact["player_adjustment"] < 1.0
    assert any("desfalque" in reason.lower() for reason in impact["reasons"])


def test_build_match_player_impact_returns_mandante_and_visitante_adjustments():
    from src.player_impact import build_match_player_impact

    impact = build_match_player_impact(
        "Brasil",
        "Alemanha",
        lineups={"Brasil": _lineup(), "Alemanha": None},
        player_ratings={"Brasil": _ratings(), "Alemanha": {}},
        unavailable_players={"Brasil": ["Neymar"], "Alemanha": []},
    )

    assert impact["mandante"]["team"] == "Brasil"
    assert impact["visitante"]["team"] == "Alemanha"
    assert impact["modelo_com_jogadores"] is True
    assert impact["fator_jogadores_mandante"] != 1.0
    assert math.isclose(impact["fator_jogadores_visitante"], 1.0)


def test_external_signals_apply_player_impact_metadata():
    from src.external_signals import apply_external_signal_adjustment
    from src.player_impact import build_match_player_impact

    player_impact = build_match_player_impact(
        "Brasil",
        "Alemanha",
        lineups={"Brasil": _lineup(), "Alemanha": None},
        player_ratings={"Brasil": _ratings(), "Alemanha": {}},
    )
    external_signals = {
        "mandante": {"external_adjustment": 1.0, "reasons": []},
        "visitante": {"external_adjustment": 1.0, "reasons": []},
        "sinais_externos_usados": True,
        "motivos_sinais_externos": [],
        "player_impact": player_impact,
    }

    lambda_m, lambda_v, meta = apply_external_signal_adjustment(
        1.4, 1.1, "Brasil", "Alemanha", external_signals
    )

    assert lambda_m > 1.4
    assert lambda_v == 1.1
    assert meta["modelo_com_jogadores"] is True
    assert meta["fator_jogadores_mandante"] > 1.0
    assert any("titulares" in reason.lower() for reason in meta["motivos_jogadores"])
