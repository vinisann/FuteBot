import importlib

import pandas as pd


def _use_temp_db(tmp_path, monkeypatch):
    database = importlib.import_module("src.database")
    monkeypatch.setattr(database, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "futebot.db"))
    database.init_db()
    return database


def _team_ids(database, *names):
    conn = database.get_connection()
    try:
        rows = conn.execute(
            f"SELECT nome, id FROM selecoes WHERE nome IN ({','.join('?' for _ in names)})",
            names,
        ).fetchall()
        return {row["nome"]: row["id"] for row in rows}
    finally:
        conn.close()


def _insert_match(database, tmp_path, monkeypatch, *, status="SCHEDULED", origem="api", gols=(None, None)):
    ids = _team_ids(database, "Brasil", "Alemanha")
    conn = database.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO partidas
            (ano_copa, data_hora, mandante_id, visitante_id, gols_mandante, gols_visitante,
             fase, grupo, status, vencedor_id, origem_dados)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                2026,
                "2026-06-11 14:00",
                ids["Brasil"],
                ids["Alemanha"],
                gols[0],
                gols[1],
                "Grupo",
                "A",
                status,
                ids["Brasil"] if gols[0] is not None and gols[1] is not None and gols[0] > gols[1] else None,
                origem,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _prediction():
    return {
        "xG_mandante": 1.45,
        "xG_visitante": 0.85,
        "prob_vitoria_mandante": 0.55,
        "prob_empate": 0.25,
        "prob_vitoria_visitante": 0.20,
        "placar_mais_provavel": (1, 0, 0.13),
    }


def test_prediction_snapshot_is_saved_once_per_match_and_model(tmp_path, monkeypatch):
    database = _use_temp_db(tmp_path, monkeypatch)
    match_id = _insert_match(database, tmp_path, monkeypatch)

    first_id = database.save_prediction_snapshot(match_id, _prediction(), "calibrated-v1")
    second_id = database.save_prediction_snapshot(match_id, _prediction(), "calibrated-v1")
    evaluations = database.load_prediction_evaluations()

    assert first_id == second_id
    assert len(evaluations) == 1
    assert evaluations.iloc[0]["partida_id"] == match_id
    assert evaluations.iloc[0]["evaluated_at"] is None


def test_finished_real_match_evaluates_prediction_metrics(tmp_path, monkeypatch):
    database = _use_temp_db(tmp_path, monkeypatch)
    match_id = _insert_match(database, tmp_path, monkeypatch)
    database.save_prediction_snapshot(match_id, _prediction(), "calibrated-v1")
    database.update_live_match(match_id, 2, 1, "FINISHED")

    evaluated = database.evaluate_finished_predictions()
    evaluations = database.load_prediction_evaluations()
    row = evaluations.iloc[0]

    assert evaluated == 0
    assert row["gols_mandante_real"] == 2
    assert row["gols_visitante_real"] == 1
    assert bool(row["outcome_correct"]) is True
    assert bool(row["score_exact"]) is False
    assert row["goal_error"] == 2
    assert row["brier_score"] > 0
    assert row["evaluated_at"] is not None


def test_seed_or_unfinished_matches_are_not_evaluated(tmp_path, monkeypatch):
    database = _use_temp_db(tmp_path, monkeypatch)
    scheduled_id = _insert_match(database, tmp_path, monkeypatch, status="SCHEDULED", origem="api")
    seed_id = _insert_match(database, tmp_path, monkeypatch, status="FINISHED", origem="seed", gols=(2, 0))
    database.save_prediction_snapshot(scheduled_id, _prediction(), "calibrated-v1")
    database.save_prediction_snapshot(seed_id, _prediction(), "calibrated-v1")

    evaluated = database.evaluate_finished_predictions()
    evaluations = database.load_prediction_evaluations()

    assert evaluated == 0
    assert evaluations["evaluated_at"].isna().all()


def test_calibration_is_neutral_without_minimum_sample():
    calibration_module = importlib.import_module("src.model_calibration")
    calibration = calibration_module.build_model_calibration(
        pd.DataFrame(),
        pd.DataFrame(
            [
                {
                    "mandante_nome": "Brasil",
                    "visitante_nome": "Alemanha",
                    "xg_mandante": 1.0,
                    "xg_visitante": 1.0,
                    "gols_mandante_real": 2,
                    "gols_visitante_real": 0,
                    "prob_mandante": 0.5,
                    "prob_empate": 0.25,
                    "prob_visitante": 0.25,
                    "evaluated_at": "2026-06-11 16:00",
                }
            ]
        ),
    )

    lambda_m, lambda_v, metadata = calibration_module.apply_calibration_to_lambdas(
        1.4, 0.9, "Brasil", "Alemanha", calibration
    )

    assert lambda_m == 1.4
    assert lambda_v == 0.9
    assert metadata["modelo_calibrado"] is False
    assert metadata["fator_zebra"] == 0.0


def test_conservative_calibration_caps_adjustments_and_adds_small_upset_factor():
    calibration_module = importlib.import_module("src.model_calibration")
    rows = []
    for idx in range(4):
        rows.append(
            {
                "mandante_nome": "Favorito",
                "visitante_nome": "Azarao",
                "xg_mandante": 2.4,
                "xg_visitante": 0.4,
                "gols_mandante_real": 0,
                "gols_visitante_real": 2,
                "prob_mandante": 0.78,
                "prob_empate": 0.14,
                "prob_visitante": 0.08,
                "evaluated_at": f"2026-06-1{idx} 18:00",
            }
        )

    calibration = calibration_module.build_model_calibration(pd.DataFrame(), pd.DataFrame(rows))
    lambda_m, lambda_v, metadata = calibration_module.apply_calibration_to_lambdas(
        2.0, 0.8, "Favorito", "Azarao", calibration
    )

    assert round(lambda_m, 6) >= 1.8
    assert round(lambda_v, 6) <= 0.88
    assert metadata["modelo_calibrado"] is True
    assert 0.0 < metadata["fator_zebra"] <= 0.04


def test_prediction_model_applies_optional_calibration_metadata():
    models = importlib.import_module("src.ML_models")
    matches = pd.DataFrame(
        [
            {
                "id": 1,
                "mandante_nome": "Brasil",
                "visitante_nome": "Alemanha",
                "gols_mandante": 2,
                "gols_visitante": 0,
            },
            {
                "id": 2,
                "mandante_nome": "Alemanha",
                "visitante_nome": "Brasil",
                "gols_mandante": 1,
                "gols_visitante": 1,
            },
        ]
    )
    calibration = {
        "active": True,
        "team_adjustments": {
            "Brasil": {"attack": 1.10, "defense": 0.95},
            "Alemanha": {"attack": 0.90, "defense": 1.05},
        },
        "upset_factor": 0.03,
    }

    base = models.predict_match_probabilities("Brasil", "Alemanha", 1900, 1800, matches)
    calibrated = models.predict_match_probabilities(
        "Brasil", "Alemanha", 1900, 1800, matches, calibration=calibration
    )

    assert calibrated["modelo_calibrado"] is True
    assert calibrated["fator_zebra"] == 0.03
    assert calibrated["xG_mandante"] != base["xG_mandante"]
    assert calibrated["xG_visitante"] != base["xG_visitante"]
