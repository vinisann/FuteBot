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


def _insert_match(database, *, status="SCHEDULED", gols=(None, None)):
    ids = _team_ids(database, "Brasil", "Alemanha")
    conn = database.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO partidas
            (ano_copa, data_hora, mandante_id, visitante_id, gols_mandante, gols_visitante,
             fase, grupo, status, vencedor_id, origem_dados)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'api')
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
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _prediction_with_external_signals():
    return {
        "xG_mandante": 1.55,
        "xG_visitante": 0.82,
        "prob_vitoria_mandante": 0.58,
        "prob_empate": 0.24,
        "prob_vitoria_visitante": 0.18,
        "placar_mais_provavel": (1, 0, 0.14),
        "sinais_externos_usados": True,
        "ajuste_externo_mandante": 1.03,
        "ajuste_externo_visitante": 0.97,
        "motivos_sinais_externos": [
            "Brasil: escalacao provavel encontrada com boa confianca.",
            "Alemanha: possivel desfalque detectado em noticia recente.",
        ],
    }


def test_prediction_snapshot_schema_tracks_external_signal_metadata(tmp_path, monkeypatch):
    database = _use_temp_db(tmp_path, monkeypatch)

    conn = database.get_connection()
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(previsoes_partidas)").fetchall()}
    finally:
        conn.close()

    assert "sinais_externos_usados" in columns
    assert "ajuste_externo_mandante" in columns
    assert "ajuste_externo_visitante" in columns
    assert "motivos_sinais_externos" in columns


def test_prediction_snapshot_persists_external_signal_metadata(tmp_path, monkeypatch):
    database = _use_temp_db(tmp_path, monkeypatch)
    match_id = _insert_match(database)

    database.save_prediction_snapshot(match_id, _prediction_with_external_signals(), "external-v1")
    evaluations = database.load_prediction_evaluations()
    row = evaluations.iloc[0]

    assert bool(row["sinais_externos_usados"]) is True
    assert row["ajuste_externo_mandante"] == 1.03
    assert row["ajuste_externo_visitante"] == 0.97
    assert "Brasil" in row["motivos_sinais_externos"]


def test_external_signal_evaluation_summary_groups_evaluated_snapshots():
    evaluator = importlib.import_module("src.external_signal_evaluation")
    df = pd.DataFrame(
        [
            {
                "sinais_externos_usados": 1,
                "outcome_correct": 1,
                "score_exact": 0,
                "goal_error": 1.0,
                "brier_score": 0.20,
                "evaluated_at": "2026-06-12 12:00:00",
            },
            {
                "sinais_externos_usados": 1,
                "outcome_correct": 0,
                "score_exact": 0,
                "goal_error": 3.0,
                "brier_score": 0.80,
                "evaluated_at": "2026-06-13 12:00:00",
            },
            {
                "sinais_externos_usados": 0,
                "outcome_correct": 1,
                "score_exact": 1,
                "goal_error": 0.0,
                "brier_score": 0.10,
                "evaluated_at": "2026-06-14 12:00:00",
            },
            {
                "sinais_externos_usados": 1,
                "outcome_correct": None,
                "score_exact": None,
                "goal_error": None,
                "brier_score": None,
                "evaluated_at": None,
            },
        ]
    )

    summary = evaluator.summarize_external_signal_evaluations(df)

    with_signals = summary[summary["grupo"] == "Com sinais externos"].iloc[0]
    without_signals = summary[summary["grupo"] == "Sem sinais externos"].iloc[0]
    assert with_signals["jogos"] == 2
    assert with_signals["acuracia_1x2"] == 0.5
    assert with_signals["brier_score"] == 0.5
    assert without_signals["jogos"] == 1
    assert without_signals["placar_exato"] == 1.0
