import importlib

import pandas as pd
import pytest


def test_api_key_has_no_hardcoded_default(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)

    config = importlib.import_module("src.config")

    class FakeStreamlit:
        secrets = {}

    assert config.get_api_key(FakeStreamlit()) == ""


def test_special_api_statuses_are_valid_filters(tmp_path, monkeypatch):
    database = importlib.import_module("src.database")
    monkeypatch.setattr(database, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "futebot.db"))

    database.init_db()

    df = database.load_matches_by_status("POSTPONED")

    assert list(df.columns)
    assert df.empty


def test_seed_2026_matches_can_be_excluded_from_training(tmp_path, monkeypatch):
    database = importlib.import_module("src.database")
    monkeypatch.setattr(database, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "futebot.db"))

    database.init_db()

    all_finished = database.load_historical_matches(include_seed_2026=True)
    real_training = database.load_historical_matches(include_seed_2026=False)

    assert len(all_finished[all_finished["ano_copa"] == 2026]) > 0
    assert len(real_training[real_training["ano_copa"] == 2026]) == 0
    assert len(real_training) > 0


def test_accuracy_history_excludes_current_match_and_seed_2026():
    accuracy = importlib.import_module("src.accuracy")
    df = pd.DataFrame(
        [
            {"id": 1, "ano_copa": 2022, "data_hora": "2022-11-20 13:00", "origem_dados": "seed"},
            {"id": 2, "ano_copa": 2022, "data_hora": "2022-11-21 13:00", "origem_dados": "seed"},
            {"id": 3, "ano_copa": 2026, "data_hora": "2026-06-11 14:00", "origem_dados": "seed"},
        ]
    )
    row = pd.Series({"id": 2, "data_hora": "2022-11-21 13:00"})

    history = accuracy.build_prediction_history(df, row)

    assert history["id"].tolist() == [1]


def test_team_stats_empty_input_keeps_expected_columns():
    statistics = importlib.import_module("src.statistics")
    teams = pd.DataFrame(
        [
            {"nome": "Brasil", "sigla": "BRA", "elo_rating": 1986.0, "ranking_fifa": 6},
        ]
    )
    matches = pd.DataFrame(
        columns=["mandante_nome", "visitante_nome", "gols_mandante", "gols_visitante"]
    )

    df_stats = statistics.build_team_stats(matches, teams, {}, {})

    assert df_stats.empty
    assert "Seleção" in df_stats.columns


def test_seed_2026_stats_are_available_for_offline_statistics_page(tmp_path, monkeypatch):
    database = importlib.import_module("src.database")
    models = importlib.import_module("src.ML_models")
    statistics = importlib.import_module("src.statistics")
    monkeypatch.setattr(database, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "futebot.db"))

    database.init_db()
    matches = database.load_historical_matches(include_seed_2026=True)
    teams = database.load_all_teams()
    matches_2026 = matches[matches["ano_copa"] == 2026]
    ataque, defesa, _, _ = models.calculate_team_strengths(matches_2026)

    df_stats = statistics.build_team_stats(matches_2026, teams, ataque, defesa)

    assert not df_stats.empty
    assert "Seleção" in df_stats.columns


def test_openfootball_parser_uses_only_finished_matches_when_requested():
    database = importlib.import_module("src.database")
    team_to_id = {"Brasil": 1, "Alemanha": 2, "Argentina": 3, "França": 4}
    payload = {
        "matches": [
            {
                "date": "2026-06-11",
                "time": "14:00",
                "team1": "Brazil",
                "team2": "Germany",
                "group": "Group A",
                "score": {"ft": [2, 1]},
            },
            {
                "date": "2026-06-12",
                "time": "17:00",
                "team1": "Argentina",
                "team2": "France",
                "group": "Group A",
            },
        ]
    }

    partidas = database.parse_openfootball_matches(payload, 2026, team_to_id, finished_only=True)

    assert len(partidas) == 1
    assert partidas[0][0] == 2026
    assert partidas[0][5] == 1
    assert partidas[0][8] == "FINISHED"


def test_openfootball_sync_marks_finished_matches_as_openfootball(tmp_path, monkeypatch):
    database = importlib.import_module("src.database")
    monkeypatch.setattr(database, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "futebot.db"))

    def fake_loader(ano, team_to_id, finished_only=True):
        return [
            (
                ano,
                "2026-06-11 14:00",
                team_to_id["Brasil"],
                team_to_id["Alemanha"],
                2,
                1,
                "1",
                "A",
                "FINISHED",
                team_to_id["Brasil"],
            )
        ]

    monkeypatch.setattr(database, "load_openfootball_data", fake_loader)
    database.init_db()

    updated, status = database.sync_openfootball_finished_matches(2026)
    df = database.load_historical_matches(include_seed_2026=True)
    synced = df[
        (df["ano_copa"] == 2026)
        & (df["mandante_nome"] == "Brasil")
        & (df["visitante_nome"] == "Alemanha")
    ]

    assert updated == 1
    assert status == "ok"
    assert not synced.empty
    assert synced.iloc[0]["origem_dados"] == "openfootball"


def test_openfootball_sync_does_not_duplicate_group_match_when_phase_differs(tmp_path, monkeypatch):
    database = importlib.import_module("src.database")
    monkeypatch.setattr(database, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "futebot.db"))

    def fake_loader(ano, team_to_id, finished_only=True):
        return [
            (
                ano,
                "2026-06-18 15:00",
                team_to_id["Canadá"],
                team_to_id["Catar"],
                6,
                0,
                "1",
                "B",
                "FINISHED",
                team_to_id["Canadá"],
            )
        ]

    monkeypatch.setattr(database, "load_openfootball_data", fake_loader)
    database.init_db()

    updated, status = database.sync_openfootball_finished_matches(2026)
    df = database.load_historical_matches(include_seed_2026=True)
    canada_qatar = df[
        (df["ano_copa"] == 2026)
        & (
            ((df["mandante_nome"] == "Canadá") & (df["visitante_nome"] == "Catar"))
            | ((df["mandante_nome"] == "Catar") & (df["visitante_nome"] == "Canadá"))
        )
    ]

    assert updated == 1
    assert status == "ok"
    assert len(canada_qatar) == 1
    assert canada_qatar.iloc[0]["origem_dados"] == "openfootball"


def test_init_db_removes_existing_seed_duplicate_group_matches(tmp_path, monkeypatch):
    database = importlib.import_module("src.database")
    monkeypatch.setattr(database, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "futebot.db"))
    database.init_db()

    ids = {}
    conn = database.get_connection()
    try:
        rows = conn.execute("SELECT nome, id FROM selecoes WHERE nome IN ('Canadá', 'Catar')").fetchall()
        ids = {row["nome"]: row["id"] for row in rows}
        conn.execute(
            """
            INSERT INTO partidas
            (ano_copa, data_hora, mandante_id, visitante_id, gols_mandante, gols_visitante,
             fase, grupo, status, vencedor_id, origem_dados)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'openfootball')
            """,
            (
                2026,
                "2026-06-18 15:00",
                ids["Canadá"],
                ids["Catar"],
                6,
                0,
                "1",
                "B",
                "FINISHED",
                ids["Canadá"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    database.init_db()
    df = database.load_historical_matches(include_seed_2026=True)
    canada_qatar = df[
        (df["ano_copa"] == 2026)
        & (df["mandante_nome"] == "Canadá")
        & (df["visitante_nome"] == "Catar")
    ]

    assert len(canada_qatar) == 1
    assert canada_qatar.iloc[0]["origem_dados"] == "openfootball"


def test_init_db_removes_duplicate_2026_knockout_matches_from_multiple_sources(tmp_path, monkeypatch):
    database = importlib.import_module("src.database")
    monkeypatch.setattr(database, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "futebot.db"))
    database.init_db()

    conn = database.get_connection()
    try:
        rows = conn.execute("SELECT nome, id FROM selecoes WHERE nome IN ('Brasil', 'Alemanha')").fetchall()
        ids = {row["nome"]: row["id"] for row in rows}
        duplicate_rows = [
            (
                2026,
                "2026-06-29 12:00",
                ids["Brasil"],
                ids["Alemanha"],
                2,
                1,
                "Round of 32",
                None,
                "FINISHED",
                ids["Brasil"],
                "openfootball",
            ),
            (
                2026,
                "2026-06-29 14:00",
                ids["Brasil"],
                ids["Alemanha"],
                2,
                1,
                "Fase de 32",
                None,
                "FINISHED",
                ids["Brasil"],
                "api",
            ),
        ]
        conn.executemany(
            """
            INSERT INTO partidas
            (ano_copa, data_hora, mandante_id, visitante_id, gols_mandante, gols_visitante,
             fase, grupo, status, vencedor_id, origem_dados)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            duplicate_rows,
        )
        conn.commit()
    finally:
        conn.close()

    database.init_db()
    df = database.load_historical_matches(include_seed_2026=False)
    brasil_alemanha = df[
        (df["ano_copa"] == 2026)
        & (
            ((df["mandante_nome"] == "Brasil") & (df["visitante_nome"] == "Alemanha"))
            | ((df["mandante_nome"] == "Alemanha") & (df["visitante_nome"] == "Brasil"))
        )
    ]

    assert len(brasil_alemanha) == 1


def test_api_sync_reuses_2026_knockout_match_when_phase_label_differs(tmp_path, monkeypatch):
    database = importlib.import_module("src.database")
    monkeypatch.setattr(database, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "futebot.db"))
    database.init_db()

    conn = database.get_connection()
    try:
        rows = conn.execute("SELECT nome, id FROM selecoes WHERE nome IN ('Brasil', 'Alemanha')").fetchall()
        ids = {row["nome"]: row["id"] for row in rows}
        conn.execute(
            """
            INSERT INTO partidas
            (ano_copa, data_hora, mandante_id, visitante_id, gols_mandante, gols_visitante,
             fase, grupo, status, vencedor_id, origem_dados)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'openfootball')
            """,
            (
                2026,
                "2026-06-29 12:00",
                ids["Brasil"],
                ids["Alemanha"],
                2,
                1,
                "Round of 32",
                None,
                "FINISHED",
                ids["Brasil"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    database.sync_api_match_to_db(
        {
            "ano_copa": 2026,
            "data_hora": "2026-06-29 14:00",
            "mandante_nome": "Brasil",
            "visitante_nome": "Alemanha",
            "gols_mandante": 2,
            "gols_visitante": 1,
            "status": "FINISHED",
            "fase": "Fase de 32",
            "grupo": None,
            "vencedor_nome": "Brasil",
        }
    )

    df = database.load_historical_matches(include_seed_2026=False)
    brasil_alemanha = df[
        (df["ano_copa"] == 2026)
        & (
            ((df["mandante_nome"] == "Brasil") & (df["visitante_nome"] == "Alemanha"))
            | ((df["mandante_nome"] == "Alemanha") & (df["visitante_nome"] == "Brasil"))
        )
    ]

    assert len(brasil_alemanha) == 1
    assert brasil_alemanha.iloc[0]["origem_dados"] == "api"
