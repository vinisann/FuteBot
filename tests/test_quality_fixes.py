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
