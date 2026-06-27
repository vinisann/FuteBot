import pandas as pd


def build_prediction_history(df_matches, target_row):
    """Return only matches that were available before the target match."""
    if df_matches.empty:
        return df_matches.copy()

    history = df_matches.copy()
    target_date = pd.to_datetime(target_row["data_hora"], errors="coerce")
    history_dates = pd.to_datetime(history["data_hora"], errors="coerce")
    history = history[history_dates < target_date]

    if "id" in history.columns and "id" in target_row:
        history = history[history["id"] != target_row["id"]]

    if "origem_dados" in history.columns:
        seed_2026 = (history["ano_copa"] == 2026) & (history["origem_dados"] == "seed")
        history = history[~seed_2026]

    return history.copy()
