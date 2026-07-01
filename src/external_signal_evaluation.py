import pandas as pd


def summarize_external_signal_evaluations(df_predictions):
    """Resume snapshots avaliados agrupando previsoes com e sem sinais externos."""
    if df_predictions is None or df_predictions.empty:
        return pd.DataFrame(
            columns=[
                "grupo",
                "jogos",
                "acuracia_1x2",
                "placar_exato",
                "erro_medio_gols",
                "brier_score",
            ]
        )

    df = df_predictions.copy()
    if "evaluated_at" in df.columns:
        df = df[df["evaluated_at"].notna()]
    required = {"outcome_correct", "score_exact", "goal_error", "brier_score"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "grupo",
                "jogos",
                "acuracia_1x2",
                "placar_exato",
                "erro_medio_gols",
                "brier_score",
            ]
        )

    if "sinais_externos_usados" not in df.columns:
        df["sinais_externos_usados"] = 0
    df["sinais_externos_usados"] = df["sinais_externos_usados"].fillna(0).astype(int)
    df["grupo"] = df["sinais_externos_usados"].map(
        {1: "Com sinais externos", 0: "Sem sinais externos"}
    )

    summary = (
        df.groupby("grupo", as_index=False)
        .agg(
            jogos=("outcome_correct", "count"),
            acuracia_1x2=("outcome_correct", "mean"),
            placar_exato=("score_exact", "mean"),
            erro_medio_gols=("goal_error", "mean"),
            brier_score=("brier_score", "mean"),
        )
        .sort_values("grupo")
        .reset_index(drop=True)
    )
    return summary
