# Descriptive PST analyses: learning curves, RT trends, test-phase readouts, the fitting table.
# Work on the PST response table from the browser export or from simulated players; an
# optional ``source`` column ("You", "Simulated", ...) keeps groups apart.

from __future__ import annotations

import pandas as pd

from actors.pst_rl import CONTEXT_FIELDS
from schemas.tasks.pst import PST_MIN_RT_S, block_accuracy, classify_test_pair

MIN_RT_S = PST_MIN_RT_S
FIT_COLUMNS = ["participant_id", "trial_id", "rt", "response", *CONTEXT_FIELDS]
READOUT_LABELS = {"choose_A": "Chose the best symbol", "avoid_B": "Avoided the worst symbol"}


def _keys(df: pd.DataFrame, *columns: str) -> list[str]:
    return [c for c in ("source",) if c in df.columns] + list(columns)


def learning_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Valid learning choices, with ``trial_in_pair`` counting all pair presentations."""
    rows = df[df["phase"] == "learning"].sort_values(_keys(df, "participant_id", "trial"))
    rows = rows.assign(trial_in_pair=rows.groupby(_keys(df, "participant_id", "pair")).cumcount() + 1)
    anticipated = rows.get("anticipated", pd.Series(False, index=rows.index)).astype(bool)
    return rows[~rows["timed_out"].astype(bool) & ~anticipated]


def fit_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Balanced learning panel fitted by RLSSM: no omissions or anticipations.

    RLSSM currently requires equal trial counts. If exclusions differ across participants, each
    participant is truncated to the smallest retained count while preserving within-person order.
    """
    learning = df[df["phase"] == "learning"]
    participant_ids = learning["participant_id"].drop_duplicates()
    rows = learning_rows(df)
    rows = rows[rows["rt"] >= MIN_RT_S]
    if rows.empty:
        return rows
    counts = rows.groupby("participant_id", sort=False).size().reindex(participant_ids, fill_value=0)
    keep = int(counts.min())
    if keep == 0:
        return rows.iloc[0:0]
    return rows.groupby("participant_id", sort=False, group_keys=False).head(keep)


def fit_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """HSSM input (participant_id, rt, response, context fields) and how many trials were dropped."""
    learning = df[df["phase"] == "learning"]
    timeouts = int(learning["timed_out"].astype(bool).sum())
    anticipated = learning.get("anticipated", pd.Series(False, index=learning.index)).astype(bool)
    anticipated = anticipated | learning["rt"].lt(MIN_RT_S).fillna(False)
    anticipations = int((anticipated & ~learning["timed_out"].astype(bool)).sum())
    eligible = learning_rows(df)
    eligible = eligible[eligible["rt"] >= MIN_RT_S]
    rows = fit_rows(df).copy()
    rows["trial_id"] = rows.groupby("participant_id", sort=False).cumcount()
    table = rows[FIT_COLUMNS].astype(float).astype({"participant_id": int, "trial_id": int}).reset_index(drop=True)
    methods = sorted(str(v) for v in learning.get("response_method", pd.Series(dtype=object)).dropna().unique())
    return table, {
        "kept": len(table),
        "timeouts": timeouts,
        "anticipations": anticipations,
        "balancing": len(eligible) - len(rows),
        "response_methods": methods,
        "mixed_response_methods": len(methods) > 1,
    }


def bin_trials(rows: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    """Add ``bin`` (per pair, every ``bin_size`` presentations), its centre ``trials``, and ``chose_better``."""
    rows = rows.assign(
        bin=(rows["trial_in_pair"] - 1) // bin_size + 1,
        chose_better=rows["response"].eq(1),
    )
    return rows.assign(trials=(rows["bin"] - 0.5) * bin_size)


def learning_curves(df: pd.DataFrame, *, bin_size: int = 10, rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of better-symbol choices per pair, in bins of ``bin_size`` presentations of that pair."""
    binned = bin_trials(learning_rows(df) if rows is None else rows, bin_size)
    return (
        binned.groupby(_keys(binned, "pair", "bin", "trials"))
        .agg(p_better=("chose_better", "mean"), n=("chose_better", "size"))
        .reset_index()
    )


def rt_curves(df: pd.DataFrame, *, bin_size: int = 10, rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean and median RT (s) per pair and bin."""
    binned = bin_trials(learning_rows(df) if rows is None else rows, bin_size)
    return (
        binned.groupby(_keys(binned, "pair", "bin", "trials"))
        .agg(rt_mean=("rt", "mean"), rt_median=("rt", "median"), n=("rt", "size"))
        .reset_index()
    )


def signed_rts(df: pd.DataFrame) -> pd.DataFrame:
    """Learning-phase RTs, negative when the worse symbol was chosen (as in the paper's Fig. 4)."""
    rows = learning_rows(df)
    return rows.assign(signed_rt=rows["rt"] * rows["response"])[_keys(rows, "pair", "signed_rt")]


def final_round_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Choose-A and avoid-B: share of better choices on test pairings with A (not B) or B (not A)."""
    anticipated = df.get("anticipated", pd.Series(False, index=df.index)).astype(bool)
    rows = df[(df["phase"] == "test") & ~df["timed_out"].astype(bool) & ~anticipated]
    rows = rows.assign(readout=rows["pair"].map(classify_test_pair), chose_better=rows["response"].eq(1))
    rows = rows.dropna(subset=["readout"])
    out = rows.groupby(_keys(rows, "readout")).agg(p_better=("chose_better", "mean"), n=("chose_better", "size"))
    out = out.reset_index()
    return out.assign(label=out["readout"].map(READOUT_LABELS))


def session_stats(df: pd.DataFrame) -> dict[str, object]:
    """Headline numbers for one session (or pooled players)."""
    learning = df[df["phase"] == "learning"]
    anticipated = learning.get("anticipated", pd.Series(False, index=learning.index)).astype(bool)
    answered = learning[~learning["timed_out"].astype(bool) & ~anticipated]
    last_block = int(learning["block"].max()) if len(learning) else 0
    test = final_round_summary(df).set_index("readout")["p_better"] if (df["phase"] == "test").any() else pd.Series(dtype=float)
    return {
        "points": int(pd.to_numeric(learning["feedback"], errors="coerce").eq(1).sum()),
        "blocks": last_block,
        "trials": len(learning),
        "timeouts": int(learning["timed_out"].astype(bool).sum()),
        "anticipations": int(anticipated.sum()),
        "accuracy": float(answered["response"].eq(1).mean()) if len(answered) else float("nan"),
        "mean_rt": float(answered["rt"].mean()) if len(answered) else float("nan"),
        "last_block_accuracy": block_accuracy(learning[learning["block"] == last_block]),
        "choose_A": float(test.get("choose_A", float("nan"))),
        "avoid_B": float(test.get("avoid_B", float("nan"))),
    }


def near_ceiling(df: pd.DataFrame, threshold: float = 0.95) -> bool:
    """True when every pair is at or above ``threshold`` in the last block: learning rates and
    value sensitivity then trade off and cannot be told apart."""
    learning = df[df["phase"] == "learning"]
    if learning.empty:
        return False
    accuracy = block_accuracy(learning[learning["block"] == learning["block"].max()])
    return bool(accuracy) and min(accuracy.values()) >= threshold
