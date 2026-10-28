"""Descriptive summaries and HSSM-ready tables for PAAT responses."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

MIN_RT_S = 0.25
MAX_TIMEOUT_RATE_FOR_UNCENSORED_FIT = 0.05
FIT_COLUMNS = [
    "participant_id",
    "rt",
    "response",
    "rel_reward_z",
    "rel_aversive_z",
    "congruency_code",
]


def _group_keys(df: pd.DataFrame, *keys: str) -> list[str]:
    return (["source"] if "source" in df.columns else []) + list(keys)


def answered_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[~df["timed_out"].astype(bool) & df["response"].isin((-1, 1))].copy()


def fit_rows(df: pd.DataFrame, *, min_rt_s: float = MIN_RT_S) -> pd.DataFrame:
    rows = answered_rows(df)
    return rows[pd.to_numeric(rows["rt"], errors="coerce") > float(min_rt_s)].copy()


def fit_table(df: pd.DataFrame, *, min_rt_s: float = MIN_RT_S) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return uncensored HSSM rows plus explicit omission/exclusion diagnostics.

    The installed stock ``angle`` likelihood models observed choices and RTs,
    not this task's right-censored no-response trials.  Callers must not fit
    when ``high_timeout_rate`` is true.
    """
    answered = answered_rows(df)
    rows = fit_rows(df, min_rt_s=min_rt_s)
    table = rows[FIT_COLUMNS].copy()
    if not table.empty:
        table = table.astype(
            {
                "participant_id": int,
                "rt": float,
                "response": int,
                "rel_reward_z": float,
                "rel_aversive_z": float,
                "congruency_code": float,
            }
        )
    timeouts = int(df["timed_out"].astype(bool).sum()) if not df.empty else 0
    timeout_rate = timeouts / len(df) if len(df) else 0.0
    if not df.empty and "participant_id" in df:
        participant_timeout_rates = {
            int(participant_id): float(rate)
            for participant_id, rate in (
                df.assign(_timed_out=df["timed_out"].astype(bool))
                .groupby("participant_id")["_timed_out"]
                .mean()
                .items()
            )
        }
    else:
        participant_timeout_rates = {}
    max_participant_timeout_rate = max(participant_timeout_rates.values(), default=0.0)
    participants_above_timeout_limit = sum(
        rate > MAX_TIMEOUT_RATE_FOR_UNCENSORED_FIT
        for rate in participant_timeout_rates.values()
    )
    return table.reset_index(drop=True), {
        "kept": len(table),
        "total": len(df),
        "timeouts": timeouts,
        "timeout_rate": timeout_rate,
        "participant_timeout_rates": participant_timeout_rates,
        "max_participant_timeout_rate": max_participant_timeout_rate,
        "participants_above_timeout_limit": participants_above_timeout_limit,
        "high_timeout_rate": (
            timeout_rate > MAX_TIMEOUT_RATE_FOR_UNCENSORED_FIT
            or participants_above_timeout_limit > 0
        ),
        "anticipations": len(answered) - len(rows),
    }


def require_defensible_uncensored_fit(exclusions: dict[str, object]) -> None:
    """Reject an observed-response fit when deadline censoring is material."""
    if bool(exclusions.get("high_timeout_rate", False)):
        raise ValueError(
            f"{int(exclusions['timeouts'])} of {int(exclusions['total'])} trials "
            f"({float(exclusions['timeout_rate']):.1%}) missed the deadline; the maximum "
            f"participant rate is {float(exclusions['max_participant_timeout_rate']):.1%} "
            f"and {int(exclusions['participants_above_timeout_limit'])} participant(s) exceed 5%. "
            "The stock angle LAN does not model these right-censored no-response trials. "
            f"Observed-response fits require at most {MAX_TIMEOUT_RATE_FOR_UNCENSORED_FIT:.0%} omissions."
        )


def regression_design_diagnostics(df: pd.DataFrame) -> dict[str, object]:
    """Expose the reward/congruency support limitation of the paper formula."""
    required = {"rel_reward_z", "rel_aversive_z", "congruency_code"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"PAAT design diagnostics are missing columns: {missing}")
    if df.empty:
        raise ValueError("PAAT design diagnostics require at least one row")

    reward = pd.to_numeric(df["rel_reward_z"], errors="raise").to_numpy(dtype=float)
    congruency = pd.to_numeric(df["congruency_code"], errors="raise").to_numpy(dtype=float)
    correlation = float(np.corrcoef(reward, congruency)[0, 1])
    support: dict[int, tuple[float, float]] = {}
    for code in (0, 1):
        values = reward[congruency == code]
        if not len(values):
            raise ValueError(f"PAAT design has no congruency_code={code} rows")
        support[code] = (float(values.min()), float(values.max()))
    overlap = max(support[0][0], support[1][0]) <= min(support[0][1], support[1][1])
    zero_supported = all(lower <= 0.0 <= upper for lower, upper in support.values())
    return {
        "reward_congruency_correlation": correlation,
        "reward_support": support,
        "reward_supports_overlap": overlap,
        "nearest_abs_reward_z": float(np.abs(reward).min()),
        "zero_reward_observed": bool(np.isclose(reward, 0.0).any()),
        "congruency_main_effect_supported": zero_supported,
    }


def data_fingerprint(table: pd.DataFrame) -> str:
    """Stable fingerprint for the exact ordered rows supplied to HSSM."""
    missing = [column for column in FIT_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"fit table is missing columns: {missing}")
    payload = table[FIT_COLUMNS].reset_index(drop=True).to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
        na_rep="<NA>",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = answered_rows(df).assign(chose_risky=lambda x: x["response"].eq(1))
    if rows.empty:
        return pd.DataFrame(columns=[*_group_keys(df, "congruency"), "p_risky", "rt_median", "rt_mean", "n"])
    return (
        rows.groupby(_group_keys(rows, "congruency"), as_index=False)
        .agg(p_risky=("chose_risky", "mean"), rt_median=("rt", "median"), rt_mean=("rt", "mean"), n=("rt", "size"))
    )


def evidence_summary(df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    if predictor not in ("rel_reward", "rel_aversive"):
        raise ValueError("predictor must be rel_reward or rel_aversive")
    rows = answered_rows(df).assign(
        chose_risky=lambda x: x["response"].eq(1),
        **{predictor: lambda x: pd.to_numeric(x[predictor], errors="coerce").round(6)},
    )
    if rows.empty:
        return pd.DataFrame()
    return (
        rows.groupby(_group_keys(rows, "congruency", predictor), as_index=False)
        .agg(p_risky=("chose_risky", "mean"), rt_median=("rt", "median"), n=("rt", "size"))
        .sort_values(_group_keys(rows, "congruency", predictor))
    )


def choice_surface(df: pd.DataFrame) -> pd.DataFrame:
    rows = answered_rows(df).assign(chose_risky=lambda x: x["response"].eq(1))
    if rows.empty:
        return pd.DataFrame()
    return (
        rows.groupby(_group_keys(rows, "congruency", "rel_reward", "rel_aversive"), as_index=False)
        .agg(p_risky=("chose_risky", "mean"), rt_median=("rt", "median"), n=("rt", "size"))
    )


def signed_rts(df: pd.DataFrame) -> pd.DataFrame:
    rows = answered_rows(df)
    if rows.empty:
        return rows.assign(signed_rt=pd.Series(dtype=float))
    return rows.assign(signed_rt=rows["rt"] * rows["response"])


def session_stats(df: pd.DataFrame) -> dict[str, object]:
    rows = answered_rows(df)
    by_condition = condition_summary(df).set_index("congruency") if not rows.empty else pd.DataFrame()
    rewards = pd.to_numeric(df["reward_value"], errors="coerce").fillna(0) if "reward_value" in df else pd.Series(dtype=float)
    return {
        "trials": len(df),
        "answered": len(rows),
        "timeouts": len(df) - len(rows),
        "p_risky": float(rows["response"].eq(1).mean()) if len(rows) else float("nan"),
        "median_rt": float(rows["rt"].median()) if len(rows) else float("nan"),
        "reward": float(rewards.sum()),
        "p_risky_incongruent": (
            float(by_condition.loc["incongruent", "p_risky"])
            if not by_condition.empty and "incongruent" in by_condition.index
            else float("nan")
        ),
        "p_risky_congruent": (
            float(by_condition.loc["congruent", "p_risky"])
            if not by_condition.empty and "congruent" in by_condition.index
            else float("nan")
        ),
    }


def drift_grid(coefficients, design: pd.DataFrame) -> pd.DataFrame:
    """Evaluate drift only at predictor cells present in the supplied design."""
    required = {"rel_reward_z", "rel_aversive_z", "congruency_code"}
    missing = sorted(required - set(design.columns))
    if missing:
        raise ValueError(f"drift-grid design is missing columns: {missing}")
    frames = []
    for code, label in ((0, "incongruent"), (1, "congruent")):
        observed = design.loc[design["congruency_code"].eq(code)]
        if observed.empty:
            raise ValueError(f"drift-grid design has no {label} rows")
        cells = (
            observed.groupby(["rel_reward_z", "rel_aversive_z"], as_index=False)
            .size()
            .rename(columns={"size": "design_rows"})
        )
        cells["congruency_code"] = code
        cells["congruency"] = label
        cells["v"] = coefficients.drift(
            cells["rel_reward_z"],
            cells["rel_aversive_z"],
            code,
        )
        frames.append(cells)
    return pd.concat(frames, ignore_index=True)
