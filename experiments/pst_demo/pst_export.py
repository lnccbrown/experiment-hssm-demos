# PST task adapter: jsPsych export rows → PST response table (same columns as simulated players).
# Every choice is re-scored with schemas.tasks.pst.score_choice and checked against the browser.

from __future__ import annotations

from typing import Any

import pandas as pd

from runtime.jspsych_export import jspsych_rows_to_dataframe
from schemas.tasks.pst import (
    PST_MIN_RT_S,
    PST_RESPONSE_COLUMNS,
    PST_RESPONSE_DEADLINE_S,
    PSTSchedule,
    PSTTrialSpec,
    score_choice,
)

PST_TASK = "pst"


def is_pst_choice_row(row: dict[str, Any]) -> bool:
    return row.get("task") == PST_TASK


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _validate_against_schedule(trial: PSTTrialSpec, schedule: PSTSchedule) -> None:
    """Reject a row whose embedded schedule fields do not match the Python schedule."""
    phase_trials = {"practice": schedule.practice, "learning": schedule.learning, "test": schedule.test}
    if trial.phase not in phase_trials or not 1 <= trial.trial <= len(phase_trials[trial.phase]):
        raise ValueError(f"unknown {trial.phase!r} trial {trial.trial} for this game")
    expected = phase_trials[trial.phase][trial.trial - 1]
    if trial != expected:
        raise ValueError(f"{trial.phase} trial {trial.trial}: exported schedule fields do not match this game")


def pst_row_to_record(
    row: dict[str, Any], *, participant_id: int = 0, schedule: PSTSchedule | None = None
) -> dict[str, object]:
    """Map one jsPsych choice row to the PST response table (rt in seconds)."""
    trial = PSTTrialSpec(
        phase=str(row["phase"]),
        block=int(row["block"]),
        trial=int(row["trial"]),
        pair=str(row["pair"]),
        pair_id=int(row["pair_id"]),
        left_symbol=int(row["left_symbol"]),
        right_symbol=int(row["right_symbol"]),
        better_symbol=int(row["better_symbol"]),
        reward_left=_optional_int(row.get("reward_left")),
        reward_right=_optional_int(row.get("reward_right")),
    )
    if schedule is not None:
        _validate_against_schedule(trial, schedule)
    side = row.get("choice_side")
    scored = score_choice(trial, side)
    raw_response = row.get("response")
    expected_response = None if side is None else f"arrow{side}"
    normalized_response = None if raw_response is None else str(raw_response).lower()
    if normalized_response != expected_response:
        raise ValueError(f"{trial.phase} trial {trial.trial}: response and choice_side disagree")
    if "timed_out" in row and bool(row["timed_out"]) != scored["timed_out"]:
        raise ValueError(f"{trial.phase} trial {trial.trial}: inconsistent timeout flag")

    response_method = row.get("response_method")
    allowed_methods = {None} if scored["timed_out"] else {"key", "click"}
    if response_method not in allowed_methods:
        raise ValueError(f"{trial.phase} trial {trial.trial}: invalid response method")
    rt_s = None if scored["timed_out"] else float(row["rt"]) / 1000.0
    if rt_s is not None and (not pd.notna(rt_s) or rt_s < 0 or rt_s > PST_RESPONSE_DEADLINE_S):
        raise ValueError(f"{trial.phase} trial {trial.trial}: response time is outside the task window")
    anticipated = bool(not scored["timed_out"] and rt_s is not None and rt_s < PST_MIN_RT_S)
    if "anticipated" in row and bool(row["anticipated"]) != anticipated:
        raise ValueError(f"{trial.phase} trial {trial.trial}: inconsistent anticipation flag")
    if anticipated:
        scored["feedback"] = None
    browser_feedback = _optional_int(row.get("feedback"))
    if scored["feedback"] != browser_feedback:
        raise ValueError(
            f"{trial.phase} trial {trial.trial}: browser showed feedback {browser_feedback}, "
            f"schedule says {scored['feedback']}"
        )
    return {
        "participant_id": participant_id,
        "phase": trial.phase,
        "block": trial.block,
        "trial": trial.trial,
        "pair": trial.pair,
        "pair_id": trial.pair_id,
        "left_symbol": trial.left_symbol,
        "right_symbol": trial.right_symbol,
        "better_symbol": trial.better_symbol,
        "worse_symbol": trial.worse_symbol,
        "choice_side": side,
        **scored,
        "rt": rt_s,
        "anticipated": anticipated,
        "response_method": response_method,
    }


def pst_trials_dataframe(
    rows: list[dict[str, Any]] | str,
    *,
    participant_id: int = 0,
    schedule: PSTSchedule | None = None,
) -> pd.DataFrame:
    """Build the PST response table from a jsPsych ``.json()`` export (all phases)."""
    frame = jspsych_rows_to_dataframe(
        rows,
        include_row=is_pst_choice_row,
        row_to_record=lambda flat: pst_row_to_record(flat, participant_id=participant_id, schedule=schedule),
        columns=list(PST_RESPONSE_COLUMNS),
    )
    if frame.empty:
        return frame

    if frame.duplicated(["phase", "trial"]).any():
        raise ValueError("browser export contains duplicate phase/trial rows")
    phase_order = frame["phase"].map({"practice": 0, "learning": 1, "test": 2})
    if phase_order.isna().any() or not phase_order.is_monotonic_increasing:
        raise ValueError("browser export phases are out of order")
    for phase, phase_rows in frame.groupby("phase", sort=False):
        expected = list(range(1, len(phase_rows) + 1))
        if phase_rows["trial"].tolist() != expected:
            raise ValueError(f"browser export has missing or out-of-order {phase} trials")
    return frame
