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
    block_accuracy,
    score_choice,
    should_stop_learning,
)

PST_TASK = "pst"


def is_pst_choice_row(row: dict[str, Any]) -> bool:
    return row.get("task") == PST_TASK


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    number = float(value)
    if not pd.notna(number) or not number.is_integer():
        raise ValueError(f"expected an integer or null, got {value!r}")
    return int(number)


def _validate_bool_field(row: dict[str, Any], field: str, expected: bool, *, label: str) -> None:
    if field not in row:
        return
    value = row[field]
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label}: inconsistent {field.replace('_', ' ')} flag")


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
    if "worse_symbol" in row and int(row["worse_symbol"]) != trial.worse_symbol:
        raise ValueError(f"{trial.phase} trial {trial.trial}: exported worse symbol is inconsistent")
    side = row.get("choice_side")
    scored = score_choice(trial, side)
    raw_response = row.get("response")
    expected_response = None if side is None else f"arrow{side}"
    normalized_response = None if raw_response is None else str(raw_response).lower()
    if normalized_response != expected_response:
        raise ValueError(f"{trial.phase} trial {trial.trial}: response and choice_side disagree")
    label = f"{trial.phase} trial {trial.trial}"
    _validate_bool_field(row, "timed_out", bool(scored["timed_out"]), label=label)

    response_method = row.get("response_method")
    allowed_methods = {None} if scored["timed_out"] else {"key", "click"}
    if response_method not in allowed_methods:
        raise ValueError(f"{trial.phase} trial {trial.trial}: invalid response method")
    if scored["timed_out"] and row.get("rt") is not None and pd.notna(row.get("rt")):
        raise ValueError(f"{label}: a timeout must not carry a response time")
    rt_s = None if scored["timed_out"] else float(row["rt"]) / 1000.0
    if rt_s is not None and (not pd.notna(rt_s) or rt_s < 0 or rt_s > PST_RESPONSE_DEADLINE_S):
        raise ValueError(f"{trial.phase} trial {trial.trial}: response time is outside the task window")
    anticipated = bool(not scored["timed_out"] and rt_s is not None and rt_s < PST_MIN_RT_S)
    _validate_bool_field(row, "anticipated", anticipated, label=label)
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


def _validate_complete_session(
    frame: pd.DataFrame,
    schedule: PSTSchedule,
    *,
    include_practice: bool,
    include_test: bool,
) -> None:
    """Validate the complete endpoint of one finished adaptive browser session."""
    expected_phase_counts = {
        "practice": len(schedule.practice) if include_practice else 0,
        "test": len(schedule.test) if include_test else 0,
    }
    for phase, expected in expected_phase_counts.items():
        observed = int(frame["phase"].eq(phase).sum())
        if observed != expected:
            raise ValueError(f"finished browser export has {observed} {phase} rows; expected {expected}")

    learning = frame[frame["phase"] == "learning"]
    block_length = schedule.config.block_length
    if learning.empty or len(learning) % block_length:
        raise ValueError("finished browser export ends partway through a learning block")
    n_blocks = len(learning) // block_length
    config = schedule.config
    if not config.min_blocks <= n_blocks <= config.max_blocks:
        raise ValueError(
            f"finished browser export has {n_blocks} learning blocks; "
            f"expected {config.min_blocks} to {config.max_blocks}"
        )

    # A block is run only if the previous completed block failed at least one criterion. If the
    # session stops before max_blocks, its last block must be the first eligible block to pass.
    for block in range(config.min_blocks, n_blocks):
        accuracy = block_accuracy(learning[learning["block"] == block])
        if should_stop_learning(block, accuracy, config):
            raise ValueError(f"finished browser export continued after meeting the criterion in block {block}")
    if n_blocks < config.max_blocks:
        accuracy = block_accuracy(learning[learning["block"] == n_blocks])
        if not should_stop_learning(n_blocks, accuracy, config):
            raise ValueError(f"finished browser export stopped before meeting the criterion in block {n_blocks}")


def pst_trials_dataframe(
    rows: list[dict[str, Any]] | str,
    *,
    participant_id: int = 0,
    schedule: PSTSchedule | None = None,
    require_complete: bool = False,
    include_practice: bool = True,
    include_test: bool = True,
) -> pd.DataFrame:
    """Build the PST response table from a jsPsych ``.json()`` export (all phases).

    Set ``require_complete`` for a finished browser run. This additionally proves that enabled
    phases are complete and that the adaptive learning phase stopped at a valid block boundary.
    """
    if require_complete and schedule is None:
        raise ValueError("require_complete needs the Python schedule used by the browser")
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
    if require_complete:
        _validate_complete_session(
            frame,
            schedule,
            include_practice=include_practice,
            include_test=include_test,
        )
    return frame
