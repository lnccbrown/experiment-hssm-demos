"""Validate browser PAAT rows and convert them to the canonical response table."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from runtime.jspsych_export import flatten_jspsych_row, parse_rows_json
from schemas.tasks.paat import (
    PAAT_RESPONSE_COLUMNS,
    PAATSchedule,
    PAATTrialSpec,
    response_record,
    score_choice,
)

PAAT_TASK = "paat"


def is_paat_choice_row(row: dict[str, Any]) -> bool:
    return row.get("task") == PAAT_TASK


EXPECTED_TRIAL_FIELDS = (
    "profile",
    "block",
    "trial",
    "congruency",
    "congruency_code",
    "p_reward_risky",
    "p_reward_safe",
    "p_aversive_risky",
    "p_aversive_safe",
    "rel_reward_z",
    "rel_aversive_z",
    "risky_side",
    "spin_risky",
    "spin_safe",
    "reward_amount",
    "reward_unit",
)


def _same(expected: object, actual: object) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        try:
            return actual is not None and math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _validate_trial_identity(row: dict[str, Any], trial: PAATTrialSpec) -> None:
    for field in EXPECTED_TRIAL_FIELDS:
        expected = getattr(trial, field)
        if not _same(expected, row.get(field)):
            raise ValueError(
                f"trial {trial.trial}: browser {field}={row.get(field)!r}, expected {expected!r}"
            )
    for field, expected in (
        ("safe_side", trial.safe_side),
        ("rel_reward", trial.rel_reward),
        ("rel_aversive", trial.rel_aversive),
    ):
        if not _same(expected, row.get(field)):
            raise ValueError(
                f"trial {trial.trial}: browser {field}={row.get(field)!r}, expected {expected!r}"
            )


def paat_row_to_record(
    row: dict[str, Any],
    *,
    expected_trial: PAATTrialSpec,
    participant_id: int = 0,
    response_window_ms: int = 6000,
) -> dict[str, object]:
    trial = expected_trial
    _validate_trial_identity(row, trial)
    side = row.get("choice_side")
    expected = score_choice(trial, side)
    for field in (
        "chosen_option",
        "response",
        "timed_out",
        "selected_wheel",
        "spin_value",
        "outcome",
        "reward_value",
        "aversive_outcome",
    ):
        if not _same(expected[field], row.get(field)):
            raise ValueError(
                f"trial {trial.trial}: browser {field}={row.get(field)!r}, expected {expected[field]!r}"
            )
    response_method = row.get("response_method")
    if expected["timed_out"]:
        if row.get("rt") is not None or response_method is not None:
            raise ValueError(f"trial {trial.trial}: timeout must have null rt and response_method")
        rt = None
    else:
        try:
            rt_ms = float(row["rt"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"trial {trial.trial}: answered trial has invalid rt") from error
        if not math.isfinite(rt_ms) or not 0 <= rt_ms <= response_window_ms:
            raise ValueError(
                f"trial {trial.trial}: rt must be finite and between 0 and {response_window_ms} ms"
            )
        if response_method not in ("click", "key"):
            raise ValueError(f"trial {trial.trial}: invalid response_method {response_method!r}")
        rt = rt_ms / 1000.0
    return response_record(
        trial,
        participant_id=participant_id,
        choice_side=side,
        rt=rt,
        response_method=response_method,
    )


def paat_trials_dataframe(
    rows: list[dict[str, Any]] | str,
    *,
    schedule: PAATSchedule,
    participant_id: int = 0,
    response_window_ms: int = 6000,
    result_received: bool = True,
) -> pd.DataFrame:
    """Validate a complete browser export against its exact Python schedule."""
    choice_rows = [
        flat
        for row in parse_rows_json(rows)
        if is_paat_choice_row(flat := flatten_jspsych_row(row))
    ]
    if not choice_rows:
        if result_received:
            raise ValueError("completed browser export contains no PAAT choice rows")
        return pd.DataFrame(columns=list(PAAT_RESPONSE_COLUMNS))

    expected_ids = [trial.trial for trial in schedule.trials]
    try:
        actual_ids = [int(row["trial"]) for row in choice_rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("PAAT choice rows must contain integer trial IDs") from error
    if actual_ids != expected_ids:
        raise ValueError(
            "browser PAAT trials do not match the expected schedule: "
            f"expected {expected_ids}, found {actual_ids}"
        )

    records = [
        paat_row_to_record(
            row,
            expected_trial=trial,
            participant_id=participant_id,
            response_window_ms=response_window_ms,
        )
        for row, trial in zip(choice_rows, schedule.trials, strict=True)
    ]
    return pd.DataFrame(records, columns=list(PAAT_RESPONSE_COLUMNS))
