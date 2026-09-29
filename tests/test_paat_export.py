from dataclasses import asdict

import pandas as pd
import pytest

from experiments.paat_demo.paat_export import paat_trials_dataframe
from schemas.tasks.paat import PAAT_RESPONSE_COLUMNS, make_schedule, score_choice


def _browser_row(spec, side, rt_ms=800):
    scored = score_choice(spec, side)
    return {
        **asdict(spec),
        "task": "paat",
        "rel_reward": spec.rel_reward,
        "rel_aversive": spec.rel_aversive,
        "safe_side": spec.safe_side,
        **scored,
        "rt": None if side is None else rt_ms,
        "response_method": None if side is None else "click",
    }


def _browser_rows(schedule):
    return [
        _browser_row(
            spec,
            None if index == 2 else (spec.risky_side if index % 2 == 0 else spec.safe_side),
            600 + index,
        )
        for index, spec in enumerate(schedule.trials)
    ]


def test_export_revalidates_choices_and_converts_rt():
    schedule = make_schedule(seed=1)
    rows = _browser_rows(schedule)
    rows[0]["rt"] = 812
    rows[1]["rt"] = 640
    df = paat_trials_dataframe([*rows, {"task": "paat_spin"}], schedule=schedule)

    assert list(df.columns) == list(PAAT_RESPONSE_COLUMNS)
    assert df["response"].tolist()[:2] == [1, -1]
    assert df["rt"].iloc[0] == pytest.approx(0.812)
    assert pd.isna(df["rt"].iloc[2])
    assert df["timed_out"].tolist()[:3] == [False, False, True]
    assert df["response_method"].tolist()[:3] == ["click", "click", None]


def test_timeout_uses_forced_aversive_wheel_even_if_reward_was_predrawn():
    schedule = make_schedule(seed=1)
    rows = _browser_rows(schedule)
    index = next(
        index
        for index, spec in enumerate(schedule.trials)
        if spec.selected_wheel == "reward"
    )
    rows[index] = _browser_row(schedule.trials[index], None)

    df = paat_trials_dataframe(rows, schedule=schedule)

    assert df.loc[index, "timed_out"]
    assert df.loc[index, "selected_wheel"] == "aversive"
    assert df.loc[index, "aversive_outcome"]


def test_export_rejects_a_browser_outcome_mismatch():
    schedule = make_schedule(seed=2)
    rows = _browser_rows(schedule)
    rows[0]["outcome"] = 1 - rows[0]["outcome"]
    with pytest.raises(ValueError, match="browser outcome"):
        paat_trials_dataframe(rows, schedule=schedule)


def test_export_is_bound_to_the_exact_python_schedule():
    expected = make_schedule(seed=3)
    other = make_schedule(seed=4)
    with pytest.raises(ValueError, match="browser .*expected"):
        paat_trials_dataframe(_browser_rows(other), schedule=expected)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_export_requires_each_scheduled_trial_exactly_once_and_in_order(mutation):
    schedule = make_schedule(seed=5)
    rows = _browser_rows(schedule)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="do not match the expected schedule"):
        paat_trials_dataframe(rows, schedule=schedule)


def test_export_rejects_tampered_predictors_and_invalid_rts():
    schedule = make_schedule(seed=6)
    rows = _browser_rows(schedule)
    rows[0]["rel_reward_z"] = 999.0
    with pytest.raises(ValueError, match="rel_reward_z"):
        paat_trials_dataframe(rows, schedule=schedule)

    rows = _browser_rows(schedule)
    rows[0]["rt"] = 6001
    with pytest.raises(ValueError, match="rt must be finite"):
        paat_trials_dataframe(rows, schedule=schedule)

    rows[0]["rt"] = float("inf")
    with pytest.raises(ValueError, match="rt must be finite"):
        paat_trials_dataframe(rows, schedule=schedule)


def test_empty_bridge_state_produces_an_empty_response_table():
    df = paat_trials_dataframe(
        "[]",
        schedule=make_schedule(seed=7),
        result_received=False,
    )
    assert df.empty
    assert list(df.columns) == list(PAAT_RESPONSE_COLUMNS)


def test_completed_empty_export_is_rejected():
    with pytest.raises(ValueError, match="completed browser export contains no PAAT choice rows"):
        paat_trials_dataframe("[]", schedule=make_schedule(seed=8))
