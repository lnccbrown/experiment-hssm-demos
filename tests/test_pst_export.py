import json

import pandas as pd
import pytest

from experiments.pst_demo.pst_export import pst_trials_dataframe
from schemas.tasks.pst import PST_RESPONSE_COLUMNS, make_schedule, pst_config_for_profile


def _browser_rows(schedule, sides, rts_ms, phase="learning"):
    """Rows as jsPsych + pst_session.js export them, plus a non-choice row that must be ignored."""
    rows = []
    for spec, side, rt in zip(getattr(schedule, phase), sides, rts_ms):
        reward = None if side is None else (spec.reward_left if side == "left" else spec.reward_right)
        rows.append({
            "task": "pst", "phase": spec.phase, "block": spec.block, "trial": spec.trial,
            "pair": spec.pair, "pair_id": spec.pair_id,
            "left_symbol": spec.left_symbol, "right_symbol": spec.right_symbol,
            "better_symbol": spec.better_symbol, "worse_symbol": spec.worse_symbol,
            "reward_left": spec.reward_left, "reward_right": spec.reward_right,
            "response": None if side is None else f"arrow{side}", "choice_side": side,
            "feedback": reward, "rt": rt, "response_method": None if side is None else "key",
        })
    return [*rows, {"task": "pst_feedback", "rt": 1000}]


def test_export_rescores_choices_and_converts_rt():
    schedule = make_schedule(seed=1)
    df = pst_trials_dataframe(json.dumps(_browser_rows(schedule, ["left", "right", None], [812, 640, None])))
    first = schedule.learning[0]

    assert list(df.columns) == list(PST_RESPONSE_COLUMNS) and len(df) == 3
    assert df["rt"].iloc[0] == pytest.approx(0.812)
    assert df["chosen_symbol"].iloc[0] == first.left_symbol
    assert df["response"].iloc[0] == (1 if first.left_symbol == first.better_symbol else -1)
    assert df["timed_out"].tolist() == [False, False, True]
    assert pd.isna(df["rt"].iloc[2]) and pd.isna(df["feedback"].iloc[2])


def test_export_retains_response_method_and_marks_feedback_free_anticipations():
    schedule = make_schedule(seed=11)
    rows = _browser_rows(schedule, ["left", "right"], [150, 650])
    rows[0]["feedback"] = None
    rows[0]["anticipated"] = True
    rows[0]["response_method"] = "key"
    rows[1]["response_method"] = "click"
    df = pst_trials_dataframe(rows, schedule=schedule)
    assert df["anticipated"].tolist() == [True, False]
    assert pd.isna(df.loc[0, "feedback"])
    assert df["response_method"].tolist() == ["key", "click"]


def test_export_rejects_feedback_that_disagrees_with_the_schedule():
    rows = _browser_rows(make_schedule(seed=2), ["left"], [700])
    rows[0]["feedback"] = 1 - rows[0]["feedback"]
    with pytest.raises(ValueError, match="browser showed feedback"):
        pst_trials_dataframe(rows)


def test_export_rejects_schedule_fields_from_a_different_game():
    schedule = make_schedule(seed=20)
    rows = _browser_rows(schedule, ["left"], [700])
    rows[0]["left_symbol"], rows[0]["right_symbol"] = rows[0]["right_symbol"], rows[0]["left_symbol"]
    with pytest.raises(ValueError, match="do not match this game"):
        pst_trials_dataframe(rows, schedule=schedule)


def test_export_rejects_inconsistent_response_and_method_fields():
    schedule = make_schedule(seed=25)
    rows = _browser_rows(schedule, ["left"], [700])
    rows[0]["response"] = "arrowright"
    with pytest.raises(ValueError, match="disagree"):
        pst_trials_dataframe(rows, schedule=schedule)

    rows = _browser_rows(schedule, ["left"], [700])
    rows[0]["response_method"] = "voice"
    with pytest.raises(ValueError, match="response method"):
        pst_trials_dataframe(rows, schedule=schedule)

    rows = _browser_rows(schedule, ["left"], [700])
    rows[0]["anticipated"] = 0
    with pytest.raises(ValueError, match="anticipated flag"):
        pst_trials_dataframe(rows, schedule=schedule)

    rows = _browser_rows(schedule, ["left"], [700])
    rows[0]["worse_symbol"] = rows[0]["better_symbol"]
    with pytest.raises(ValueError, match="worse symbol"):
        pst_trials_dataframe(rows, schedule=schedule)

    rows = _browser_rows(schedule, [None], [None])
    rows[0]["rt"] = 4000
    with pytest.raises(ValueError, match="timeout must not carry"):
        pst_trials_dataframe(rows, schedule=schedule)


def test_export_rejects_duplicate_missing_reordered_and_over_deadline_rows():
    schedule = make_schedule(seed=21)
    rows = _browser_rows(schedule, ["left", "right", "left"], [700, 800, 900])

    with pytest.raises(ValueError, match="duplicate"):
        pst_trials_dataframe([rows[0], rows[0]], schedule=schedule)
    with pytest.raises(ValueError, match="missing or out-of-order"):
        pst_trials_dataframe([rows[0], rows[2]], schedule=schedule)
    with pytest.raises(ValueError, match="out-of-order"):
        pst_trials_dataframe([rows[1], rows[0]], schedule=schedule)

    rows[0]["rt"] = 4001
    with pytest.raises(ValueError, match="outside the task window"):
        pst_trials_dataframe([rows[0]], schedule=schedule)


def test_complete_export_validates_enabled_phases_and_adaptive_endpoint():
    schedule = make_schedule(pst_config_for_profile("quick"), seed=26)
    first_block = schedule.learning[: schedule.config.block_length]
    better_sides = ["left" if trial.left_symbol == trial.better_symbol else "right" for trial in first_block]
    first_rows = _browser_rows(schedule, better_sides, [700] * len(first_block))[:-1]

    complete = pst_trials_dataframe(
        first_rows,
        schedule=schedule,
        require_complete=True,
        include_practice=False,
        include_test=False,
    )
    assert len(complete) == schedule.config.block_length

    both_blocks = schedule.learning
    both_sides = ["left" if trial.left_symbol == trial.better_symbol else "right" for trial in both_blocks]
    with pytest.raises(ValueError, match="continued after meeting"):
        pst_trials_dataframe(
            _browser_rows(schedule, both_sides, [700] * len(both_blocks)),
            schedule=schedule,
            require_complete=True,
            include_practice=False,
            include_test=False,
        )

    worse_sides = ["right" if side == "left" else "left" for side in better_sides]
    with pytest.raises(ValueError, match="stopped before meeting"):
        pst_trials_dataframe(
            _browser_rows(schedule, worse_sides, [700] * len(first_block)),
            schedule=schedule,
            require_complete=True,
            include_practice=False,
            include_test=False,
        )

    with pytest.raises(ValueError, match="partway through"):
        pst_trials_dataframe(
            first_rows[:-1],
            schedule=schedule,
            require_complete=True,
            include_practice=False,
            include_test=False,
        )

    with pytest.raises(ValueError, match="practice rows"):
        pst_trials_dataframe(
            first_rows,
            schedule=schedule,
            require_complete=True,
            include_practice=True,
            include_test=False,
        )


def test_test_phase_rows_have_no_feedback():
    df = pst_trials_dataframe(_browser_rows(make_schedule(seed=3), ["left", "right"], [900, 950], phase="test"))
    assert (df["phase"] == "test").all() and df["feedback"].isna().all()
