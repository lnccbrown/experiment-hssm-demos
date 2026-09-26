import json

import pandas as pd
import pytest

from experiments.pst_demo.pst_export import pst_trials_dataframe
from schemas.tasks.pst import PST_RESPONSE_COLUMNS, make_schedule


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


def test_test_phase_rows_have_no_feedback():
    df = pst_trials_dataframe(_browser_rows(make_schedule(seed=3), ["left", "right"], [900, 950], phase="test"))
    assert (df["phase"] == "test").all() and df["feedback"].isna().all()
