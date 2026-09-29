import json
import shutil
import subprocess

import pytest

from experiments.pst_demo.pst_timeline import PSTTiming, build_pst_timeline, pst_runner_config
from runtime.jspsych_runner import build_jspsych_runner_html
from schemas.tasks.pst import PSTConfig, make_schedule

CALLBACK_KEYS = ("stimulus", "on_finish", "on_load", "conditional_function")


def _nodes(timeline):
    for node in timeline:
        yield node
        yield from _nodes(node.get("timeline", []))


def _choices(timeline, phase):
    return [n for n in _nodes(timeline) if n.get("data", {}).get("task") == "pst" and n["data"]["phase"] == phase]


def test_timeline_contains_every_scheduled_trial():
    schedule = make_schedule(seed=1)
    config = schedule.config
    timeline = build_pst_timeline(schedule)

    assert len(_choices(timeline, "practice")) == config.practice_trials
    assert len(_choices(timeline, "learning")) == config.max_blocks * config.block_length
    assert len(_choices(timeline, "test")) == len(schedule.test)
    feedback = [n for n in _nodes(timeline) if n.get("data", {}).get("task") == "pst_feedback"]
    assert len(feedback) == config.practice_trials + config.max_blocks * config.block_length


def test_blocks_after_the_minimum_are_conditional():
    timeline = build_pst_timeline(make_schedule(PSTConfig(min_blocks=2, max_blocks=4), seed=2))
    blocks = [n for n in timeline if "timeline" in n]

    assert ["conditional_function" in b for b in blocks] == [False, False, True, True]
    assert "shouldRunBlock(2, " in blocks[2]["conditional_function"]
    assert '"EF": 0.5' in blocks[3]["conditional_function"]
    assert all(b["timeline"][-1]["data"]["task"] == "pst_break" for b in blocks)


def test_choice_trial_carries_schedule_and_renders_glyphs():
    schedule = make_schedule(PSTConfig(symbol_set="shapes"), seed=3)
    trial = _choices(build_pst_timeline(schedule), "learning")[0]
    spec = schedule.learning[0]

    assert trial["type"] == "html-keyboard-response"
    assert trial["choices"] == ["ArrowLeft", "ArrowRight"]
    assert trial["trial_duration"] == PSTTiming().response_window_ms
    assert trial["data"]["reward_left"] == spec.reward_left
    assert trial["data"]["minimum_rt_ms"] == 200
    assert trial["data"]["better_symbol"] == spec.better_symbol
    assert trial["data"]["correct_key"] == ("ArrowLeft" if spec.left_symbol == spec.better_symbol else "ArrowRight")
    assert f'aria-label="{schedule.glyphs[spec.left_symbol]}"' in trial["stimulus"]
    assert 'data-side="left"' in trial["stimulus"] and 'data-side="right"' in trial["stimulus"]
    assert "PSTSession.scoreTrial" in trial["on_finish"]
    assert "PSTSession.armCards" in trial["on_load"]


def test_first_screen_answers_to_the_arrow_keys_and_clicks():
    first = build_pst_timeline(make_schedule(seed=6))[0]

    assert first["type"] == "html-keyboard-response"
    assert first["choices"] == ["ArrowLeft", "ArrowRight"]
    assert "PSTSession.armCards" in first["on_load"]
    assert first["stimulus"].count("data-side=") == 2


def test_runner_html_loads_pst_assets_without_vega():
    config = pst_runner_config(session_id="test-session")
    html = build_jspsych_runner_html(build_pst_timeline(make_schedule(seed=4)), config=config)
    assert "function scoreTrial" in html and ".pst-card" in html
    assert config.focus_guard and ".runner-focus-guard" in html
    assert "vega" not in html
    assert "test-session" not in html  # runner config is base64 encoded, not interpolated into scripts


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_callback_strings_compile_as_javascript():
    timeline = build_pst_timeline(make_schedule(seed=5))
    sources = {
        node[key]
        for node in _nodes(timeline)
        for key in CALLBACK_KEYS
        if isinstance(node.get(key), str) and node[key].lstrip().startswith("function")
    }
    script = "".join(f"new Function('return (' + {json.dumps(src)} + ')');\n" for src in sorted(sources))
    subprocess.run(["node", "-e", script], check=True)
