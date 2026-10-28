import base64
import json
import re
import shutil
import subprocess

import pytest

from experiments.paat_demo.paat_timeline import PAATTiming, build_paat_timeline, paat_runner_config
from runtime.jspsych_export import _bridge_esm
from runtime.jspsych_runner import build_jspsych_runner_html
from schemas.tasks.paat import make_schedule


def _task_nodes(timeline, task):
    return [node for node in timeline if node.get("data", {}).get("task") == task]


def test_timeline_has_choice_spin_and_outcome_for_every_trial():
    schedule = make_schedule("conference", seed=1)
    timeline = build_paat_timeline(schedule)

    assert len(_task_nodes(timeline, "paat")) == 24
    assert len(_task_nodes(timeline, "paat_spin")) == 24
    assert len(_task_nodes(timeline, "paat_outcome")) == 24
    assert len(_task_nodes(timeline, "paat_break")) == 1


def test_choice_rows_carry_the_full_design_and_clickable_options():
    schedule = make_schedule("conference", seed=2)
    first = _task_nodes(build_paat_timeline(schedule), "paat")[0]
    spec = schedule.trials[0]

    assert first["choices"] == ["ArrowLeft", "ArrowRight"]
    assert first["trial_duration"] == 6000
    assert first["data"]["p_reward_risky"] == spec.p_reward_risky
    assert first["data"]["rel_reward_z"] == spec.rel_reward_z
    assert first["stimulus"].count('data-side="') == 2
    assert "PAATSession.armOptions" in first["on_load"]
    assert "PAATSession.scoreTrial" in first["on_finish"]


def test_research_profile_uses_original_spin_and_outcome_timing():
    schedule = make_schedule("published_study2", seed=3)
    timing = PAATTiming.research()
    timeline = build_paat_timeline(schedule, timing=timing)
    assert _task_nodes(timeline, "paat_spin")[0]["trial_duration"] == 4000
    assert _task_nodes(timeline, "paat_outcome")[0]["trial_duration"] == 2000


def test_runner_inlines_paat_assets_without_vega():
    html = build_jspsych_runner_html(
        build_paat_timeline(make_schedule(seed=4)),
        config=paat_runner_config(session_id="session-123"),
    )
    assert "function scoreTrial" in html
    assert ".paat-wheel" in html
    assert "PAATResults" in html
    assert "vega" not in html
    assert html.count('rel="preload" as="script"') == 3
    assert html.count('<script defer crossorigin="anonymous" src=') == 3
    assert "__SCRIPT_PRELOAD_TAGS__" not in html
    assert 'rel="preconnect" href="https://cdn.jsdelivr.net"' in html
    assert 'class="runner-loading"' in html
    assert 'document.addEventListener("DOMContentLoaded", boot' in html
    config_b64 = re.search(r'decodeB64Json\("([^"]+)"\)', html).group(1)
    config = json.loads(base64.b64decode(config_b64))
    assert config["results_session_id"] == "session-123"


def test_result_bridge_filters_for_the_current_session_nonce():
    esm = _bridge_esm(
        message_type="paat-results",
        session_id="session-123",
        iframe_id="paat-game-session-123",
    )
    assert 'payload.type !== "paat-results"' in esm
    assert 'payload.session_id !== "session-123"' in esm
    assert 'document.getElementById("paat-game-session-123")' in esm
    assert "event.source !== frame.contentWindow" in esm
    assert 'model.set("result_received", true)' in esm


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_callback_strings_compile_as_javascript():
    timeline = build_paat_timeline(make_schedule(seed=5))
    sources = {
        node[key]
        for node in timeline
        for key in ("stimulus", "on_load", "on_finish")
        if isinstance(node.get(key), str) and node[key].lstrip().startswith("function")
    }
    script = "".join(f"new Function('return (' + {json.dumps(source)} + ')');\n" for source in sorted(sources))
    subprocess.run(["node", "-e", script], check=True)
