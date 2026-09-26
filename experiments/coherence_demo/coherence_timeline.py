# jsPsych timeline assembly for the motion-coherence demonstration (intro, countdown, trials, feedback).
# Builds an ``ExperimentGenerator`` from demo levels and calls ``constant_stimuli_afc_timeline``.
# Exports ``coherence_runner_config`` and helpers used by the marimo coherence demo app.

from __future__ import annotations

from collections.abc import Callable

from runtime.jspsych_runner import RunnerConfig
from schemas.experimentGenerator import ExperimentGenerator
from schemas.timelines.constant_stimuli_afc_timeline import constant_stimuli_afc_timeline
from schemas.trial_generator import FactorTrialGenerator

import motion_stimulus_plugin  # noqa: F401 — registers motion_rdk stimulus plugin

INTRO_TRIAL: dict[str, object] = {
    "type": "html-button-response",
    "stimulus": (
        '<div style="font-size:20px;line-height:1.6;text-align:center;">'
        "Press left arrow for motion to the left, and right arrow for motion to the right"
        "</div>"
    ),
    "choices": ["click here to begin."],
    "button_layout": "grid",
    "grid_rows": 1,
    "grid_columns": 1,
}

COUNTDOWN_SECONDS = 3

COUNTDOWN_TRIALS: list[dict[str, object]] = [
    {
        "type": "html-keyboard-response",
        "stimulus": (
            f'<div style="font-size:72px;font-weight:700;text-align:center;">{n}</div>'
        ),
        "choices": "NO_KEYS",
        "trial_duration": 1000,
        "data": {"task": "countdown"},
    }
    for n in range(COUNTDOWN_SECONDS, 0, -1)
]

FEEDBACK_TRIAL: dict[str, object] = {
    "type": "html-keyboard-response",
    "stimulus": (
        "function(){"
        "const j = window.__jsPsychInstance;"
        "const d = j.data.get().last(1).values()[0] || {};"
        "const ok = !!d.correct;"
        "const txt = ok ? 'Correct' : 'Incorrect';"
        "const color = ok ? '#15803d' : '#dc2626';"
        "return `<div style=\"font-size:30px;font-weight:700;color:${color};text-align:center;\">${txt}</div>`;"
        "}"
    ),
    "choices": "NO_KEYS",
    "trial_duration": 1000,
}


def coherence_runner_config(*, title: str = "jsPsych Demo", session_id: str | None = None) -> RunnerConfig:
    """Runner settings for the coherence demo iframe."""
    return RunnerConfig(
        title=title,
        plugins=("html-keyboard-response", "html-button-response"),
        extra_scripts=(
            "renderers/motion_coherence/motion_coherence.js",
            "demo_results_charts.js",
        ),
        extra_styles=("renderers/motion_coherence/motion_coherence.css",),
        input_arrow_keys=True,
        results_session_id=session_id,
        show_results_charts=True,
        results_task_filter="motion_coherence",
    )


def build_coherence_demo_levels(
    level_a: float,
    level_b: float,
    level_c: float,
    *,
    reps_per_level: int = 5,
) -> list[tuple[str, float]]:
    """Expand A/B/C coherence levels with ``reps_per_level`` trials each."""
    levels = [("A", float(level_a)), ("B", float(level_b)), ("C", float(level_c))]
    return [
        (label, level)
        for label, level in levels
        for _ in range(int(reps_per_level))
    ]


def build_coherence_timeline(
    demo_levels: list[tuple[str, float]],
    *,
    display_params: dict[str, object],
    make_motion_coherence_trials: Callable[..., FactorTrialGenerator],
    presentation_duration_ms: int | None = None,
) -> list[dict[str, object]]:
    """Build intro -> countdown -> motion trials (+ feedback) for the marimo demo."""
    experiment = ExperimentGenerator(
        experiment_params={"display_params": dict(display_params)},
    )
    for idx, (label, level) in enumerate(demo_levels, start=1):
        block = make_motion_coherence_trials(
            n_trials=1,
            coherence=level,
            presentation_duration_ms=presentation_duration_ms,
        )
        trial = block.all_trials()[0]
        trial["data"]["label"] = label
        trial["data"]["trial_num"] = idx
        experiment.add_trial_generator(FactorTrialGenerator([trial]))

    afc_trials = constant_stimuli_afc_timeline(
        experiment.all_trials(),  # type: ignore[arg-type]
        stimulus_plugin="motion_rdk",
    )
    timeline: list[dict[str, object]] = [INTRO_TRIAL, *COUNTDOWN_TRIALS]
    for trial in afc_trials:
        timeline.extend([trial, FEEDBACK_TRIAL])
    return timeline
