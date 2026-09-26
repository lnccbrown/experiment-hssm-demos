# jsPsych timeline for the PST: instructions, practice, adaptive learning blocks, breaks, test phase.
# Turns a PSTSchedule into Trial contracts rendered by the generic n-AFC builder with ``pst_cards``.
# Exports ``build_pst_timeline`` and ``pst_runner_config`` for the marimo PST app.

from __future__ import annotations

import json
from dataclasses import dataclass

from experiments.pst_demo.pst_stimulus_plugin import ARM_CARDS  # importing registers the pst_cards plugin
from renderers.pst_symbols.pst_symbols import example_cards_html
from runtime.jspsych_runner import RunnerConfig
from schemas.contracts import Trial
from schemas.tasks.pst import PSTConfig, PSTSchedule, PSTTrialSpec, PST_MIN_RT_S, PST_RESPONSE_DEADLINE_S
from schemas.timelines.constant_stimuli_afc_timeline import constant_stimuli_afc_timeline
from schemas.trial_generator import FactorTrialGenerator

PST_TASK = "pst"
PST_MESSAGE_TYPE = "pst-results"
_SECONDS_PER_LEARNING_TRIAL = 2.4  # typical choice + feedback + gap, for the duration estimate


@dataclass(frozen=True)
class PSTTiming:
    """Presentation timing in milliseconds."""

    response_window_ms: int = int(PST_RESPONSE_DEADLINE_S * 1000)
    feedback_ms: int = 1000
    inter_trial_ms: int = 300
    test_inter_trial_ms: int = 500


def pst_runner_config(*, title: str = "Probabilistic Selection Task", session_id: str | None = None) -> RunnerConfig:
    """Runner settings for the PST iframe (arrow keys, focus guard, PST scripts/styles, PST end screen)."""
    return RunnerConfig(
        title=title,
        plugins=("html-keyboard-response", "html-button-response"),
        extra_scripts=("renderers/pst_symbols/pst_session.js",),
        extra_styles=("renderers/pst_symbols/pst_symbols.css",),
        input_arrow_keys=True,
        focus_guard=True,
        results_message_type=PST_MESSAGE_TYPE,
        results_session_id=session_id,
        show_results_charts=True,
        results_task_filter=PST_TASK,
        results_view="PSTResults",
        load_vega=False,
    )


def _pair_label(schedule: PSTSchedule, trial: PSTTrialSpec) -> str:
    config = schedule.config
    pairs = [*config.pairs, config.practice_pair]
    for pair in pairs:
        if pair.name == trial.pair:
            return f"{round(100 * pair.p_better)}/{round(100 * pair.p_worse)} pair"
    return trial.pair


def pst_trial_contract(trial: PSTTrialSpec, schedule: PSTSchedule, *, status_label: str, timing: PSTTiming) -> Trial:
    """One scheduled PST trial as a generic Trial (correct = the better symbol's side)."""
    correct_index = 0 if trial.left_symbol == trial.better_symbol else 1
    return FactorTrialGenerator.generate_trials(
        task=PST_TASK,
        stimulus_factors={
            "pair": trial.pair,
            "left_symbol": trial.left_symbol,
            "right_symbol": trial.right_symbol,
            "better_symbol": trial.better_symbol,
        },
        display_params={
            "left_glyph": schedule.glyphs[trial.left_symbol],
            "right_glyph": schedule.glyphs[trial.right_symbol],
            "symbol_set": schedule.config.symbol_set,
            "status_label": status_label,
        },
        presentation_duration_ms=timing.response_window_ms,
        correct_index=correct_index,
        choices=["ArrowLeft", "ArrowRight"],
        correct_key=("ArrowLeft", "ArrowRight")[correct_index],
        data={
            "task": PST_TASK,
            "phase": trial.phase,
            "block": trial.block,
            "trial": trial.trial,
            "pair": trial.pair,
            "pair_id": trial.pair_id,
            "pair_label": _pair_label(schedule, trial),
            "left_symbol": trial.left_symbol,
            "right_symbol": trial.right_symbol,
            "better_symbol": trial.better_symbol,
            "worse_symbol": trial.worse_symbol,
            "reward_left": trial.reward_left,
            "reward_right": trial.reward_right,
            "minimum_rt_ms": int(PST_MIN_RT_S * 1000),
        },
    )


def _choice_trials(schedule: PSTSchedule, trials, status_label: str, timing: PSTTiming) -> list[dict[str, object]]:
    contracts = [pst_trial_contract(t, schedule, status_label=status_label, timing=timing) for t in trials]
    return constant_stimuli_afc_timeline(contracts, stimulus_plugin="pst_cards")


def _feedback_trial(timing: PSTTiming) -> dict[str, object]:
    return {
        "type": "html-keyboard-response",
        "stimulus": "function(){ return window.PSTSession.feedbackHTML(); }",
        "choices": "NO_KEYS",
        "trial_duration": timing.feedback_ms,
        "post_trial_gap": timing.inter_trial_ms,
        "data": {"task": "pst_feedback"},
    }


def _with_feedback(choices: list[dict[str, object]], timing: PSTTiming) -> list[dict[str, object]]:
    sequence: list[dict[str, object]] = []
    for choice in choices:
        sequence.extend([choice, _feedback_trial(timing)])
    return sequence


def _screen(html: str, button: str, task: str = "pst_screen") -> dict[str, object]:
    return {"type": "html-button-response", "stimulus": html, "choices": [button], "data": {"task": task}}


def _duration_text(config: PSTConfig) -> str:
    low, high = (round(n * config.block_length * _SECONDS_PER_LEARNING_TRIAL / 60) for n in (config.min_blocks, config.max_blocks))
    return f"about {low} minutes" if low == high else f"{low} to {high} minutes"


def _instructions(schedule: PSTSchedule, timing: PSTTiming, include_practice: bool) -> list[dict[str, object]]:
    config = schedule.config
    practice_glyphs = schedule.glyphs[config.n_symbols : config.n_symbols + 2]
    blocks = (
        f"{config.max_blocks} blocks"
        if config.min_blocks == config.max_blocks
        else f"{config.min_blocks} to {config.max_blocks} blocks; learn fast and you finish early"
    )
    intro = (
        '<div class="pst-panel"><h2>Find the lucky symbols</h2>'
        "<p>You will see two symbols at a time. Pick the left one with <kbd>&larr;</kbd>, the right one "
        "with <kbd>&rarr;</kbd>, or click the one you want.</p>"
        f"{example_cards_html(practice_glyphs, config.symbol_set, clickable=True)}"
        "<p>Some symbols are correct more often than others, but <strong>no symbol is always "
        "correct</strong>. Work out which ones are luckier and earn as many points as you can.</p>"
        '<p class="pst-try">Try it now: press <kbd>&larr;</kbd> or <kbd>&rarr;</kbd>, or click a symbol.</p></div>'
    )
    rules = (
        '<div class="pst-panel"><h2>Before you start</h2>'
        f"<p>&bull; There are {len(config.pairs)} pairs of symbols to learn, mixed together.</p>"
        f"<p>&bull; Answer within {timing.response_window_ms // 1000} seconds. Go with your gut.</p>"
        f"<p>&bull; The game has {blocks} of {config.block_length} choices ({_duration_text(config)}).</p>"
        "<p>&bull; Stay on this tab until the end.</p>"
        + ('<p class="pst-muted">First, a short practice with two symbols you will not see again.</p>' if include_practice else "")
        + "</div>"
    )
    try_it = {
        "type": "html-keyboard-response",
        "stimulus": intro,
        "choices": ["ArrowLeft", "ArrowRight"],
        "on_load": ARM_CARDS,
        "data": {"task": "pst_screen"},
    }
    return [try_it, _screen(rules, "Start practice" if include_practice else "Start the game")]


def _break_screen(block: int) -> dict[str, object]:
    return {
        "type": "html-button-response",
        "stimulus": f"function(){{ return window.PSTSession.blockSummaryHTML({block}); }}",
        "choices": ["Continue"],
        "data": {"task": "pst_break", "block": block},
    }


def build_pst_timeline(
    schedule: PSTSchedule,
    *,
    timing: PSTTiming | None = None,
    include_practice: bool = True,
    include_test: bool = True,
) -> list[dict[str, object]]:
    """Instructions → practice → learning blocks (adaptive length) → test phase."""
    timing = timing or PSTTiming()
    config = schedule.config
    include_practice = include_practice and bool(schedule.practice)
    timeline: list[dict[str, object]] = _instructions(schedule, timing, include_practice)

    if include_practice:
        timeline += _with_feedback(_choice_trials(schedule, schedule.practice, "Practice", timing), timing)
        timeline.append(
            _screen(
                '<div class="pst-panel"><h2>Practice done</h2><p>Now the real game begins, with new '
                "symbols. Every <strong>Correct!</strong> earns a point.</p></div>",
                "Start the game",
            )
        )

    criterion = json.dumps(dict(config.criterion))
    of_blocks = f"of {config.max_blocks}" if config.min_blocks == config.max_blocks else f"of up to {config.max_blocks}"
    for block in range(1, config.max_blocks + 1):
        trials = [t for t in schedule.learning if t.block == block]
        choices = _choice_trials(schedule, trials, f"Block {block} {of_blocks}", timing)
        node: dict[str, object] = {"timeline": [*_with_feedback(choices, timing), _break_screen(block)]}
        if block > config.min_blocks:
            node["conditional_function"] = (
                f"function(){{ return window.PSTSession.shouldRunBlock({block - 1}, {criterion}); }}"
            )
        timeline.append(node)

    if include_test and schedule.test:
        timeline.append(
            _screen(
                '<div class="pst-panel"><h2>Final round</h2><p>Now you will see new combinations of the '
                "symbols you learned, <strong>without feedback</strong>. Each time, pick the symbol you "
                "think is more likely to be correct. If you are unsure, go with your gut.</p></div>",
                "Start the final round",
            )
        )
        for choice in _choice_trials(schedule, schedule.test, "Final round", timing):
            timeline.append({**choice, "post_trial_gap": timing.test_inter_trial_ms})

    return timeline
