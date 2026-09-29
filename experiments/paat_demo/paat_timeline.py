"""jsPsych timeline for the Probabilistic Approach-Avoidance Task."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from renderers.paat_wheels import choice_screen_html, example_option_html
from runtime.jspsych_runner import RunnerConfig
from schemas.tasks.paat import PAATSchedule, PAATTrialSpec

PAAT_TASK = "paat"
PAAT_MESSAGE_TYPE = "paat-results"


@dataclass(frozen=True)
class PAATTiming:
    response_window_ms: int = 6000
    spin_ms: int = 800
    outcome_ms: int = 700
    inter_trial_ms: int = 150

    @classmethod
    def research(cls) -> "PAATTiming":
        return cls(response_window_ms=6000, spin_ms=4000, outcome_ms=2000, inter_trial_ms=250)


def paat_runner_config(
    *,
    title: str = "Probabilistic Approach-Avoidance Task",
    session_id: str | None = None,
) -> RunnerConfig:
    return RunnerConfig(
        title=title,
        plugins=("html-keyboard-response", "html-button-response"),
        extra_scripts=("renderers/paat_wheels/paat_session.js",),
        extra_styles=("renderers/paat_wheels/paat_wheels.css",),
        input_arrow_keys=True,
        focus_guard=True,
        results_message_type=PAAT_MESSAGE_TYPE,
        results_session_id=session_id,
        show_results_charts=True,
        results_task_filter=PAAT_TASK,
        results_view="PAATResults",
        load_vega=False,
    )


def _screen(html: str, button: str, *, task: str = "paat_screen") -> dict[str, object]:
    return {
        "type": "html-button-response",
        "stimulus": html,
        "choices": [button],
        "data": {"task": task},
    }


def _instructions(schedule: PAATSchedule, timing: PAATTiming) -> list[dict[str, object]]:
    profile = schedule.profile
    minutes = round(profile.n_trials * (timing.spin_ms + timing.outcome_ms + 1500) / 60_000)
    if profile.name == "conference":
        timing_note = "This short demonstration compresses the post-choice animation."
    else:
        timing_note = (
            "This count-matched synthetic profile uses the original four-second wheel spin and "
            "two-second outcome timing; it is not the released canonical trial sequence."
        )
    intro = (
        '<div class="paat-panel"><h2>Choose between uncertain outcomes</h2>'
        "<p>Each option contains two wheels. The <strong>green wheel</strong> is the chance of a reward; "
        "the <strong>red wheel</strong> is the chance of an aversive outcome.</p>"
        f"{example_option_html()}"
        "<p>Choose the pair you prefer. Then one of its two wheels is selected at random and spun.</p>"
        '<p class="paat-muted">The public demo uses a labeled placeholder instead of disturbing images.</p></div>'
    )
    rules = (
        '<div class="paat-panel"><h2>Ready?</h2>'
        f"<p>&bull; {profile.n_trials} choices in {profile.n_blocks} blocks"
        + (f" (about {minutes} minute{'s' if minutes != 1 else ''})" if minutes else "")
        + ".</p>"
        f"<p>&bull; Respond within {timing.response_window_ms // 1000} seconds using "
        "<kbd>&larr;</kbd>/<kbd>&rarr;</kbd> or by clicking an option.</p>"
        "<p>&bull; A missed response produces the aversive outcome, as in the original task.</p>"
        f'<p class="paat-muted">{timing_note}</p></div>'
    )
    return [_screen(intro, "Continue"), _screen(rules, "Start")]


def _trial_data(trial: PAATTrialSpec, timing: PAATTiming) -> dict[str, object]:
    data = asdict(trial)
    data.update(
        {
            "task": PAAT_TASK,
            "rel_reward": trial.rel_reward,
            "rel_aversive": trial.rel_aversive,
            "safe_side": trial.safe_side,
            "spin_duration_ms": timing.spin_ms,
        }
    )
    return data


def _choice_trial(trial: PAATTrialSpec, schedule: PAATSchedule, timing: PAATTiming) -> dict[str, object]:
    return {
        "type": "html-keyboard-response",
        "stimulus": choice_screen_html(
            trial,
            status_label=f"Block {trial.block} of {schedule.profile.n_blocks} · Choice {trial.trial} of {schedule.profile.n_trials}",
        ),
        "choices": ["ArrowLeft", "ArrowRight"],
        "response_ends_trial": True,
        "trial_duration": timing.response_window_ms,
        "data": _trial_data(trial, timing),
        "on_load": "function(){ window.PAATSession.updateReward(); window.PAATSession.armOptions(); }",
        "on_finish": "function(data){ window.PAATSession.scoreTrial(data); }",
    }


def _spin_trial(timing: PAATTiming) -> dict[str, object]:
    return {
        "type": "html-keyboard-response",
        "stimulus": "function(){ return window.PAATSession.spinHTML(); }",
        "choices": "NO_KEYS",
        "trial_duration": timing.spin_ms,
        "data": {"task": "paat_spin"},
    }


def _outcome_trial(timing: PAATTiming) -> dict[str, object]:
    return {
        "type": "html-keyboard-response",
        "stimulus": "function(){ return window.PAATSession.outcomeHTML(); }",
        "choices": "NO_KEYS",
        "trial_duration": timing.outcome_ms,
        "post_trial_gap": timing.inter_trial_ms,
        "data": {"task": "paat_outcome"},
    }


def _break_screen(block: int, n_blocks: int) -> dict[str, object]:
    return _screen(
        '<div class="paat-panel">'
        f"<h2>Block {block} of {n_blocks} complete</h2>"
        '<p>Take a short break. Continue when you are ready.</p></div>',
        "Continue",
        task="paat_break",
    )


def build_paat_timeline(
    schedule: PAATSchedule,
    *,
    timing: PAATTiming | None = None,
) -> list[dict[str, object]]:
    """Instructions, free-choice trials with anticipation/outcomes, and breaks."""
    timing = timing or (PAATTiming() if schedule.profile.name == "conference" else PAATTiming.research())
    timeline = _instructions(schedule, timing)
    for trial in schedule.trials:
        timeline.extend([_choice_trial(trial, schedule, timing), _spin_trial(timing), _outcome_trial(timing)])
        if trial.trial % schedule.profile.trials_per_block == 0 and trial.block < schedule.profile.n_blocks:
            timeline.append(_break_screen(trial.block, schedule.profile.n_blocks))
    return timeline
