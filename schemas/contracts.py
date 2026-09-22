# Shared typed contracts for trials, experiments, and jsPsych result payloads.
# Separates stimulus semantics (``stimulus_factors``) from presentation-only ``display_params``.
# Imported by trial generators, timelines, actors, and analysis code.

from __future__ import annotations

from typing import TypedDict


class ExperimentParams(TypedDict, total=False):
    """Experiment-wide settings applied when serving trials from an ``ExperimentGenerator``.

    ``display_params`` are merged into each trial (trial keys override experiment defaults).
    ``data_output_path`` is attached to each trial's ``data`` dict when set.
    """

    display_params: dict[str, object]
    data_output_path: str


class Trial(TypedDict):
    """Task-agnostic trial contract for human and simulated runs.

    ``stimulus_factors`` holds experimental factors that control the stimulus.
    ``display_params`` holds rendering/presentation parameters.
    Latent ``stim_strengths`` for the actor are derived from
    ``stimulus_factors`` inside the actor, not stored on the trial.
    ``presentation_duration_ms`` is ``None`` for unlimited (response-terminated).
    """

    task: str
    stimulus_factors: dict[str, object]
    display_params: dict[str, object]
    presentation_duration_ms: int | None
    correct_index: int
    choices: list[str]
    correct_key: str
    data: dict[str, object]


# Backward-compatible alias used by jsPsych adapters.
JsPsychTrial = Trial


class SimulatedObservation(TypedDict):
    subj: int
    stim_level: float
    choice_index: int
    response: int  # binary mapping for current DDM pipeline
    rt: float
    correct: int


class JsPsychResultsMessage(TypedDict):
    type: str
    rows: list[dict[str, object]]
