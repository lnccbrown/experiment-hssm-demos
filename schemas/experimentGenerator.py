# Experiment-level orchestration: params, ordered trial-generator blocks, and simulation.
# Merges shared ``display_params`` into each trial and exposes ``all_trials()`` / ``simulate()``.
# Used by marimo demos and any Python-side batch actor runs.

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from schemas.contracts import ExperimentParams, Trial
from schemas.trial_generator import TrialGenerator


class ActorLike(Protocol):
    def choose(self, stimulus_factors: dict[str, object], ndt: float) -> tuple[int, float]: ...


class ExperimentGenerator:
    """Organize one experiment: shared parameters and one or more ``TrialGenerator`` blocks.

    Trial generators build and iterate trial parameter dicts. This class holds experiment-level
    defaults (display size, data writeout path) and walks attached generators in order.
    """

    def __init__(self, experiment_params: ExperimentParams | None = None) -> None:
        self._experiment_params: ExperimentParams = dict(experiment_params or {})
        self._trial_generators: list[TrialGenerator] = []
        self._generator_index = 0

    @property
    def experiment_params(self) -> ExperimentParams:
        return dict(self._experiment_params)

    @property
    def trial_generators(self) -> tuple[TrialGenerator, ...]:
        return tuple(self._trial_generators)

    @property
    def trial_generator_count(self) -> int:
        return len(self._trial_generators)

    @property
    def current_generator_index(self) -> int:
        return self._generator_index

    def set_experiment_params(self, experiment_params: ExperimentParams) -> None:
        self._experiment_params = dict(experiment_params)

    def add_trial_generator(self, generator: TrialGenerator) -> None:
        """Append a trial generator block (e.g. one coherence level, one condition)."""
        self._trial_generators.append(generator)

    def reset(self) -> None:
        """Reset experiment cursor and every attached trial generator."""
        self._generator_index = 0
        for generator in self._trial_generators:
            generator.reset()

    def has_next_trial(self) -> bool:
        return any(
            g.has_next() for g in self._trial_generators[self._generator_index :]
        )

    def next_trial(self) -> dict[str, object]:
        """Return the next trial from the current generator block, with experiment params applied."""
        while self._generator_index < len(self._trial_generators):
            generator = self._trial_generators[self._generator_index]
            if generator.has_next():
                return self._apply_experiment_params(generator.next_trial())
            self._generator_index += 1
        raise StopIteration("No more trials in experiment")

    def all_trials(self) -> list[dict[str, object]]:
        """Materialize every trial from all generators (does not advance cursors)."""
        return [
            self._apply_experiment_params(trial)
            for generator in self._trial_generators
            for trial in generator.all_trials()
        ]

    def simulate(
        self,
        *,
        actor_factory: Callable[[int], ActorLike],
        n_subjects: int,
        ndt: float,
        response_mapper: Callable[[int], int] | None = None,
    ) -> list[dict[str, object]]:
        """Run the experiment with simulated actors and return row records.

        Args:
            actor_factory: Builds one actor per subject.
            n_subjects: Number of simulated subjects.
            ndt: Non-decision time passed into ``actor.choose``.
            response_mapper: Optional mapping from ``choice_index`` to output response code.
        """
        subjects = max(0, int(n_subjects))
        rows: list[dict[str, object]] = []
        trials = self.all_trials()

        for subj in range(subjects):
            actor = actor_factory(subj)
            for tr in trials:
                trial_data = dict(tr.get("data") or {})
                stim_factors = dict(tr.get("stimulus_factors") or {})
                choice_index, rt = actor.choose(stim_factors, ndt=float(ndt))
                choice_idx = int(choice_index)
                correct_index = int(tr["correct_index"])
                if response_mapper is None:
                    response = -1 if choice_idx == 0 else 1
                else:
                    response = int(response_mapper(choice_idx))
                stim_level = float(
                    trial_data.get("stim_level", stim_factors.get("coherence", 0.0))
                )
                rows.append(
                    {
                        "subj": int(subj),
                        "stim_level": stim_level,
                        "choice_index": choice_idx,
                        "response": response,
                        "rt": float(rt),
                        "correct": int(choice_idx == correct_index),
                    }
                )
        return rows

    def _apply_experiment_params(self, trial: Trial | dict[str, object]) -> dict[str, object]:
        out = dict(trial)
        exp_display = dict(self._experiment_params.get("display_params") or {})
        trial_display = dict(out.get("display_params") or {})
        out["display_params"] = {**exp_display, **trial_display}

        output_path = self._experiment_params.get("data_output_path")
        if output_path:
            data = dict(out.get("data") or {})
            data.setdefault("data_output_path", output_path)
            out["data"] = data
        return out
