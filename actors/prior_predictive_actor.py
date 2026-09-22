# Prior-predictive virtual n-AFC actor: noisy latent strengths, argmax choice, coherence-scaled RT.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

EvidenceModelFn = Callable[["NAfcActor", np.ndarray, np.ndarray], np.ndarray]
StimulusToStrengthsFn = Callable[[dict[str, object]], list[float]]


@dataclass(frozen=True)
class NAfcActor:
    """Virtual n-AFC actor: ``stimulus_factors`` → ``(choice_index, rt)``."""

    sigma0: float = 0
    sigma_scale: float = 1.0
    lapse_rate: float = 0.02
    rt_scale: float = 0.35
    rt_noise: float = 0.03
    evidence_weight: tuple[float, ...] | None = None
    stimulus_to_strengths: StimulusToStrengthsFn | None = None
    evidence_model: EvidenceModelFn | None = None
    rng: np.random.Generator | None = field(default=None, compare=False)

    def _rng(self) -> np.random.Generator:
        """Return the injected RNG or a fresh default."""
        return self.rng if self.rng is not None else np.random.default_rng()

    def latent_strengths(self, stimulus_factors: dict[str, object]) -> list[float]:
        """Map trial factors to per-alternative latent strengths (task-specific hook)."""
        if self.stimulus_to_strengths is None:
            raise ValueError("stimulus_to_strengths must be set to derive latent strengths")
        return list(self.stimulus_to_strengths(stimulus_factors))

    def _weights(self, n_alternatives: int) -> np.ndarray:
        """Per-alternative evidence multipliers; defaults to ones."""
        if self.evidence_weight is None:
            return np.ones(n_alternatives, dtype=float)
        weights = np.asarray(self.evidence_weight, dtype=float)
        if weights.shape != (n_alternatives,):
            raise ValueError("evidence_weight length must match number of alternatives")
        return weights

    def _latent_evidence(self, weights: np.ndarray, strengths: np.ndarray) -> np.ndarray:
        """Noisy evidence per alternative: weight × strength + Gaussian noise."""
        if weights.shape != strengths.shape or weights.ndim != 1 or len(weights) == 0:
            raise ValueError(
                "evidence_weight and stim_strengths must be matching non-empty 1D arrays"
            )
        if self.evidence_model is not None:
            return self.evidence_model(self, weights, strengths)

        rng = self._rng()
        coherence = float(np.max(strengths))
        # Harder trials (low coherence) add more sensory noise.
        sigma = self.sigma0 + self.sigma_scale * max(0.0, 1.0 - coherence)
        return weights * strengths + rng.normal(0.0, sigma, size=len(weights))

    def _trial(
        self,
        weights: np.ndarray,
        strengths: np.ndarray,
        ndt: float,
    ) -> tuple[int, float]:
        """Simulate one trial: lapse or argmax choice, then coherence-scaled RT."""
        rng = self._rng()
        evidence = self._latent_evidence(weights, strengths)

        if rng.random() < self.lapse_rate:
            choice = int(rng.integers(0, len(evidence)))
            rt = max(0.05, ndt + self.rt_scale + rng.normal(0.0, self.rt_noise))
            return choice, rt

        if len(evidence) == 1:
            choice = int(evidence[0] > 0.0)
        else:
            choice = int(np.argmax(evidence))

        coherence = max(0.0, float(np.max(strengths)))
        rt = ndt + self.rt_scale * (1.0 - coherence) + rng.normal(0.0, self.rt_noise)
        return choice, rt

    def choose(self, stimulus_factors: dict[str, object], ndt: float) -> tuple[int, float]:
        """Public entry: factors → strengths → one simulated choice and RT."""
        strengths = np.asarray(self.latent_strengths(stimulus_factors), dtype=float)
        return self._trial(self._weights(len(strengths)), strengths, ndt)

    def choose_from_latent(
        self,
        evidence_weight: list[float] | np.ndarray,
        stim_strengths: list[float] | np.ndarray,
        ndt: float,
    ) -> tuple[int, float]:
        """Simulate from precomputed weights and strengths (testing / advanced use)."""
        return self._trial(
            np.asarray(evidence_weight, dtype=float),
            np.asarray(stim_strengths, dtype=float),
            ndt,
        )
