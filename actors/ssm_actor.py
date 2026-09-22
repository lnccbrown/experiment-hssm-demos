# Forward DDM actor via ssm-simulators: stimulus strengths → signed drift → (choice, RT).

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

StimulusToStrengthsFn = Callable[[dict[str, object]], list[float]]


@dataclass(frozen=True)
class DdmActor:
    """Virtual 2AFC actor driven by a forward DDM simulation.

    Maps latent alternative strengths to a signed drift rate, runs one draw from
    ``ssms.basic_simulators.simulator`` (model ``ddm``), and returns
    ``(choice_index, rt)`` compatible with ``ExperimentGenerator.simulate()``.

    DDM parameters (HSSM / ssm-simulators convention):
        v_intercept: baseline drift (evidence-neutral trials).
        v_scale: drift increment per unit signed evidence
            (``strength[1] - strength[0]`` for left/right alternatives).
        a: boundary separation.
        z: starting point as a fraction of ``a`` (0.5 = unbiased).
        lapse_rate: probability of a random choice and lapse RT draw.
        lapse_rt_extra: mean extra decision time on lapse trials (added to ``ndt``).
    """

    v_intercept: float = 0.0
    v_scale: float = 2.5
    a: float = 1.2
    z: float = 0.5
    lapse_rate: float = 0.02
    lapse_rt_extra: float = 0.35
    model: str = "ddm"
    stimulus_to_strengths: StimulusToStrengthsFn | None = None
    rng: np.random.Generator | None = field(default=None, compare=False)

    def _rng(self) -> np.random.Generator:
        return self.rng if self.rng is not None else np.random.default_rng()

    def latent_strengths(self, stimulus_factors: dict[str, object]) -> list[float]:
        if self.stimulus_to_strengths is None:
            raise ValueError("stimulus_to_strengths must be set to derive latent strengths")
        return list(self.stimulus_to_strengths(stimulus_factors))

    def signed_evidence(self, strengths: np.ndarray) -> float:
        """Scalar evidence for drift: difference between right and left strengths."""
        if strengths.size == 0:
            raise ValueError("stim_strengths must be non-empty")
        if strengths.size == 1:
            return float(strengths[0])
        return float(strengths[1] - strengths[0])

    def drift_rate(self, strengths: np.ndarray) -> float:
        return float(self.v_intercept + self.v_scale * self.signed_evidence(strengths))

    def _simulate_ddm(self, *, v: float, ndt: float, seed: int) -> tuple[int, float]:
        from ssms.basic_simulators.simulator import simulator

        out = simulator(
            {"v": float(v), "a": float(self.a), "z": float(self.z), "t": float(ndt)},
            model=self.model,
            n_samples=1,
            random_state=int(seed),
            return_option="full",
        )
        ddm_choice = int(out["choices"].ravel()[0])
        rt = float(out["rts"].ravel()[0])
        choice_index = 0 if ddm_choice < 0 else 1
        return choice_index, max(0.05, rt)

    def _lapse_trial(self, n_alternatives: int, ndt: float) -> tuple[int, float]:
        rng = self._rng()
        choice = int(rng.integers(0, max(1, n_alternatives)))
        rt = max(
            0.05,
            float(ndt) + float(self.lapse_rt_extra) + rng.normal(0.0, 0.03),
        )
        return choice, rt

    def choose_from_latent(
        self,
        stim_strengths: list[float] | np.ndarray,
        ndt: float,
    ) -> tuple[int, float]:
        """Simulate one trial from precomputed per-alternative strengths."""
        strengths = np.asarray(stim_strengths, dtype=float)
        rng = self._rng()
        n_alternatives = max(1, int(strengths.size))

        if rng.random() < self.lapse_rate:
            return self._lapse_trial(n_alternatives, ndt)

        seed = int(rng.integers(0, 2**31 - 1))
        return self._simulate_ddm(v=self.drift_rate(strengths), ndt=ndt, seed=seed)

    def choose(self, stimulus_factors: dict[str, object], ndt: float) -> tuple[int, float]:
        """Public entry: factors → strengths → one DDM choice and RT."""
        strengths = np.asarray(self.latent_strengths(stimulus_factors), dtype=float)
        return self.choose_from_latent(strengths, ndt=float(ndt))
