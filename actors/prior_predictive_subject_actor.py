# Prior-predictive subject actor: Gaussian draws over actor parameters, then forward simulate.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from actors.prior_predictive_actor import NAfcActor, StimulusToStrengthsFn


def _draw_normal(rng: np.random.Generator, mean: float, sd: float) -> float:
    if sd <= 0.0:
        return float(mean)
    return float(rng.normal(float(mean), float(sd)))


@dataclass(frozen=True)
class PriorPredictiveActor:
    """Sample ``NAfcActor`` parameters from Gaussian priors, then simulate trials.

    Mean fields match the demo sliders; ``*_sd`` fields set between-subject spread.
    A draw of zero SD fixes that parameter at its mean (same as a point actor).
    """

    sigma0_mean: float = 0.0
    sigma0_sd: float = 0.0
    sigma_scale_mean: float = 1.0
    sigma_scale_sd: float = 0.0
    lapse_rate_mean: float = 0.02
    lapse_rate_sd: float = 0.0
    ndt_mean: float = 0.3
    ndt_sd: float = 0.0
    rt_scale_mean: float = 0.35
    rt_scale_sd: float = 0.0
    rt_noise_mean: float = 0.03
    rt_noise_sd: float = 0.0
    evidence_weight: tuple[float, ...] | None = None
    stimulus_to_strengths: StimulusToStrengthsFn | None = None
    rng: np.random.Generator | None = field(default=None, compare=False)
    _actor: NAfcActor = field(init=False, repr=False, compare=False)
    _ndt: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        object.__setattr__(self, "_ndt", max(0.05, _draw_normal(rng, self.ndt_mean, self.ndt_sd)))
        object.__setattr__(
            self,
            "_actor",
            NAfcActor(
                sigma0=max(0.0, _draw_normal(rng, self.sigma0_mean, self.sigma0_sd)),
                sigma_scale=max(
                    0.0, _draw_normal(rng, self.sigma_scale_mean, self.sigma_scale_sd)
                ),
                lapse_rate=float(
                    np.clip(_draw_normal(rng, self.lapse_rate_mean, self.lapse_rate_sd), 0.0, 1.0)
                ),
                rt_scale=max(0.0, _draw_normal(rng, self.rt_scale_mean, self.rt_scale_sd)),
                rt_noise=max(0.0, _draw_normal(rng, self.rt_noise_mean, self.rt_noise_sd)),
                evidence_weight=self.evidence_weight,
                stimulus_to_strengths=self.stimulus_to_strengths,
                rng=rng,
            ),
        )

    def choose(self, stimulus_factors: dict[str, object], ndt: float) -> tuple[int, float]:
        """Simulate one trial using this subject's sampled parameters."""
        _ = ndt
        return self._actor.choose(stimulus_factors, self._ndt)
