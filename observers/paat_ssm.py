"""PAAT simulation through the stock ssm-simulators ``angle`` model.

Only the experimental regression is task-specific.  The stochastic decision
process, response times, and choices all come from ``Simulator('angle')``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

from schemas.tasks.paat import PAATSchedule, make_schedule, response_record


@dataclass(frozen=True)
class PAATDriftCoefficients:
    intercept: float = 0.0
    reward: float = 0.63
    aversive: float = -0.30
    congruency: float = 0.0
    reward_x_congruency: float = -0.14
    aversive_x_congruency: float = 0.26

    def drift(self, rel_reward_z, rel_aversive_z, congruency_code):
        reward = np.asarray(rel_reward_z, dtype=float)
        aversive = np.asarray(rel_aversive_z, dtype=float)
        congruency = np.asarray(congruency_code, dtype=float)
        return (
            self.intercept
            + self.reward * reward
            + self.aversive * aversive
            + self.congruency * congruency
            + self.reward_x_congruency * reward * congruency
            + self.aversive_x_congruency * aversive * congruency
        )


def drift_coefficients_from_mapping(values: Mapping[str, object]) -> PAATDriftCoefficients:
    """Build every drift coefficient from UI/config values without silent defaults."""
    names = tuple(field.name for field in fields(PAATDriftCoefficients))
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"missing PAAT drift coefficient(s): {missing}")
    parsed = {name: float(values[name]) for name in names}
    if not all(np.isfinite(value) for value in parsed.values()):
        raise ValueError("PAAT drift coefficients must be finite")
    return PAATDriftCoefficients(**parsed)


@dataclass(frozen=True)
class AngleParameters:
    a: float = 1.4
    z: float = 0.53
    t: float = 0.30
    theta: float = 0.18


@dataclass(frozen=True)
class PAATPreset:
    label: str
    description: str
    coefficients: PAATDriftCoefficients
    angle: AngleParameters


PRESETS: dict[str, PAATPreset] = {
    "Study 1 pattern": PAATPreset(
        label="Study 1 pattern",
        description=(
            "Published Study 1 drift-slope pattern on the shared reference scale; "
            "the intercept and remaining angle parameters are illustrative."
        ),
        coefficients=PAATDriftCoefficients(),
        angle=AngleParameters(),
    ),
    "Reward driven": PAATPreset(
        label="Reward driven",
        description="Reward differences dominate choices; aversive differences have a smaller effect.",
        coefficients=PAATDriftCoefficients(reward=1.0, aversive=-0.18, reward_x_congruency=-0.1, aversive_x_congruency=0.1),
        angle=AngleParameters(z=0.52),
    ),
    "Avoidance driven": PAATPreset(
        label="Avoidance driven",
        description="Aversive differences strongly push evidence toward the safe option.",
        coefficients=PAATDriftCoefficients(reward=0.45, aversive=-0.9, reward_x_congruency=0.0, aversive_x_congruency=0.15),
        angle=AngleParameters(z=0.48),
    ),
}


def angle_parameter_bounds() -> dict[str, tuple[float, float]]:
    """Return the installed stock ``angle`` model's simulation/LAN bounds."""
    from ssms.config import model_config

    return {
        name: (float(bounds[0]), float(bounds[1]))
        for name, bounds in model_config["angle"]["param_bounds_dict"].items()
    }


def validate_angle_parameters(theta: pd.DataFrame) -> None:
    """Reject non-finite or out-of-support parameters before simulation."""
    violations: list[str] = []
    for name, (lower, upper) in angle_parameter_bounds().items():
        if name not in theta:
            violations.append(f"missing {name}")
            continue
        values = pd.to_numeric(theta[name], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(values) | (values < lower) | (values > upper)
        if bad.any():
            finite = values[np.isfinite(values)]
            observed = (
                f"observed {finite.min():.3g} to {finite.max():.3g}"
                if finite.size
                else "all values non-finite"
            )
            violations.append(
                f"{name}: {int(bad.sum())}/{len(values)} outside [{lower:g}, {upper:g}] ({observed})"
            )
    if violations:
        raise ValueError(
            "Generated parameters fall outside the installed angle model bounds: "
            + "; ".join(violations)
            + ". Reduce the effects or individual-difference spread."
        )


def trialwise_parameters(
    schedule: PAATSchedule,
    coefficients: PAATDriftCoefficients,
    angle: AngleParameters,
) -> pd.DataFrame:
    """The exact trial-wise matrix consumed by ``Simulator('angle')``."""
    design = schedule.to_frame()
    return pd.DataFrame(
        {
            "v": coefficients.drift(
                design["rel_reward_z"],
                design["rel_aversive_z"],
                design["congruency_code"],
            ),
            "a": angle.a,
            "z": angle.z,
            "t": angle.t,
            "theta": angle.theta,
        }
    )


def simulate_schedule(
    schedule: PAATSchedule,
    *,
    coefficients: PAATDriftCoefficients | None = None,
    angle: AngleParameters | None = None,
    participant_id: int = 0,
    seed: int | None = None,
    deadline_s: float = 6.0,
) -> pd.DataFrame:
    """Simulate one participant with the official linear-collapse simulator."""
    coefficients = coefficients or PAATDriftCoefficients()
    angle = angle or AngleParameters()
    theta = trialwise_parameters(schedule, coefficients, angle)
    validate_angle_parameters(theta)

    from ssms.basic_simulators import Simulator

    result = Simulator(model="angle").simulate(
        theta=theta,
        n_samples=1,
        delta_t=0.001,
        max_t=float(deadline_s),
        random_state=seed,
        return_option="full",
        n_threads=1,
    )
    choices = np.asarray(result["choices"]).reshape(len(schedule.trials), -1)[:, 0]
    rts = np.asarray(result["rts"]).reshape(len(schedule.trials), -1)[:, 0]

    records: list[dict[str, object]] = []
    for trial, response, rt, drift in zip(schedule.trials, choices, rts, theta["v"], strict=True):
        response = int(response)
        rt = float(rt)
        timed_out = response not in (-1, 1) or not np.isfinite(rt) or rt > deadline_s
        if timed_out:
            side = None
            rt_value = None
            method = None
        else:
            chose_risky = response == 1
            side = trial.risky_side if chose_risky else trial.safe_side
            rt_value = rt
            method = "simulated"
        record = response_record(
            trial,
            participant_id=participant_id,
            choice_side=side,
            rt=rt_value,
            response_method=method,
        )
        record.update(
            {
                "true_v": float(drift),
                "true_a": angle.a,
                "true_z": angle.z,
                "true_t": angle.t,
                "true_theta": angle.theta,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _vary_drift(
    coefficients: PAATDriftCoefficients,
    angle: AngleParameters,
    *,
    spread: float,
    rng: np.random.Generator,
) -> tuple[PAATDriftCoefficients, AngleParameters]:
    """Vary drift coefficients only, matching the notebook's teaching hierarchy."""
    if spread <= 0:
        return coefficients, angle
    coefficient_values = {}
    for name, value in asdict(coefficients).items():
        scale = spread * max(abs(float(value)), 0.2)
        coefficient_values[name] = float(rng.normal(float(value), scale))
    return PAATDriftCoefficients(**coefficient_values), angle


def simulate_players(
    *,
    profile: str = "published_study2",
    coefficients: PAATDriftCoefficients | None = None,
    angle: AngleParameters | None = None,
    n_players: int = 5,
    spread: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate independent schedules for a cohort of PAAT participants."""
    if n_players < 1:
        raise ValueError("n_players must be positive")
    coefficients = coefficients or PAATDriftCoefficients()
    angle = angle or AngleParameters()
    rng = np.random.default_rng(seed)
    frames = []
    for participant_id in range(n_players):
        participant_coefficients, participant_angle = _vary_drift(
            coefficients,
            angle,
            spread=spread,
            rng=rng,
        )
        schedule_seed = int(rng.integers(0, 2**31 - 1))
        simulator_seed = int(rng.integers(0, 2**31 - 1))
        frames.append(
            simulate_schedule(
                make_schedule(profile, seed=schedule_seed),
                coefficients=participant_coefficients,
                angle=participant_angle,
                participant_id=participant_id,
                seed=simulator_seed,
            )
        )
    return pd.concat(frames, ignore_index=True)
