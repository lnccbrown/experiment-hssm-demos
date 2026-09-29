"""Probabilistic Approach-Avoidance Task (PAAT) designs and scoring.

The task layer owns probabilities, counterbalancing, reproducible outcome draws,
and response coding.  Decision dynamics are deliberately not implemented here:
``observers.paat_ssm`` passes the resulting trial-wise parameters to the stock
``ssm-simulators`` angle model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

PROBABILITY_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
RELATIVE_LEVELS = (0.2, 0.4, 0.6, 0.8)


@dataclass(frozen=True)
class PAATDesignProfile:
    """A named task design.

    ``incongruent_per_block`` is explicit because the final paper reports a
    60/36 split for Study 2, whereas complete sessions in the released OSF
    trial data contain 66/30.  The named Study 2 profiles are count-matched
    synthetic designs with a balanced evidence grid, not released trial lists.
    """

    name: str
    label: str
    n_blocks: int
    trials_per_block: int
    incongruent_per_block: int
    reward_amount: float
    reward_unit: str

    @property
    def n_trials(self) -> int:
        return self.n_blocks * self.trials_per_block

    @property
    def congruent_per_block(self) -> int:
        return self.trials_per_block - self.incongruent_per_block


DESIGN_PROFILES: dict[str, PAATDesignProfile] = {
    "conference": PAATDesignProfile(
        name="conference",
        label="Conference demo · 24 trials",
        n_blocks=2,
        trials_per_block=12,
        incongruent_per_block=6,
        reward_amount=10.0,
        reward_unit="points",
    ),
    "published_study2": PAATDesignProfile(
        name="published_study2",
        label="Count-matched Study 2 (paper) · 96 trials (60/36)",
        n_blocks=6,
        trials_per_block=16,
        incongruent_per_block=10,
        reward_amount=0.10,
        reward_unit="dollars",
    ),
    "osf_study2": PAATDesignProfile(
        name="osf_study2",
        label="Count-matched Study 2 (OSF) · 96 trials (66/30)",
        n_blocks=6,
        trials_per_block=16,
        incongruent_per_block=11,
        reward_amount=0.10,
        reward_unit="dollars",
    ),
}


@dataclass(frozen=True)
class PAATPredictorScale:
    """Frozen predictor scale shared by every synthetic schedule.

    The reference distribution uses the four intended evidence magnitudes
    uniformly and the published Study 2 60/36 congruency ratio.  It is fixed
    across seeds, profiles, and participants; it is not re-estimated from each
    realized schedule.
    """

    reward_mean: float
    reward_sd: float
    aversive_mean: float
    aversive_sd: float

    def reward_z(self, value):
        return (np.asarray(value, dtype=float) - self.reward_mean) / self.reward_sd

    def aversive_z(self, value):
        return (np.asarray(value, dtype=float) - self.aversive_mean) / self.aversive_sd


_REFERENCE_MAGNITUDES = np.asarray(RELATIVE_LEVELS, dtype=float)
_REFERENCE_INCONGRUENT_FRACTION = 60 / 96
_REFERENCE_REWARD_MEAN = float(
    (2 * _REFERENCE_INCONGRUENT_FRACTION - 1) * _REFERENCE_MAGNITUDES.mean()
)
PAAT_PREDICTOR_SCALE = PAATPredictorScale(
    reward_mean=_REFERENCE_REWARD_MEAN,
    reward_sd=float(
        np.sqrt(np.mean(_REFERENCE_MAGNITUDES**2) - _REFERENCE_REWARD_MEAN**2)
    ),
    aversive_mean=float(_REFERENCE_MAGNITUDES.mean()),
    aversive_sd=float(_REFERENCE_MAGNITUDES.std()),
)


@dataclass(frozen=True)
class PAATTrialSpec:
    """One free-choice PAAT trial, with all random events drawn in advance."""

    profile: str
    block: int
    trial: int
    congruency: str
    congruency_code: int  # 0 = incongruent reference, 1 = congruent
    p_reward_risky: float
    p_reward_safe: float
    p_aversive_risky: float
    p_aversive_safe: float
    rel_reward_z: float
    rel_aversive_z: float
    risky_side: str
    selected_wheel: str
    spin_risky: float
    spin_safe: float
    reward_amount: float
    reward_unit: str

    @property
    def safe_side(self) -> str:
        return "right" if self.risky_side == "left" else "left"

    @property
    def rel_reward(self) -> float:
        return round(self.p_reward_risky - self.p_reward_safe, 10)

    @property
    def rel_aversive(self) -> float:
        return round(self.p_aversive_risky - self.p_aversive_safe, 10)

    def option_for_side(self, side: str) -> str:
        if side not in ("left", "right"):
            raise ValueError(f"choice side must be left or right, got {side!r}")
        return "risky" if side == self.risky_side else "safe"

    def probability(self, option: str, wheel: str) -> float:
        if option not in ("risky", "safe"):
            raise ValueError(f"option must be risky or safe, got {option!r}")
        if wheel not in ("reward", "aversive"):
            raise ValueError(f"wheel must be reward or aversive, got {wheel!r}")
        return float(getattr(self, f"p_{wheel}_{option}"))

    def spin_for(self, option: str) -> float:
        if option not in ("risky", "safe"):
            raise ValueError(f"option must be risky or safe, got {option!r}")
        return float(getattr(self, f"spin_{option}"))


@dataclass(frozen=True)
class PAATSchedule:
    profile: PAATDesignProfile
    seed: int | None
    trials: tuple[PAATTrialSpec, ...]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for trial in self.trials:
            rows.append(
                {
                    **asdict(trial),
                    "safe_side": trial.safe_side,
                    "rel_reward": trial.rel_reward,
                    "rel_aversive": trial.rel_aversive,
                }
            )
        return pd.DataFrame(rows)


PAAT_RESPONSE_COLUMNS = (
    "participant_id",
    "profile",
    "block",
    "trial",
    "congruency",
    "congruency_code",
    "p_reward_risky",
    "p_reward_safe",
    "p_aversive_risky",
    "p_aversive_safe",
    "rel_reward",
    "rel_aversive",
    "rel_reward_z",
    "rel_aversive_z",
    "risky_side",
    "choice_side",
    "chosen_option",
    "response",
    "rt",
    "timed_out",
    "response_method",
    "selected_wheel",
    "spin_value",
    "outcome",
    "reward_value",
    "aversive_outcome",
)


def get_profile(profile: str | PAATDesignProfile) -> PAATDesignProfile:
    if isinstance(profile, PAATDesignProfile):
        return profile
    try:
        return DESIGN_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown PAAT profile {profile!r}; choose from {sorted(DESIGN_PROFILES)}") from exc


def _pair_with_difference(difference: float, rng: np.random.Generator) -> tuple[float, float]:
    """Return two allowed probabilities ``(low, high)`` with the requested difference."""
    difference_units = int(round(10 * difference))
    levels = np.asarray([1, 3, 5, 7, 9], dtype=int)
    lows = levels[levels + difference_units <= 9]
    low = int(rng.choice(lows))
    return low / 10.0, (low + difference_units) / 10.0


def _evidence_cells(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    """Cycle through the 4×4 relative-evidence grid before repeating cells."""
    grid = [(rew, avr) for rew in RELATIVE_LEVELS for avr in RELATIVE_LEVELS]
    cells: list[tuple[float, float]] = []
    while len(cells) < n:
        order = rng.permutation(len(grid))
        cells.extend(grid[int(i)] for i in order)
    return cells[:n]


def make_schedule(
    profile: str | PAATDesignProfile = "conference",
    *,
    seed: int | None = None,
) -> PAATSchedule:
    """Create a seeded, side-balanced PAAT schedule.

    Each block has the profile's exact congruency counts and an equal number of
    risky-left and risky-right trials.  Reward/aversive wheel selection and the
    spin values for either possible choice are pre-drawn, making browser and
    simulated sessions reproducible.
    """

    design = get_profile(profile)
    rng = np.random.default_rng(seed)
    n_incongruent = design.n_blocks * design.incongruent_per_block
    n_congruent = design.n_blocks * design.congruent_per_block
    cells = {
        "incongruent": iter(_evidence_cells(n_incongruent, rng)),
        "congruent": iter(_evidence_cells(n_congruent, rng)),
    }
    raw: list[PAATTrialSpec] = []

    for block in range(1, design.n_blocks + 1):
        labels = np.asarray(
            ["incongruent"] * design.incongruent_per_block
            + ["congruent"] * design.congruent_per_block,
            dtype=object,
        )
        labels = labels[rng.permutation(len(labels))]
        risky_left = np.asarray(
            [True] * (design.trials_per_block // 2)
            + [False] * (design.trials_per_block // 2)
        )
        risky_left = risky_left[rng.permutation(len(risky_left))]

        for label, on_left in zip(labels, risky_left, strict=True):
            reward_diff, aversive_diff = next(cells[str(label)])
            p_aversive_safe, p_aversive_risky = _pair_with_difference(aversive_diff, rng)
            reward_low, reward_high = _pair_with_difference(reward_diff, rng)
            if label == "incongruent":
                p_reward_safe, p_reward_risky = reward_low, reward_high
                congruency_code = 0
            else:
                p_reward_risky, p_reward_safe = reward_low, reward_high
                congruency_code = 1
            raw.append(
                PAATTrialSpec(
                    profile=design.name,
                    block=block,
                    trial=len(raw) + 1,
                    congruency=str(label),
                    congruency_code=congruency_code,
                    p_reward_risky=p_reward_risky,
                    p_reward_safe=p_reward_safe,
                    p_aversive_risky=p_aversive_risky,
                    p_aversive_safe=p_aversive_safe,
                    rel_reward_z=0.0,
                    rel_aversive_z=0.0,
                    risky_side="left" if bool(on_left) else "right",
                    selected_wheel="reward" if rng.random() < 0.5 else "aversive",
                    spin_risky=float(rng.random()),
                    spin_safe=float(rng.random()),
                    reward_amount=design.reward_amount,
                    reward_unit=design.reward_unit,
                )
            )

    trials = tuple(
        replace(
            trial,
            rel_reward_z=float(PAAT_PREDICTOR_SCALE.reward_z(trial.rel_reward)),
            rel_aversive_z=float(PAAT_PREDICTOR_SCALE.aversive_z(trial.rel_aversive)),
        )
        for trial in raw
    )
    schedule = PAATSchedule(profile=design, seed=seed, trials=trials)
    validate_schedule(schedule)
    return schedule


def validate_schedule(schedule: PAATSchedule) -> None:
    """Raise when a generated or reconstructed schedule violates its design."""
    profile = schedule.profile
    if len(schedule.trials) != profile.n_trials:
        raise ValueError(f"expected {profile.n_trials} trials, found {len(schedule.trials)}")
    for block in range(1, profile.n_blocks + 1):
        trials = [trial for trial in schedule.trials if trial.block == block]
        if len(trials) != profile.trials_per_block:
            raise ValueError(f"block {block} has {len(trials)} trials")
        incongruent = sum(trial.congruency == "incongruent" for trial in trials)
        if incongruent != profile.incongruent_per_block:
            raise ValueError(f"block {block} has {incongruent} incongruent trials")
        if sum(trial.risky_side == "left" for trial in trials) != profile.trials_per_block // 2:
            raise ValueError(f"block {block} does not counterbalance risky-option side")
    for trial in schedule.trials:
        if trial.rel_aversive <= 0:
            raise ValueError(f"trial {trial.trial}: risky option is not more aversive")
        expected = "incongruent" if trial.rel_reward > 0 else "congruent"
        if trial.congruency != expected:
            raise ValueError(f"trial {trial.trial}: probabilities disagree with congruency")
        if not np.isclose(
            trial.rel_reward_z,
            PAAT_PREDICTOR_SCALE.reward_z(trial.rel_reward),
        ):
            raise ValueError(f"trial {trial.trial}: relative reward uses the wrong reference scale")
        if not np.isclose(
            trial.rel_aversive_z,
            PAAT_PREDICTOR_SCALE.aversive_z(trial.rel_aversive),
        ):
            raise ValueError(f"trial {trial.trial}: relative aversive uses the wrong reference scale")
        probabilities = (
            trial.p_reward_risky,
            trial.p_reward_safe,
            trial.p_aversive_risky,
            trial.p_aversive_safe,
        )
        if any(round(value, 1) not in PROBABILITY_LEVELS for value in probabilities):
            raise ValueError(f"trial {trial.trial}: probability outside the PAAT grid")


def score_choice(trial: PAATTrialSpec, choice_side: str | None) -> dict[str, object]:
    """Score a choice using the paper's boundary coding: risky +1, safe −1."""
    if choice_side is None:
        return {
            "choice_side": None,
            "chosen_option": None,
            "response": None,
            "timed_out": True,
            "selected_wheel": "aversive",
            "spin_value": None,
            "outcome": 1,
            "reward_value": 0.0,
            "aversive_outcome": True,
        }

    option = trial.option_for_side(choice_side)
    wheel = trial.selected_wheel
    spin = trial.spin_for(option)
    outcome = int(spin < trial.probability(option, wheel))
    return {
        "choice_side": choice_side,
        "chosen_option": option,
        "response": 1 if option == "risky" else -1,
        "timed_out": False,
        "selected_wheel": wheel,
        "spin_value": spin,
        "outcome": outcome,
        "reward_value": trial.reward_amount if wheel == "reward" and outcome else 0.0,
        "aversive_outcome": bool(wheel == "aversive" and outcome),
    }


def response_record(
    trial: PAATTrialSpec,
    *,
    participant_id: int,
    choice_side: str | None,
    rt: float | None,
    response_method: str | None,
) -> dict[str, object]:
    """Return one canonical browser/simulator response row."""
    scored = score_choice(trial, choice_side)
    return {
        "participant_id": int(participant_id),
        "profile": trial.profile,
        "block": trial.block,
        "trial": trial.trial,
        "congruency": trial.congruency,
        "congruency_code": trial.congruency_code,
        "p_reward_risky": trial.p_reward_risky,
        "p_reward_safe": trial.p_reward_safe,
        "p_aversive_risky": trial.p_aversive_risky,
        "p_aversive_safe": trial.p_aversive_safe,
        "rel_reward": trial.rel_reward,
        "rel_aversive": trial.rel_aversive,
        "rel_reward_z": trial.rel_reward_z,
        "rel_aversive_z": trial.rel_aversive_z,
        "risky_side": trial.risky_side,
        **scored,
        "rt": None if scored["timed_out"] else float(rt),
        "response_method": response_method,
    }
