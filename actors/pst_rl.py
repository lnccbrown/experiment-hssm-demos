# Simulated PST players: the RL-DDM of Pedersen, Frank & Biele (2017) as ssms.rl plug-ins.
# One model definition drives ssms.rl.Simulator (players) and hssm.RLSSM (fitting).
# Units follow ssms/HSSM: `a` is half the Wiener boundary separation, so the paper's bb is halved.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from schemas.tasks.pst import (
    PST_RESPONSE_COLUMNS,
    PST_RESPONSE_DEADLINE_S,
    PSTConfig,
    PSTSchedule,
    PSTTrialSpec,
    make_schedule,
    score_choice,
    symbol_label,
)

CONTEXT_FIELDS = ("pair_id", "better_symbol", "worse_symbol", "trial", "feedback")

# HSSM's built-in DDM likelihood is a LAN trained on this decision-parameter support. The learner
# clamps the *computed* trial-wise parameters, which otherwise bypass HSSM's parameter transforms.
LAN_DDM_BOUNDS = {"v": (-3.0, 3.0), "a": (0.3, 2.5)}

# Group means from Pedersen et al. (2017), Table 2 (bb halved into ssms/HSSM units), and the
# middle of their parameter-recovery grid. z is fixed at 0.5 (unbiased) in every preset.
PRESETS: dict[str, dict[str, float]] = {
    "ADHD, on medication (Table 2)": {
        "eta_pos": 0.032, "eta_neg": 0.023, "m": 3.566, "bb": 1.849 / 2, "bp": 0.008, "t": 0.326,
    },
    "ADHD, off medication (Table 2)": {
        "eta_pos": 0.057, "eta_neg": 0.040, "m": 1.864, "bb": 1.659 / 2, "bp": 0.018, "t": 0.224,
    },
    "Recovery study, middle values": {
        "eta_pos": 0.09, "eta_neg": 0.09, "m": 2.75, "bb": 2.0 / 2, "bp": 0.0, "t": 0.40,
    },
}


class PSTLearner:
    """Values per symbol learned by a delta rule, with a DDM choice rule.

    drift     v = clip(m * (V[better] - V[worse]), -3, 3)
    boundary  a = clip(bb * (trial / 10) ** bp, .3, 2.5), or a free fixed ``a``
    update    V[chosen] += eta * (feedback - V[chosen]), with eta_pos / eta_neg chosen by the
              sign of the prediction error, or one ``eta`` when ``dual_learning_rates=False``

    The defaults are Pedersen et al.'s best model ("Model 6"), with values starting at 0.
    """

    n_actions = 2
    available_backends = ("python", "jax")
    supports_gradient = True

    def __init__(
        self,
        *,
        dual_learning_rates: bool = True,
        boundary: str = "power",
        initial_value: float = 0.0,
        n_symbols: int = 6,
    ) -> None:
        if boundary not in ("power", "fixed"):
            raise ValueError("boundary must be 'power' or 'fixed'")
        self.dual_learning_rates = dual_learning_rates
        self.boundary = boundary
        self.initial_value = float(initial_value)
        self.n_symbols = int(n_symbols)

    @property
    def computed_params(self) -> list[str]:
        return ["v", "a"] if self.boundary == "power" else ["v"]

    @property
    def free_params(self) -> list[str]:
        rates = ["eta_pos", "eta_neg"] if self.dual_learning_rates else ["eta"]
        return [*rates, "m", *(["bb", "bp"] if self.boundary == "power" else [])]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = {
            "eta_pos": (0.001, 0.999), "eta_neg": (0.001, 0.999), "eta": (0.001, 0.999),
            "m": (0.1, 10.0), "bb": (0.3, 2.5), "bp": (-0.5, 0.5),
        }
        return {name: bounds[name] for name in self.free_params}

    @property
    def default_params(self) -> dict[str, float]:
        defaults = {"eta_pos": 0.1, "eta_neg": 0.1, "eta": 0.1, "m": 3.0, "bb": 1.0, "bp": 0.0}
        return {name: defaults[name] for name in self.free_params}

    @property
    def required_context_fields(self) -> list[str]:
        return ["better_symbol", "worse_symbol", "trial", "choice", "feedback"]

    def init_state(self) -> dict[str, Any]:
        return {"values": np.full(self.n_symbols, self.initial_value)}

    def init_jax_state(self) -> dict[str, Any]:
        import jax.numpy as jnp

        return {"values": jnp.full(self.n_symbols, self.initial_value)}

    def compute_python(self, state: dict[str, Any], params: Mapping[str, float], context: Mapping[str, Any]) -> dict[str, float]:
        values = state["values"]
        better, worse = int(context["better_symbol"]), int(context["worse_symbol"])
        raw_v = params["m"] * (values[better] - values[worse])
        out = {"v": float(np.clip(raw_v, *LAN_DDM_BOUNDS["v"]))}
        if self.boundary == "power":
            raw_a = params["bb"] * (float(context["trial"]) / 10.0) ** params["bp"]
            out["a"] = float(np.clip(raw_a, *LAN_DDM_BOUNDS["a"]))
        return out

    def update_python(self, state: dict[str, Any], params: Mapping[str, float], context: Mapping[str, Any]) -> dict[str, Any]:
        values = np.array(state["values"], dtype=float)
        chosen = int(context["better_symbol"] if int(context["choice"]) == 1 else context["worse_symbol"])
        error = float(context["feedback"]) - values[chosen]
        if self.dual_learning_rates:
            rate = params["eta_pos"] if error > 0 else params["eta_neg"]
        else:
            rate = params["eta"]
        values[chosen] += rate * error
        return {"values": values}

    def compute_jax(self, state: dict[str, Any], params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
        import jax.numpy as jnp

        values = state["values"]
        better = jnp.asarray(context["better_symbol"], dtype=jnp.int32)
        worse = jnp.asarray(context["worse_symbol"], dtype=jnp.int32)
        raw_v = params["m"] * (values[better] - values[worse])
        out = {"v": jnp.clip(raw_v, *LAN_DDM_BOUNDS["v"])}
        if self.boundary == "power":
            trial = jnp.asarray(context["trial"], dtype=values.dtype)
            raw_a = params["bb"] * (trial / 10.0) ** params["bp"]
            out["a"] = jnp.clip(raw_a, *LAN_DDM_BOUNDS["a"])
        return out

    def update_jax(self, state: dict[str, Any], params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
        import jax.numpy as jnp

        values = state["values"]
        choice = jnp.asarray(context["choice"], dtype=jnp.int32)
        better = jnp.asarray(context["better_symbol"], dtype=jnp.int32)
        worse = jnp.asarray(context["worse_symbol"], dtype=jnp.int32)
        feedback = jnp.asarray(context["feedback"], dtype=values.dtype)
        chosen = jnp.where(choice == 1, better, worse).astype(jnp.int32)
        error = feedback - values[chosen]
        if self.dual_learning_rates:
            rate = jnp.where(error > 0, params["eta_pos"], params["eta_neg"])
        else:
            rate = params["eta"]
        return {"values": values.at[chosen].add(rate * error)}


class PSTEnvironment:
    """ssms.rl task environment that replays one PST learning-phase schedule.

    Choice 1 (response +1, upper boundary) is the better symbol of the pair. Feedback comes
    from the schedule's pre-sampled outcomes, so players sharing a schedule see the same rewards.
    """

    n_choices = 2

    def __init__(self, schedule: PSTSchedule, n_blocks: int | None = None) -> None:
        self.schedule = schedule
        self.trials = schedule.learning if n_blocks is None else schedule.learning_blocks(n_blocks)

    @property
    def context_fields(self) -> list[str]:
        return list(CONTEXT_FIELDS)

    @property
    def response_labels(self) -> list[int]:
        return [-1, 1]

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """The schedule is fixed in advance, so there is no state to reset."""

    def get_trial_context(self, trial_idx: int) -> dict[str, float]:
        trial = self.trials[trial_idx]
        return {
            "pair_id": float(trial.pair_id),
            "better_symbol": float(trial.better_symbol),
            "worse_symbol": float(trial.worse_symbol),
            "trial": float(trial.trial),
        }

    def sample_context(self, context: Mapping[str, Any], trial_idx: int) -> dict[str, float]:
        trial = self.trials[trial_idx]
        chosen = trial.better_symbol if int(context["choice"]) == 1 else trial.worse_symbol
        return {"feedback": float(trial.reward_for(chosen))}


def make_model(
    schedule: PSTSchedule | None = None,
    *,
    n_blocks: int | None = None,
    learner: PSTLearner | None = None,
    model_name: str = "PST_RLDDM",
):
    """ssms.rl ``ModelConfig`` for the PST learning phase (simulation, and fitting via HSSM)."""
    from ssms import rl

    return rl.ModelConfig(
        model_name=model_name,
        description="PST learning phase: delta-rule symbol values with a DDM choice rule (Pedersen et al., 2017)",
        decision_process="ddm",
        learning_process=learner or PSTLearner(),
        task_environment=PSTEnvironment(schedule if schedule is not None else make_schedule(seed=0), n_blocks),
        include_choice=True,
    )


class PSTSimulator:
    """Deadline-aware wrapper around :class:`ssms.rl.Simulator`.

    SSMS' DDM ``max_t`` limits integration time but still returns a boundary choice at the limit,
    and non-decision time is added afterward. PST instead has a deadline on *total* RT. Over-deadline
    draws therefore become SSMS omission sentinels before its learning loop can update values.
    """

    def __init__(self, config, *, deadline_s: float = PST_RESPONSE_DEADLINE_S) -> None:
        from ssms import rl

        class _DeadlineSimulator(rl.Simulator):
            def _simulate_decision_trial(inner_self, theta, rng):
                from ssms.basic_simulators import OMISSION_SENTINEL

                rt, response = super(_DeadlineSimulator, inner_self)._simulate_decision_trial(theta, rng)
                if not np.isfinite(rt) or rt > deadline_s:
                    return float(OMISSION_SENTINEL), response
                return rt, response

        self._simulator = _DeadlineSimulator(config)

    def simulate(self, *args, **kwargs) -> pd.DataFrame:
        """Delegate to SSMS, including its native ``mode='ppc'`` path."""
        return self._simulator.simulate(*args, **kwargs)


def vary_players(
    theta: Mapping[str, float],
    *,
    n_players: int,
    spread: float,
    seed: int | None = None,
) -> list[dict[str, float]]:
    """Individual differences: ``n_players`` parameter sets scattered around ``theta``.

    Parameters vary on bounded generalized-logit scales matching the hierarchical HSSM fit;
    ``spread`` is the SD on those scales (0 = identical players).
    """
    rng = np.random.default_rng(seed)
    bounds = {
        **PSTLearner().param_bounds,
        **PSTLearner(dual_learning_rates=False, boundary="fixed").param_bounds,
        "t": (0.05, 1.0),
        "a": LAN_DDM_BOUNDS["a"],
    }
    players = []
    for _ in range(n_players):
        player = dict(theta)
        if spread > 0:
            for name, value in theta.items():
                if name == "z" or name not in bounds:
                    continue
                lower, upper = bounds[name]
                value = float(np.clip(value, lower + 1e-9, upper - 1e-9))
                linear = np.log((value - lower) / (upper - value)) + rng.normal(0.0, spread)
                player[name] = float(lower + (upper - lower) / (1.0 + np.exp(-linear)))
        players.append(player)
    return players


def simulate_players(
    theta: Mapping[str, float] | Sequence[Mapping[str, float]],
    *,
    n_players: int | None = None,
    config: PSTConfig | None = None,
    n_blocks: int | None = None,
    learner: PSTLearner | None = None,
    schedule: PSTSchedule | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate PST learning-phase players and return the PST response table.

    ``theta`` is one parameter set for everyone, or one per player: the learner's parameters
    plus ``t`` (and ``a`` for a fixed boundary); ``z`` defaults to 0.5. Every player runs a
    fixed ``n_blocks`` (default ``max_blocks``) on their own schedule, unless ``schedule`` is
    given, e.g. to replay the schedule a human saw.
    """
    if isinstance(theta, Mapping):
        per_player = [dict(theta)] * (1 if n_players is None else n_players)
    else:
        per_player = [dict(p) for p in theta]
        if n_players is not None and n_players != len(per_player):
            raise ValueError("n_players does not match the number of parameter sets")
    config = config or (schedule.config if schedule is not None else PSTConfig())
    n_blocks = n_blocks or config.max_blocks
    rng = np.random.default_rng(seed)

    frames = []
    for player, params in enumerate(per_player):
        schedule_seed, sim_seed = (int(x) for x in rng.integers(0, 2**31 - 1, size=2))
        player_schedule = schedule if schedule is not None else make_schedule(config, seed=schedule_seed)
        model = make_model(player_schedule, n_blocks=n_blocks, learner=learner)
        missing = [name for name in model.list_params if name not in params and name != "z"]
        if missing:
            raise KeyError(f"theta is missing {missing} for this model")
        player_theta = {name: params.get(name, 0.5) for name in model.list_params}  # z -> 0.5
        raw = PSTSimulator(model).simulate(
            theta=player_theta, n_trials=model.task_environment.n_trials, n_participants=1, random_state=sim_seed
        )
        frames.append(_response_table(raw, model.task_environment.trials, player))
    return pd.concat(frames, ignore_index=True)


def _response_table(raw: pd.DataFrame, trials: Sequence[PSTTrialSpec], participant_id: int) -> pd.DataFrame:
    if len(raw) != len(trials):
        raise ValueError("simulator output does not match the schedule length")
    rows = []
    for rt, response, trial in zip(raw["rt"], raw["response"], trials):
        if rt < 0:  # ssms omission sentinel: no choice, no feedback, no learning
            side, rt = None, np.nan
        else:
            chosen = trial.better_symbol if int(response) == 1 else trial.worse_symbol
            side = "left" if chosen == trial.left_symbol else "right"
        rows.append(
            _response_row(
                trial,
                participant_id,
                side,
                float(rt),
                response_method=None if side is None else "simulated",
            )
        )
    frame = pd.DataFrame(rows, columns=list(PST_RESPONSE_COLUMNS))
    # Keep the no-feedback test phase compatible with learning tables during concatenation.
    frame["feedback"] = pd.to_numeric(frame["feedback"], errors="coerce")
    return frame


def _response_row(
    trial: PSTTrialSpec,
    participant_id: int,
    side: str | None,
    rt: float,
    *,
    response_method: str | None,
) -> dict[str, object]:
    return {
        "participant_id": participant_id,
        "phase": trial.phase,
        "block": trial.block,
        "trial": trial.trial,
        "pair": trial.pair,
        "pair_id": trial.pair_id,
        "left_symbol": trial.left_symbol,
        "right_symbol": trial.right_symbol,
        "better_symbol": trial.better_symbol,
        "worse_symbol": trial.worse_symbol,
        "choice_side": side,
        **score_choice(trial, side),
        "rt": rt,
        "anticipated": False,
        "response_method": response_method,
    }


def replay_latents(
    responses: pd.DataFrame,
    params: Mapping[str, float],
    learner: PSTLearner | None = None,
) -> pd.DataFrame:
    """Per trial of one player: symbol values before the trial (V_A..V_F), drift v, boundary a.

    ``responses`` is that player's learning-phase table in trial order.
    """
    if responses["participant_id"].nunique() > 1:
        raise ValueError("replay_latents expects a single player's trials")
    learner = learner or PSTLearner()
    state = learner.init_state()
    rows = []
    for row in responses.itertuples(index=False):
        context = {"better_symbol": row.better_symbol, "worse_symbol": row.worse_symbol, "trial": row.trial}
        values = {f"V_{symbol_label(i)}": float(v) for i, v in enumerate(state["values"])}
        rows.append({"trial": row.trial, **values, **learner.compute_python(state, params, context)})
        if not row.timed_out and not getattr(row, "anticipated", False):
            choice = int(row.response == 1)
            state = learner.update_python(state, params, {**context, "choice": choice, "feedback": row.feedback})
    return pd.DataFrame(rows)


def final_values(responses: pd.DataFrame, params: Mapping[str, float], learner: PSTLearner | None = None) -> np.ndarray:
    """Symbol values after one player's learning phase (inputs to the test phase)."""
    learner = learner or PSTLearner()
    state = learner.init_state()
    for row in responses.itertuples(index=False):
        if not row.timed_out and not getattr(row, "anticipated", False):
            context = {
                "better_symbol": row.better_symbol,
                "worse_symbol": row.worse_symbol,
                "choice": int(row.response == 1),
                "feedback": row.feedback,
            }
            state = learner.update_python(state, params, context)
    return np.asarray(state["values"], dtype=float)


def simulate_test_phase(
    values: Sequence[float],
    params: Mapping[str, float],
    schedule: PSTSchedule,
    *,
    last_learning_trial: int,
    participant_id: int = 0,
    learner: PSTLearner | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Test-phase choices from fixed values: same DDM choice rule, no feedback, no learning.

    The boundary stays where learning ended: a(t) at ``last_learning_trial``.
    """
    from ssms.basic_simulators.simulator import simulator

    learner = learner or PSTLearner()
    state = {"values": np.asarray(values, dtype=float)}
    computed = [
        learner.compute_python(
            state, params, {"better_symbol": t.better_symbol, "worse_symbol": t.worse_symbol, "trial": last_learning_trial}
        )
        for t in schedule.test
    ]
    n = len(computed)
    theta = {
        "v": np.array([c["v"] for c in computed]),
        "a": np.array([c.get("a", params.get("a")) for c in computed], dtype=float),
        "z": np.full(n, params.get("z", 0.5)),
        "t": np.full(n, params["t"]),
    }
    out = simulator(theta, model="ddm", n_samples=1, random_state=seed)
    rows = []
    for rt, choice, trial in zip(out["rts"].ravel(), out["choices"].ravel(), schedule.test):
        if not np.isfinite(rt) or float(rt) < 0 or float(rt) > PST_RESPONSE_DEADLINE_S:
            side, rt = None, np.nan
        else:
            chosen = trial.better_symbol if choice == 1 else trial.worse_symbol
            side = "left" if chosen == trial.left_symbol else "right"
        rows.append(
            _response_row(
                trial,
                participant_id,
                side,
                float(rt),
                response_method=None if side is None else "simulated",
            )
        )
    frame = pd.DataFrame(rows, columns=list(PST_RESPONSE_COLUMNS))
    frame["feedback"] = pd.to_numeric(frame["feedback"], errors="coerce")
    return frame
