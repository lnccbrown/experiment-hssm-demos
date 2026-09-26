"""Fit and check the PST RL-DDM through HSSM's native RLSSM toolchain.

The same ``ssms.rl`` model drives simulation, HSSM's built-in differentiable DDM LAN drives
inference, and ``ssms.rl.Simulator(..., mode="ppc")`` generates posterior-predictive datasets.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from actors.pst_rl import PSTLearner, PSTSimulator, make_model

PARAMETERS: dict[str, tuple[str, str]] = {
    "eta_pos": ("Learning from wins", "How far one positive prediction error moves a symbol's value."),
    "eta_neg": ("Learning from losses", "How far one negative prediction error moves a symbol's value."),
    "eta": ("Learning rate", "How far each outcome moves a symbol's value."),
    "m": ("Value sensitivity", "How strongly a difference in learned values pushes the choice."),
    "bb": ("Caution", "Evidence needed before answering, at trial 10."),
    "bp": ("Change in caution", "Below 0 means the decision boundary falls over trials."),
    "a": ("Caution", "Evidence needed before answering."),
    "t": ("Non-decision time", "Modeled seconds outside evidence accumulation."),
}

# Natural-scale prior locations. Coefficients are placed on HSSM's generalized-logit scale below,
# so hierarchical participant effects cannot leave the registered parameter support.
PRIOR_LOCATION_SCALE: dict[str, tuple[float, float]] = {
    "eta_pos": (0.20, 0.14),
    "eta_neg": (0.20, 0.14),
    "eta": (0.20, 0.14),
    "m": (2.5, 2.0),
    "bb": (1.0, 0.5),
    "bp": (0.0, 0.3),
    "a": (1.0, 0.5),
    "t": (0.30, 0.12),
}


def _inverse_link(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """HSSM's generalized-logit inverse for a finite parameter interval."""
    lower, upper = bounds
    values = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return lower + (upper - lower) / (1.0 + np.exp(-values))


def _link_value(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    value = float(np.clip(value, lower + 1e-8, upper - 1e-8))
    return float(np.log((value - lower) / (upper - value)))


@dataclass
class PSTFit:
    """A fitted HSSM RLSSM and the metadata needed for native SSMS PPC."""

    model: Any
    idata: Any
    learner: PSTLearner
    table: pd.DataFrame
    seconds: float
    bounds: dict[str, tuple[float, float]]
    hierarchical: bool

    @property
    def parameters(self) -> list[str]:
        fixed_boundary = ["a"] if self.learner.boundary == "fixed" else []
        return [*self.learner.free_params, *fixed_boundary, "t"]

    @property
    def participant_ids(self) -> list[Any]:
        return self.table["participant_id"].drop_duplicates().tolist()

    @property
    def n_samples(self) -> int:
        posterior = self.idata.posterior
        return int(posterior.sizes["chain"] * posterior.sizes["draw"])

    def _common_linear(self, parameter: str) -> np.ndarray:
        return np.asarray(self.idata.posterior[f"{parameter}_Intercept"], dtype=float)

    def population_arrays(self) -> dict[str, np.ndarray]:
        """Natural-scale group parameters, retaining ``(chain, draw)`` dimensions."""
        return {
            parameter: _inverse_link(self._common_linear(parameter), self.bounds[parameter])
            for parameter in self.parameters
        }

    def draws(self, n: int | None = None, seed: int = 0) -> dict[str, np.ndarray]:
        """Natural-scale group-level draws, with chains pooled."""
        out = {name: values.reshape(-1) for name, values in self.population_arrays().items()}
        size = len(next(iter(out.values())))
        if n is not None and n < size:
            keep = np.random.default_rng(seed).choice(size, size=n, replace=False)
            out = {name: values[keep] for name, values in out.items()}
        return out

    def participant_draws(self, sample_ids: Sequence[int]) -> dict[str, np.ndarray]:
        """Natural-scale draws shaped ``(selected samples, participants)`` for SSMS."""
        posterior = self.idata.posterior
        sample_ids = np.asarray(sample_ids, dtype=int)
        n_participants = len(self.participant_ids)
        out: dict[str, np.ndarray] = {}
        for parameter in self.parameters:
            common = self._common_linear(parameter).reshape(self.n_samples, 1)
            if self.hierarchical:
                name = f"{parameter}_1|participant_id"
                effect = posterior[name]
                participant_dim = next(dim for dim in effect.dims if dim not in ("chain", "draw"))
                labels = [str(value) for value in effect.coords[participant_dim].values]
                order = [labels.index(str(pid)) for pid in self.participant_ids]
                deviations = np.asarray(effect, dtype=float).reshape(self.n_samples, -1)[:, order]
                linear = common + deviations
            else:
                linear = np.repeat(common, n_participants, axis=1)
            out[parameter] = _inverse_link(linear[sample_ids], self.bounds[parameter])
        out["z"] = np.full((len(sample_ids), n_participants), 0.5)
        return out


def _coefficient_prior(name: str, bounds: tuple[float, float]):
    """A Normal prior on the link scale, centered at the intended natural-scale value."""
    import hssm

    natural_mu, natural_sigma = PRIOR_LOCATION_SCALE[name]
    # Dynamic bounds (notably t < the fastest retained RT) can exclude the nominal prior
    # location. Keep the adjusted centre away from an asymptote of the generalized logit.
    width = bounds[1] - bounds[0]
    margin = max(1e-6, 0.1 * width)
    natural_mu = float(np.clip(natural_mu, bounds[0] + margin, bounds[1] - margin))
    mu = _link_value(natural_mu, bounds)
    lower, upper = bounds
    derivative = (upper - lower) / ((natural_mu - lower) * (upper - natural_mu))
    sigma = float(np.clip(natural_sigma * derivative, 0.35, 1.5))
    return hssm.Prior("Normal", mu=mu, sigma=sigma)


def _param(name: str, bounds: tuple[float, float], *, hierarchical: bool):
    import hssm

    formula = f"{name} ~ 1" + (" + (1|participant_id)" if hierarchical else "")
    prior: dict[str, Any] = {"Intercept": _coefficient_prior(name, bounds)}
    if hierarchical:
        prior["1|participant_id"] = {
            "name": "Normal",
            "mu": 0.0,
            "sigma": {"name": "HalfNormal", "sigma": 0.5},
        }
    return hssm.Param(name, formula=formula, prior=prior)


def _t_upper_bound(table: pd.DataFrame) -> float:
    """Largest admissible shared bound for t, below the fastest retained response."""
    upper = min(1.5, float(table["rt"].min()) - 1e-3)
    if not np.isfinite(upper) or upper <= 0.05:
        raise ValueError("retained response times must all exceed 0.051 seconds")
    return upper


def fit_pst(
    table: pd.DataFrame,
    *,
    learner: PSTLearner | None = None,
    draws: int = 300,
    tune: int = 300,
    chains: int = 4,
    seed: int = 0,
    target_accept: float = 0.95,
) -> PSTFit:
    """Fit a single-participant or hierarchical PST model with HSSM's built-in DDM LAN."""
    import hssm

    learner = learner or PSTLearner()
    table = table.sort_values(["participant_id", "trial_id"], kind="stable").reset_index(drop=True)
    counts = table.groupby("participant_id", sort=False).size()
    if table.empty or counts.nunique() != 1:
        raise ValueError("RLSSM requires a non-empty balanced panel (equal retained trials per participant)")

    # Stock HSSM bridge: no custom SSM registration or likelihood override.
    config = hssm.rl.RLSSMConfig.from_ssms_model(make_model(learner=learner))
    if config.decision_process_loglik_kind != "approx_differentiable":
        raise RuntimeError("PST fitting expected HSSM's built-in differentiable DDM LAN")

    # Every participant-specific t must be below every fitted RT. Using the largest
    # participant minimum would admit impossible t values for faster participants.
    t_upper = _t_upper_bound(table)
    config.bounds["t"] = (0.05, t_upper)
    if "t" in config.list_params:
        config.params_default[config.list_params.index("t")] = min(0.3, t_upper - 1e-3)

    hierarchical = len(counts) > 1
    parameters = [*learner.free_params, *(["a"] if learner.boundary == "fixed" else []), "t"]
    bounds = {name: tuple(config.bounds[name]) for name in parameters}
    include = [_param(name, bounds[name], hierarchical=hierarchical) for name in parameters]
    include.append(hssm.Param("z", prior=0.5))

    start = time.perf_counter()
    model = hssm.RLSSM(
        data=table,
        model_config=config,
        p_outlier=0,
        lapse=None,
        process_initvals=False,
        link_settings="log_logit",
        prior_settings="safe",
        include=include,
    )
    idata = model.sample(
        sampler="numpyro",
        chains=chains,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        random_seed=seed,
        progressbar=False,
    )
    return PSTFit(model, idata, learner, table, time.perf_counter() - start, bounds, hierarchical)


def _interval_columns(summary: pd.DataFrame) -> tuple[str, str]:
    lower = [column for column in summary if column.endswith("_lb")]
    upper = [column for column in summary if column.endswith("_ub")]
    if lower and upper:
        return lower[0], upper[0]
    hdi = [column for column in summary if column.startswith("hdi_")]
    if len(hdi) != 2:
        raise KeyError(f"Could not identify HDI columns in {list(summary.columns)}")
    return hdi[0], hdi[1]


def posterior_table(fit: PSTFit, truth: Mapping[str, float] | None = None, prob: float = 0.94) -> pd.DataFrame:
    """Natural-scale group estimates, uncertainty, and core chain diagnostics."""
    import arviz as az
    import xarray as xr

    posterior = fit.idata.posterior
    dataset = xr.Dataset(
        {
            name: xr.DataArray(
                values,
                dims=("chain", "draw"),
                coords={"chain": posterior.chain.values, "draw": posterior.draw.values},
            )
            for name, values in fit.population_arrays().items()
        }
    )
    summary = az.summary(dataset, ci_kind="hdi", ci_prob=prob, round_to="none")
    low_col, high_col = _interval_columns(summary)
    rows: list[dict[str, object]] = []
    for parameter in fit.parameters:
        stats = summary.loc[parameter]
        row: dict[str, object] = {
            "parameter": PARAMETERS[parameter][0],
            "symbol": parameter,
            "estimate": float(stats["mean"]),
            "low": float(stats[low_col]),
            "high": float(stats[high_col]),
            "mcse_mean": float(stats["mcse_mean"]),
            "ess_bulk": float(stats["ess_bulk"]),
            "ess_tail": float(stats["ess_tail"]),
            "r_hat": float(stats["r_hat"]),
            "meaning": PARAMETERS[parameter][1],
        }
        if truth is not None and parameter in truth:
            row["true"] = float(truth[parameter])
            row["recovered"] = bool(row["low"] <= row["true"] <= row["high"])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_diagnostics(fit: PSTFit) -> dict[str, Any]:
    """Sampler checks used to gate substantive notebook interpretations."""
    import arviz as az

    chains = int(fit.idata.posterior.sizes["chain"])
    summary = az.summary(fit.idata, kind="diagnostics", round_to="none")
    rhat_values = np.asarray(summary["r_hat"], dtype=float)
    bulk_values = np.asarray(summary["ess_bulk"], dtype=float)
    tail_values = np.asarray(summary["ess_tail"], dtype=float)
    max_rhat = float(np.max(rhat_values)) if np.isfinite(rhat_values).all() else float("nan")
    min_bulk = float(np.min(bulk_values)) if np.isfinite(bulk_values).all() else float("nan")
    min_tail = float(np.min(tail_values)) if np.isfinite(tail_values).all() else float("nan")
    sample_stats = fit.idata.sample_stats
    divergences = int(np.asarray(sample_stats["diverging"]).sum()) if "diverging" in sample_stats else 0
    try:
        min_bfmi = float(np.nanmin(np.asarray(az.bfmi(fit.idata), dtype=float)))
    except (TypeError, ValueError):
        min_bfmi = float("nan")
    max_tree_depth = int(np.asarray(sample_stats["tree_depth"]).max()) if "tree_depth" in sample_stats else None
    reached_max_tree_depth = (
        int(np.asarray(sample_stats["reached_max_treedepth"]).sum())
        if "reached_max_treedepth" in sample_stats
        else 0
    )

    issues: list[str] = []
    if chains < 4:
        issues.append("At least four chains are required for the notebook's convergence gate.")
    elif not np.isfinite(max_rhat) or max_rhat > 1.01:
        issues.append(f"Maximum r-hat is {max_rhat:.3f}; the target is at most 1.01.")
    required_ess = max(400, 100 * chains)
    if not np.isfinite(min_bulk) or min_bulk < required_ess:
        issues.append(f"Minimum bulk ESS is {min_bulk:.0f}; target at least {required_ess} total.")
    if not np.isfinite(min_tail) or min_tail < required_ess:
        issues.append(f"Minimum tail ESS is {min_tail:.0f}; target at least {required_ess} total.")
    if divergences:
        issues.append(f"The sampler reported {divergences} divergent transition(s).")
    if np.isfinite(min_bfmi) and min_bfmi < 0.3:
        issues.append(f"Minimum BFMI is {min_bfmi:.2f}; values below 0.30 indicate poor energy exploration.")
    if reached_max_tree_depth:
        issues.append(f"The sampler hit maximum tree depth {reached_max_tree_depth} time(s).")
    elif max_tree_depth is not None and max_tree_depth >= 10:
        issues.append(f"The sampler reached tree depth {max_tree_depth}; the default maximum is 10.")
    return {
        "ok": not issues,
        "issues": issues,
        "chains": chains,
        "divergences": divergences,
        "max_r_hat": max_rhat,
        "min_ess_bulk": min_bulk,
        "min_ess_tail": min_tail,
        "min_bfmi": min_bfmi,
        "max_tree_depth": max_tree_depth,
    }


def posterior_predictive(fit: PSTFit, *, n_draws: int = 40, seed: int = 0) -> pd.DataFrame:
    """Generate replicated choice/RT datasets with SSMS' observed-history-conditioned PPC."""
    n_draws = min(int(n_draws), fit.n_samples)
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(fit.n_samples, size=n_draws, replace=False)
    parameters = fit.participant_draws(sample_ids)
    simulator = PSTSimulator(make_model(learner=fit.learner))
    frames = []
    for ppc_draw in range(n_draws):
        theta = {name: values[ppc_draw] for name, values in parameters.items()}
        replicated = simulator.simulate(
            theta=theta,
            mode="ppc",
            observed_data=fit.table,
            random_state=seed + 10_000 + ppc_draw,
        )
        frames.append(replicated.assign(ppc_draw=ppc_draw))
    return pd.concat(frames, ignore_index=True)


def posterior_predictive_curves(
    fit: PSTFit,
    rows: pd.DataFrame,
    *,
    n_draws: int = 40,
    bin_size: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize native PPC replicated choices, RTs, and deadline omissions by learning bin."""
    metadata = rows.sort_values(["participant_id", "trial"], kind="stable").copy()
    metadata["trial_id"] = metadata.groupby("participant_id", sort=False).cumcount()
    if len(metadata) != len(fit.table):
        raise ValueError("rows must be the same balanced retained panel used for fitting")
    metadata = metadata[["participant_id", "trial_id", "pair", "trial_in_pair"]]
    ppc = posterior_predictive(fit, n_draws=n_draws, seed=seed).merge(
        metadata, on=["participant_id", "trial_id"], how="left", validate="many_to_one"
    )
    ppc["bin"] = (ppc["trial_in_pair"] - 1) // bin_size + 1
    ppc["trials"] = (ppc["bin"] - 0.5) * bin_size
    ppc["valid"] = ppc["rt"].gt(0) & ppc["response"].isin((-1, 1))
    ppc["chose_better"] = np.where(ppc["valid"], ppc["response"].eq(1), np.nan)
    ppc["valid_rt"] = ppc["rt"].where(ppc["valid"])
    ppc["timed_out"] = ~ppc["valid"]

    per_draw = (
        ppc.groupby(["ppc_draw", "pair", "bin", "trials"], sort=False)
        .agg(
            p_better=("chose_better", "mean"),
            rt_mean=("valid_rt", "mean"),
            p_timeout=("timed_out", "mean"),
        )
        .reset_index()
    )

    def summarize(column: str, output: str) -> pd.DataFrame:
        grouped = per_draw.groupby(["pair", "bin", "trials"])[column]
        result = grouped.agg(
            **{
                output: "mean",
                "low": lambda values: values.quantile(0.05),
                "high": lambda values: values.quantile(0.95),
            }
        )
        return result.reset_index().sort_values(["pair", "bin"]).reset_index(drop=True)

    return {
        "data": ppc,
        "choice": summarize("p_better", "p_better"),
        "rt": summarize("rt_mean", "rt_mean"),
        "timeout": summarize("p_timeout", "p_timeout"),
        "timeout_rate": float(ppc["timed_out"].mean()),
    }


def personal_readout(fit: PSTFit, *, mixed_response_methods: bool = False) -> list[dict[str, str]]:
    """Cautious, session- and model-conditional summaries of selected posterior contrasts."""
    draws = fit.draws()
    items: list[dict[str, str]] = []
    prefix = "Within this shortened session, fitted model, and its regularizing priors, "
    if fit.learner.dual_learning_rates:
        probability = float(np.mean(draws["eta_pos"] > draws["eta_neg"]))
        if probability >= 0.8:
            value = "Wins"
            text = prefix + f"the learning rate for positive errors is probably larger ({probability:.0%} of draws)."
        elif probability <= 0.2:
            value = "Losses"
            text = prefix + f"the learning rate for negative errors is probably larger ({1 - probability:.0%} of draws)."
        else:
            value = "Unresolved"
            text = prefix + "the data do not clearly distinguish the two learning rates."
        items.append({"topic": "Positive vs negative learning", "value": value, "text": text})
    items.append(
        {
            "topic": "Value sensitivity m",
            "value": f"{np.median(draws['m']):.1f}",
            "text": prefix + "this is the estimated gain from learned value difference to drift.",
        }
    )
    if fit.learner.boundary == "power":
        decreasing = float(np.mean(draws["bp"] < 0))
        if decreasing >= 0.8:
            value = "Decreased"
            text = prefix + "the decision boundary probably fell as trials progressed."
        elif decreasing <= 0.2:
            value = "Increased"
            text = prefix + "the decision boundary probably rose as trials progressed."
        else:
            value = "Unresolved"
            text = prefix + "the posterior does not clearly distinguish increasing from decreasing caution."
        items.append({"topic": "Decision boundary over trials", "value": value, "text": text})
    if mixed_response_methods:
        items.append(
            {
                "topic": "Non-decision time t",
                "value": "Not interpreted",
                "text": "Key and click responses were mixed, so motor-method differences are confounded with t.",
            }
        )
    else:
        items.append(
            {
                "topic": "Non-decision time t",
                "value": f"{np.median(draws['t']):.2f} s",
                "text": prefix + "this is modeled residual time, not a direct measurement of perception or movement.",
            }
        )
    return items
