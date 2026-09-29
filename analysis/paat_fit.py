"""HSSM ``angle`` regression for PAAT data, using only built-in infrastructure."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import pandas as pd

from observers.paat_ssm import AngleParameters, PAATDriftCoefficients

V_FORMULA = "v ~ (rel_reward_z + rel_aversive_z) * congruency_code"
V_HIERARCHICAL_FORMULA = (
    "v ~ (rel_reward_z + rel_aversive_z) * congruency_code + "
    "(1 + (rel_reward_z + rel_aversive_z) * congruency_code | participant_id)"
)

TERM_LABELS = {
    "v_Intercept": ("Drift intercept", "β₀"),
    "v_rel_reward_z": ("Reward slope · incongruent", "β reward"),
    "v_rel_aversive_z": ("Aversive slope · incongruent", "β aversive"),
    "v_congruency_code": ("Off-support congruency intercept", "β congruency*"),
    "v_rel_reward_z:congruency_code": ("Reward-slope change · congruent", "β reward×C"),
    "v_rel_aversive_z:congruency_code": ("Aversive-slope change · congruent", "β aversive×C"),
    "a": ("Decision threshold", "a"),
    "z": ("Starting bias toward risky", "z"),
    "t": ("Non-decision time", "t"),
    "theta": ("Linear boundary collapse", "θ"),
}

TRUTH_TERMS = {
    "v_Intercept": "intercept",
    "v_rel_reward_z": "reward",
    "v_rel_aversive_z": "aversive",
    "v_congruency_code": "congruency",
    "v_rel_reward_z:congruency_code": "reward_x_congruency",
    "v_rel_aversive_z:congruency_code": "aversive_x_congruency",
    "a": "a",
    "z": "z",
    "t": "t",
    "theta": "theta",
}

OFF_SUPPORT_TERMS = {"v_congruency_code"}


@dataclass
class PAATFit:
    model: object
    idata: object
    data: pd.DataFrame
    hierarchical: bool
    elapsed_s: float


def build_paat_model(data: pd.DataFrame, *, hierarchical: bool | None = None):
    """Construct the paper regression with HSSM's built-in ``angle`` model.

    In the teaching hierarchy, participant effects belong to drift only.  The
    remaining angle parameters are population-level simple parameters; using a
    global formula would silently add participant effects to every parameter.
    """
    import hssm

    required = {"participant_id", "rt", "response", "rel_reward_z", "rel_aversive_z", "congruency_code"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"PAAT fit data are missing columns: {missing}")
    if data.empty:
        raise ValueError("PAAT fit data are empty")
    if not set(data["response"].unique()).issubset({-1, 1}):
        raise ValueError("HSSM responses must be coded -1=safe and +1=risky")
    if hierarchical is None:
        hierarchical = data["participant_id"].nunique() > 1
    formula = V_HIERARCHICAL_FORMULA if hierarchical else V_FORMULA
    return hssm.HSSM(
        data=data,
        model="angle",
        loglik_kind="approx_differentiable",
        include=[{"name": "v", "formula": formula, "link": "identity"}],
        prior_settings="safe",
        # The stock simulator generates no contaminant process. Keep fitting
        # matched to that DGP instead of silently adding HSSM's default lapse mixture.
        p_outlier=0,
    )


def fit_paat(
    data: pd.DataFrame,
    *,
    hierarchical: bool | None = None,
    draws: int = 300,
    tune: int = 300,
    chains: int = 2,
    seed: int = 0,
    target_accept: float = 0.9,
) -> PAATFit:
    start = time.perf_counter()
    if hierarchical is None:
        hierarchical = data["participant_id"].nunique() > 1
    model = build_paat_model(data, hierarchical=hierarchical)
    idata = model.sample(
        sampler="numpyro",
        chains=int(chains),
        draws=int(draws),
        tune=int(tune),
        target_accept=float(target_accept),
        random_seed=int(seed),
        progressbar=False,
    )
    return PAATFit(model=model, idata=idata, data=data, hierarchical=bool(hierarchical), elapsed_s=time.perf_counter() - start)


def sample_posterior_predictive(fit: PAATFit, *, draws: int = 100):
    """Use HSSM's native posterior-predictive simulator for the angle model."""
    return fit.model.sample_posterior_predictive(
        dt=fit.idata,
        draws=int(draws),
        inplace=False,
        safe_mode=True,
    )


def posterior_table(fit: PAATFit, *, prob: float = 0.94) -> pd.DataFrame:
    import arviz as az

    available = set(fit.idata.posterior.data_vars)
    names = [name for name in TERM_LABELS if name in available]
    summary = az.summary(fit.idata, var_names=names, ci_kind="hdi", ci_prob=prob, round_to="none")
    low_col = next(column for column in summary.columns if column.endswith("_lb"))
    high_col = next(column for column in summary.columns if column.endswith("_ub"))
    rows = []
    for name in names:
        stats = summary.loc[name]
        label, symbol = TERM_LABELS[name]
        rows.append(
            {
                "term": name,
                "parameter": label,
                "symbol": symbol,
                "estimate": float(stats["mean"]),
                "low": float(stats[low_col]),
                "high": float(stats[high_col]),
                "r_hat": float(stats.get("r_hat", float("nan"))),
                "ess_bulk": float(stats.get("ess_bulk", float("nan"))),
                "ess_tail": float(stats.get("ess_tail", float("nan"))),
                "interpretation_supported": name not in OFF_SUPPORT_TERMS,
            }
        )
    return pd.DataFrame(rows)


def fit_diagnostics(
    fit: PAATFit,
    *,
    max_r_hat: float = 1.01,
    min_ess: float = 100.0,
) -> dict[str, object]:
    """Summarize all posterior diagnostics and decide whether interval checks are defensible."""
    import arviz as az

    posterior = fit.idata.posterior
    chains = int(posterior.sizes.get("chain", 0))
    draws = int(posterior.sizes.get("draw", 0))
    diagnostics = az.summary(fit.idata, kind="diagnostics", round_to="none")
    r_hat_available = bool(
        chains >= 2
        and not diagnostics.empty
        and diagnostics["r_hat"].notna().all()
    )
    max_observed_r_hat = (
        float(diagnostics["r_hat"].max()) if r_hat_available else float("nan")
    )
    min_bulk = (
        float(diagnostics["ess_bulk"].min()) if not diagnostics.empty else float("nan")
    )
    min_tail = (
        float(diagnostics["ess_tail"].min()) if not diagnostics.empty else float("nan")
    )

    sample_stats = getattr(fit.idata, "sample_stats", None)
    diverging = None if sample_stats is None else getattr(sample_stats, "diverging", None)
    divergences = int(diverging.sum().item()) if diverging is not None else 0

    issues: list[str] = []
    if chains < 2:
        issues.append("at least two chains are required for R-hat")
    elif not r_hat_available:
        issues.append("R-hat is unavailable for one or more parameters")
    elif max_observed_r_hat > max_r_hat:
        issues.append(f"maximum R-hat is {max_observed_r_hat:.3f} (target ≤ {max_r_hat:.2f})")
    if not pd.notna(min_bulk) or not pd.notna(min_tail):
        issues.append("effective sample size is unavailable")
    elif min(min_bulk, min_tail) < min_ess:
        issues.append(f"minimum ESS is {min(min_bulk, min_tail):.0f} (target ≥ {min_ess:.0f})")
    if divergences:
        issues.append(f"{divergences} divergent transition(s)")

    return {
        "ok": not issues,
        "chains": chains,
        "draws": draws,
        "max_r_hat": max_observed_r_hat,
        "min_ess_bulk": min_bulk,
        "min_ess_tail": min_tail,
        "divergences": divergences,
        "issues": issues,
    }


def truth_values(coefficients: PAATDriftCoefficients, angle: AngleParameters) -> dict[str, float]:
    return {**asdict(coefficients), **asdict(angle)}


def interval_coverage_table(
    fit: PAATFit,
    coefficients: PAATDriftCoefficients,
    angle: AngleParameters,
    *,
    prob: float = 0.94,
) -> pd.DataFrame:
    """Compare one simulated truth with its interval only when diagnostics pass."""
    table = posterior_table(fit, prob=prob)
    truth = truth_values(coefficients, angle)
    table["true"] = table["term"].map(lambda term: truth.get(TRUTH_TERMS.get(term, "")))
    if fit_diagnostics(fit)["ok"]:
        table["covered"] = table["true"].between(table["low"], table["high"]).astype(object)
        table.loc[~table["interpretation_supported"], "covered"] = None
    else:
        table["covered"] = None
    return table
