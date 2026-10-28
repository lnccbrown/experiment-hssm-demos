import arviz as az
import numpy as np
import pandas as pd
import pytest

from analysis.paat_analysis import fit_table
from analysis.paat_fit import (
    PAATFit,
    V_FORMULA,
    V_HIERARCHICAL_FORMULA,
    build_paat_model,
    fit_diagnostics,
    fit_paat,
    interval_coverage_table,
    posterior_table,
    sample_posterior_predictive,
)
from observers.paat_ssm import AngleParameters, PAATDriftCoefficients, simulate_players


def _fake_fit(
    *,
    chains=2,
    draws=400,
    divergences=0,
    bad_unreported_parameter=False,
    include_off_support_term=False,
):
    rng = np.random.default_rng(21)
    diverging = np.zeros((chains, draws), dtype=bool)
    diverging.flat[:divergences] = True
    posterior = {"v_Intercept": rng.normal(0.0, 0.1, size=(chains, draws))}
    if include_off_support_term:
        posterior["v_congruency_code"] = rng.normal(0.0, 0.1, size=(chains, draws))
    if bad_unreported_parameter:
        posterior["participant_offset"] = np.vstack(
            [rng.normal(-5.0, 0.1, size=draws), rng.normal(5.0, 0.1, size=draws)]
        )
    idata = az.from_dict(
        {
            "posterior": posterior,
            "sample_stats": {"diverging": diverging},
        }
    )
    return PAATFit(model=None, idata=idata, data=pd.DataFrame(), hierarchical=False, elapsed_s=0.0)


def test_formulas_match_the_paper_model_and_limit_participant_effects_to_drift():
    assert V_FORMULA == "v ~ (rel_reward_z + rel_aversive_z) * congruency_code"
    assert V_FORMULA in V_HIERARCHICAL_FORMULA
    assert "participant_id" in V_HIERARCHICAL_FORMULA


def test_diagnostics_surface_rhat_ess_and_divergences():
    fit = _fake_fit()
    table = posterior_table(fit)
    diagnostics = fit_diagnostics(fit)

    assert {"r_hat", "ess_bulk", "ess_tail"} <= set(table.columns)
    assert diagnostics["ok"] is True
    assert diagnostics["chains"] == 2
    assert diagnostics["divergences"] == 0
    assert diagnostics["max_r_hat"] <= 1.01

    divergent = fit_diagnostics(_fake_fit(divergences=2))
    assert divergent["ok"] is False
    assert divergent["divergences"] == 2

    hidden_failure = fit_diagnostics(_fake_fit(bad_unreported_parameter=True))
    assert hidden_failure["ok"] is False
    assert hidden_failure["max_r_hat"] > 1.05


def test_one_chain_withholds_truth_interval_marks():
    fit = _fake_fit(chains=1)
    diagnostics = fit_diagnostics(fit)
    table = interval_coverage_table(fit, PAATDriftCoefficients(), AngleParameters())

    assert diagnostics["ok"] is False
    assert "at least two chains" in diagnostics["issues"][0]
    assert table["r_hat"].isna().all()
    assert table["covered"].isna().all()


def test_off_support_congruency_intercept_is_not_scored_for_recovery():
    fit = _fake_fit(include_off_support_term=True)
    table = interval_coverage_table(fit, PAATDriftCoefficients(), AngleParameters())
    row = table.set_index("term").loc["v_congruency_code"]
    assert not bool(row["interpretation_supported"])
    assert pd.isna(row["covered"])


@pytest.mark.slow
def test_hssm_builds_stock_angle_models_for_individual_and_group_data():
    individual, _ = fit_table(simulate_players(profile="conference", n_players=1, seed=6))
    group, _ = fit_table(simulate_players(profile="conference", n_players=2, seed=7))
    individual_model = build_paat_model(individual, hierarchical=False)
    group_model = build_paat_model(group, hierarchical=True)

    assert individual_model.model_name == group_model.model_name == "angle"
    assert individual_model.loglik_kind == group_model.loglik_kind == "approx_differentiable"
    assert not individual_model.has_lapse
    assert not group_model.has_lapse
    individual_names = set(individual_model.pymc_model.named_vars)
    group_names = set(group_model.pymc_model.named_vars)
    assert {"a", "z", "t", "theta"} <= individual_names
    assert {"a", "z", "t", "theta"} <= group_names
    assert not {
        name
        for name in group_names
        if name.startswith(("a_", "z_", "t_", "theta_")) and "participant_id" in name
    }
    assert any(name.startswith("v_") and "participant_id" in name for name in group_names)


@pytest.mark.slow
@pytest.mark.parametrize(("n_players", "hierarchical"), [(1, False), (2, True)])
def test_hssm_samples_individual_and_drift_hierarchical_models(n_players, hierarchical):
    table, exclusions = fit_table(
        simulate_players(profile="conference", n_players=n_players, seed=31 + n_players)
    )
    assert exclusions["timeouts"] == 0

    fit = fit_paat(
        table,
        hierarchical=hierarchical,
        draws=10,
        tune=10,
        chains=1,
        seed=13,
    )

    assert fit.hierarchical is hierarchical
    assert fit.model.model_name == "angle"
    assert fit.idata.posterior.sizes["chain"] == 1
    assert fit.idata.posterior.sizes["draw"] == 10
    summary = posterior_table(fit)
    assert {"v_Intercept", "a", "z", "t", "theta"} <= set(summary["term"])
    predictive = sample_posterior_predictive(fit, draws=2)
    assert "/posterior_predictive" in predictive.groups
    if not hierarchical:
        import hssm

        assert fit.model.plot_predictive(
            dt=predictive,
            col="congruency_code",
            uncertainty="band",
        ) is not None
        assert hssm.plotting.plot_model_cartoon(
            fit.model,
            dt=predictive,
            obs=0,
            n_samples=2,
            uncertainty="band",
            random_state=2,
        ) is not None
        assert hssm.plotting.plot_quantile_probability(
            fit.model,
            cond="congruency_code",
            dt=predictive,
            n_samples=2,
        ) is not None
