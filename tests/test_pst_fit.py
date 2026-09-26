from types import SimpleNamespace

import numpy as np
import pytest

from analysis.pst_analysis import fit_rows, fit_table, learning_curves
from analysis.pst_fit import (
    _t_upper_bound,
    fit_diagnostics,
    fit_pst,
    personal_readout,
    posterior_predictive,
    posterior_predictive_curves,
    posterior_table,
)
from actors.pst_rl import PRESETS, PSTLearner, simulate_players

MIDDLE = PRESETS["Recovery study, middle values"]


def test_t_bound_uses_fastest_observation_across_participants():
    import pandas as pd

    table = pd.DataFrame({"participant_id": [0, 0, 1, 1], "rt": [0.31, 0.40, 0.72, 0.80]})
    assert _t_upper_bound(table) == pytest.approx(0.309)


def test_posterior_predictive_uses_native_ssms_ppc_for_choices_and_rts():
    observed = simulate_players(MIDDLE, n_blocks=1, seed=5)
    table, _ = fit_table(observed)

    class FakeFit:
        learner = PSTLearner()
        n_samples = 1

        def __init__(self):
            self.table = table

        def participant_draws(self, sample_ids):
            return {name: np.full((len(sample_ids), 1), value) for name, value in {**MIDDLE, "z": 0.5}.items()}

    ppc = posterior_predictive(FakeFit(), n_draws=1, seed=9)
    assert len(ppc) == len(table)
    assert set(ppc["response"].unique()) <= {-999, -1, 1}
    assert (ppc.loc[ppc["rt"] > 0, "rt"] > 0).all()
    assert not np.array_equal(ppc["response"].to_numpy(), table["response"].to_numpy())


def test_personal_readout_is_model_and_session_conditional():
    n = 200
    draws = {
        "eta_pos": np.full(n, 0.3),
        "eta_neg": np.full(n, 0.05),
        "m": np.full(n, 2.5),
        "bp": np.full(n, -0.1),
        "t": np.full(n, 0.31),
    }
    items = personal_readout(SimpleNamespace(draws=lambda: draws, learner=PSTLearner()))
    assert [item["topic"] for item in items] == [
        "Positive vs negative learning",
        "Value sensitivity m",
        "Decision boundary over trials",
        "Non-decision time t",
    ]
    assert [item["value"] for item in items] == ["Wins", "2.5", "Decreased", "0.31 s"]
    assert all(item["text"].endswith(".") for item in items)
    assert all("shortened session" in item["text"] and "regularizing priors" in item["text"] for item in items)

    mixed = personal_readout(
        SimpleNamespace(draws=lambda: draws, learner=PSTLearner()), mixed_response_methods=True
    )
    assert mixed[-1]["value"] == "Not interpreted"

    simple = PSTLearner(dual_learning_rates=False, boundary="fixed")
    items = personal_readout(SimpleNamespace(draws=lambda: draws, learner=simple))
    assert [item["topic"] for item in items] == ["Value sensitivity m", "Non-decision time t"]


@pytest.mark.slow
def test_hierarchical_fit_uses_builtin_lan_and_native_ppc():
    df = simulate_players(MIDDLE, n_players=3, n_blocks=4, seed=6)
    table, _ = fit_table(df)
    fit = fit_pst(table, draws=300, tune=300, chains=2, seed=1)
    summary = posterior_table(fit, MIDDLE)

    assert fit.hierarchical
    assert fit.model.model_config.decision_process_loglik_kind == "approx_differentiable"
    assert list(summary["symbol"]) == ["eta_pos", "eta_neg", "m", "bb", "bp", "t"]
    assert np.isfinite(summary[["estimate", "low", "high", "mcse_mean", "ess_bulk"]]).all().all()
    rows = fit_rows(df)
    ppc = posterior_predictive_curves(fit, rows, n_draws=12)
    assert len(ppc["choice"]) == len(learning_curves(df, rows=rows))
    assert ((ppc["choice"]["low"] <= ppc["choice"]["p_better"]) &
            (ppc["choice"]["p_better"] <= ppc["choice"]["high"])).all()
    assert ((ppc["rt"]["low"] <= ppc["rt"]["rt_mean"]) &
            (ppc["rt"]["rt_mean"] <= ppc["rt"]["high"])).all()
    diagnostics = fit_diagnostics(fit)
    assert diagnostics["chains"] == 2 and diagnostics["divergences"] >= 0
    assert not diagnostics["ok"] and "four chains" in " ".join(diagnostics["issues"])


@pytest.mark.slow
def test_single_rate_fixed_boundary_variant_fits():
    learner = PSTLearner(dual_learning_rates=False, boundary="fixed")
    truth = {"eta": 0.15, "m": 3.0, "a": 1.0, "t": 0.3}
    table, _ = fit_table(simulate_players(truth, learner=learner, n_blocks=2, seed=7))
    fit = fit_pst(table, learner=learner, draws=150, tune=150, seed=2)
    assert not fit.hierarchical
    assert list(posterior_table(fit, truth)["symbol"]) == ["eta", "m", "a", "t"]
    assert fit_diagnostics(fit)["chains"] == 4
