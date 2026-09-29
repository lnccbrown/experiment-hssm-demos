import numpy as np
import pytest

from observers.paat_ssm import (
    AngleParameters,
    PAATDriftCoefficients,
    angle_parameter_bounds,
    drift_coefficients_from_mapping,
    simulate_players,
    simulate_schedule,
    trialwise_parameters,
    validate_angle_parameters,
)
from schemas.tasks.paat import make_schedule


def test_drift_regression_expands_main_effects_and_interactions():
    coefficients = PAATDriftCoefficients(
        intercept=0.1,
        reward=0.6,
        aversive=-0.3,
        congruency=-0.2,
        reward_x_congruency=-0.1,
        aversive_x_congruency=0.25,
    )
    assert coefficients.drift(1.0, 2.0, 0) == pytest.approx(0.1 + 0.6 - 0.6)
    assert coefficients.drift(1.0, 2.0, 1) == pytest.approx(0.1 + 0.6 - 0.6 - 0.2 - 0.1 + 0.5)


def test_mapping_constructor_preserves_the_drift_intercept_and_requires_every_term():
    values = {
        "intercept": 0.42,
        "reward": 0.6,
        "aversive": -0.3,
        "congruency": -0.2,
        "reward_x_congruency": -0.1,
        "aversive_x_congruency": 0.25,
        "a": 1.2,
    }
    coefficients = drift_coefficients_from_mapping(values)
    assert coefficients.intercept == pytest.approx(0.42)
    assert coefficients.reward == pytest.approx(0.6)
    with pytest.raises(ValueError, match="missing PAAT drift coefficient"):
        drift_coefficients_from_mapping({"reward": 0.6})


def test_trialwise_matrix_has_stock_angle_parameter_order():
    schedule = make_schedule(seed=1)
    angle = AngleParameters(a=1.2, z=0.55, t=0.25, theta=0.12)
    theta = trialwise_parameters(schedule, PAATDriftCoefficients(), angle)
    assert list(theta.columns) == ["v", "a", "z", "t", "theta"]
    assert len(theta) == 24
    assert theta[["a", "z", "t", "theta"]].nunique().to_dict() == {"a": 1, "z": 1, "t": 1, "theta": 1}
    validate_angle_parameters(theta)


def test_angle_bounds_come_from_the_installed_stock_model():
    assert angle_parameter_bounds() == {
        "v": (-3.0, 3.0),
        "a": (0.3, 3.0),
        "z": (0.1, 0.9),
        "t": (0.001, 2.0),
        "theta": (-0.1, 1.3),
    }


def test_out_of_lan_drift_is_rejected_before_simulation():
    schedule = make_schedule("published_study2", seed=8)
    coefficients = PAATDriftCoefficients(
        reward=1.5,
        aversive=1.5,
        congruency=1.5,
        reward_x_congruency=1.0,
        aversive_x_congruency=1.0,
    )
    theta = trialwise_parameters(schedule, coefficients, AngleParameters())
    assert (theta["v"].abs() > 3).any()
    with pytest.raises(ValueError, match="outside the installed angle model bounds"):
        simulate_schedule(schedule, coefficients=coefficients, seed=9)


def test_adapter_matches_a_direct_stock_simulator_call():
    from ssms.basic_simulators import Simulator

    schedule = make_schedule(seed=2)
    coefficients = PAATDriftCoefficients()
    angle = AngleParameters()
    theta = trialwise_parameters(schedule, coefficients, angle)
    direct = Simulator(model="angle").simulate(
        theta=theta,
        n_samples=1,
        delta_t=0.001,
        max_t=6.0,
        random_state=9,
        return_option="full",
        n_threads=1,
    )
    df = simulate_schedule(schedule, coefficients=coefficients, angle=angle, seed=9)
    choices = np.asarray(direct["choices"]).reshape(24, -1)[:, 0].astype(int)
    rts = np.asarray(direct["rts"]).reshape(24, -1)[:, 0]

    np.testing.assert_array_equal(df["response"].to_numpy(int), choices)
    np.testing.assert_allclose(df["rt"], rts)


def test_simulated_cohort_uses_independent_full_schedules():
    df = simulate_players(profile="published_study2", n_players=3, seed=3)
    assert len(df) == 3 * 96
    assert df.groupby("participant_id").size().eq(96).all()
    assert set(df["response"].unique()) <= {-1, 1}
    assert (df["rt"] > 0).all()
    assert {"true_v", "true_a", "true_z", "true_t", "true_theta"} <= set(df.columns)


def test_individual_difference_spread_varies_drift_but_not_pooled_angle_parameters():
    df = simulate_players(profile="published_study2", n_players=3, spread=0.3, seed=12)
    assert df.groupby("participant_id")["true_v"].mean().nunique() > 1
    assert df[["true_a", "true_z", "true_t", "true_theta"]].nunique().eq(1).all()
