import numpy as np
import pytest

from analysis.pst_analysis import fit_table
from actors.pst_rl import (
    CONTEXT_FIELDS,
    LAN_DDM_BOUNDS,
    PRESETS,
    PSTLearner,
    final_values,
    make_model,
    replay_latents,
    simulate_players,
    simulate_test_phase,
    vary_players,
)
from schemas.tasks.pst import make_schedule

MIDDLE = PRESETS["Recovery study, middle values"]


def test_learning_rate_depends_on_prediction_error_sign():
    learner = PSTLearner()
    params = {"eta_pos": 0.5, "eta_neg": 0.1, "m": 2.0, "bb": 1.0, "bp": 0.0}
    pair = {"better_symbol": 0, "worse_symbol": 1, "trial": 1}

    state = learner.update_python(learner.init_state(), params, {**pair, "choice": 1, "feedback": 1.0})
    assert state["values"][0] == pytest.approx(0.5)  # 0 + 0.5 * (1 - 0)
    state = learner.update_python(state, params, {**pair, "choice": 1, "feedback": 0.0})
    assert state["values"][0] == pytest.approx(0.45)  # 0.5 + 0.1 * (0 - 0.5)
    state = learner.update_python(state, params, {**pair, "choice": 0, "feedback": 1.0})
    assert state["values"][1] == pytest.approx(0.5)  # the worse symbol was chosen
    assert state["values"][0] == pytest.approx(0.45)


def test_drift_uses_value_difference_and_boundary_follows_power_law():
    state = {"values": np.array([0.6, 0.2, 0.0, 0.0, 0.0, 0.0])}
    params = {"eta_pos": 0.1, "eta_neg": 0.1, "m": 2.5, "bb": 1.0, "bp": -0.2}
    out = PSTLearner().compute_python(state, params, {"better_symbol": 0, "worse_symbol": 1, "trial": 100})
    assert out["v"] == pytest.approx(1.0)  # 2.5 * (0.6 - 0.2)
    assert out["a"] == pytest.approx(10 ** -0.2)  # 1.0 * (100 / 10) ** -0.2


def test_computed_decision_parameters_are_clipped_to_lan_support():
    state = {"values": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    params = {"eta_pos": 0.1, "eta_neg": 0.1, "m": 10.0, "bb": 2.5, "bp": 0.5}
    out = PSTLearner().compute_python(state, params, {"better_symbol": 0, "worse_symbol": 1, "trial": 360})
    assert out == {"v": LAN_DDM_BOUNDS["v"][1], "a": LAN_DDM_BOUNDS["a"][1]}

    learner = PSTLearner()
    jax_state = learner.init_jax_state()
    jax_state["values"] = jax_state["values"].at[0].set(1.0)
    jax_out = learner.compute_jax(jax_state, params, {"better_symbol": 0, "worse_symbol": 1, "trial": 360})
    assert float(jax_out["v"]) == LAN_DDM_BOUNDS["v"][1]
    assert float(jax_out["a"]) == LAN_DDM_BOUNDS["a"][1]


def test_single_rate_fixed_boundary_variant():
    learner = PSTLearner(dual_learning_rates=False, boundary="fixed")
    assert learner.free_params == ["eta", "m"] and learner.computed_params == ["v"]
    assert set(make_model(learner=learner).list_params) == {"eta", "m", "a", "z", "t"}
    df = simulate_players({"eta": 0.2, "m": 3.0, "a": 1.0, "t": 0.3}, learner=learner, n_blocks=1, seed=0)
    assert len(df) == 60 and df["rt"].min() > 0.3


def test_python_and_jax_backends_replay_the_same_drift_and_boundary():
    schedule = make_schedule(seed=4)
    df = simulate_players(MIDDLE, schedule=schedule, n_blocks=2, seed=1)
    model = make_model(schedule, n_blocks=2)
    replays = {}
    for backend in ("python", "jax"):
        assembled = model.assemble(backend=backend)
        fields = assembled.get_participant_input_fields()
        columns = [np.full(len(df), MIDDLE[f]) if f in MIDDLE else df[f].to_numpy(float) for f in fields]
        replays[backend] = assembled.assemble_participant_fn(output="dict")(np.column_stack(columns))
    for key in ("v", "a"):
        np.testing.assert_allclose(np.asarray(replays["python"][key]), np.asarray(replays["jax"][key]), atol=1e-5)


def test_feedback_comes_from_the_schedule_and_response_is_accuracy_coded():
    schedule = make_schedule(seed=5)
    df = simulate_players(MIDDLE, schedule=schedule, n_blocks=1, seed=2)
    planned = schedule.to_frame("learning").iloc[: len(df)]

    answered = ~df["timed_out"]
    expected = np.where(df.loc[answered, "choice_side"] == "left", planned.loc[answered, "reward_left"], planned.loc[answered, "reward_right"])
    assert (df.loc[answered, "feedback"].to_numpy() == expected).all()
    assert ((df.loc[answered, "response"] == 1) ==
            (df.loc[answered, "chosen_symbol"] == df.loc[answered, "better_symbol"])).all()
    assert (df["trial"].to_numpy() == planned["trial"].to_numpy()).all()


def test_simulated_players_pass_ssms_validation_and_learn():
    df = simulate_players(MIDDLE, n_players=6, n_blocks=4, seed=3)
    model = make_model(n_blocks=4)
    table, _ = fit_table(df)
    model.validate_data(table[["participant_id", "rt", "response", *CONTEXT_FIELDS]]).raise_for_errors()

    accuracy_by_block = df["response"].eq(1).groupby(df["block"]).mean()
    assert accuracy_by_block.iloc[-1] > accuracy_by_block.iloc[0] + 0.1
    assert df["rt"].min() > MIDDLE["t"]
    assert df.groupby("participant_id").size().eq(240).all()


def test_total_rt_deadline_creates_omissions_without_learning_updates():
    slow = {**MIDDLE, "m": 0.1, "bb": 2.5, "bp": 0.0, "t": 1.0}
    df = simulate_players(slow, n_blocks=1, seed=31)
    assert df["timed_out"].any()
    assert df.loc[df["timed_out"], ["rt", "response", "feedback"]].isna().all().all()
    assert (df.loc[~df["timed_out"], "rt"] <= 4.0).all()
    assert df["response_method"].dropna().eq("simulated").all()


def test_replay_starts_at_initial_values_and_matches_final_values():
    df = simulate_players(MIDDLE, n_blocks=1, seed=7)
    latents = replay_latents(df, MIDDLE)
    assert latents.loc[0, "v"] == 0.0 and latents.loc[0, "a"] == pytest.approx(MIDDLE["bb"])
    last = latents.iloc[-1]
    # values before the last trial, updated once more, equal the final values
    values = final_values(df, MIDDLE)
    changed = int(df.iloc[-1]["chosen_symbol"])
    unchanged = [i for i in range(6) if i != changed]
    np.testing.assert_allclose(last[[f"V_{c}" for c in "ABCDEF"]].to_numpy(float)[unchanged], values[unchanged])


def test_vary_players_scatters_parameters_within_their_ranges():
    assert vary_players(MIDDLE, n_players=3, spread=0.0) == [MIDDLE] * 3
    players = vary_players(MIDDLE, n_players=200, spread=0.3, seed=1)
    for name in ("eta_pos", "eta_neg"):
        values = np.array([p[name] for p in players])
        assert ((values > 0) & (values < 1)).all() and values.std() > 0
    assert all(p["m"] > 0 and p["bb"] > 0 and p["t"] > 0 for p in players)
    assert np.median([p["m"] for p in players]) == pytest.approx(MIDDLE["m"], rel=0.1)

    simple = {"eta": 0.2, "m": 2.0, "a": 1.0, "t": 0.3}
    simple_players = vary_players(simple, n_players=50, spread=0.3, seed=2)
    assert np.std([p["eta"] for p in simple_players]) > 0


def test_test_phase_follows_final_values_without_feedback():
    schedule = make_schedule(seed=8)
    params = {**MIDDLE, "m": 20.0}  # very value-sensitive player
    df = simulate_test_phase([0.8, 0.2, 0.7, 0.3, 0.6, 0.4], params, schedule, last_learning_trial=240, seed=1)
    assert len(df) == len(schedule.test)
    assert df["feedback"].isna().all() and (df["phase"] == "test").all()
    assert df["response"].eq(1).mean() > 0.9
