import numpy as np
import pandas as pd

from analysis.pst_analysis import (
    FIT_COLUMNS,
    MIN_RT_S,
    fit_table,
    learning_curves,
    near_ceiling,
    rt_curves,
    session_stats,
    signed_rts,
    final_round_summary,
)
from actors.pst_rl import PRESETS, final_values, simulate_players, simulate_test_phase
from schemas.tasks.pst import make_schedule

MIDDLE = PRESETS["Recovery study, middle values"]


def test_learning_curves_bin_each_pair_separately():
    df = simulate_players(MIDDLE, n_blocks=2, seed=1)  # 40 presentations of each pair
    curves = learning_curves(df, bin_size=10)

    assert set(curves["pair"]) == {"AB", "CD", "EF"}
    assert curves.groupby("pair")["bin"].max().eq(4).all()
    assert curves["n"].sum() == (~df["timed_out"]).sum()
    assert curves["trials"].min() == 5.0
    assert set(rt_curves(df)["pair"]) == {"AB", "CD", "EF"}
    signed = signed_rts(df)
    answered = df[~df["timed_out"]]
    assert ((signed["signed_rt"] < 0) == (answered["response"].to_numpy() == -1)).all()


def test_fit_table_drops_timeouts_and_anticipations():
    df = simulate_players(MIDDLE, n_blocks=1, seed=2)
    df["response"] = df["response"].astype(float)
    df.loc[0, ["timed_out", "rt", "response"]] = [True, np.nan, np.nan]
    df.loc[1, "rt"] = 0.15
    table, info = fit_table(df)

    assert info == {
        "kept": 58,
        "timeouts": 1,
        "anticipations": 1,
        "balancing": 0,
        "response_methods": ["simulated"],
        "mixed_response_methods": False,
    }
    assert list(table.columns) == FIT_COLUMNS
    assert table["rt"].min() >= MIN_RT_S


def test_session_stats_and_test_phase_readouts():
    schedule = make_schedule(seed=3)
    learning = simulate_players(MIDDLE, schedule=schedule, n_blocks=2, seed=3)
    test = simulate_test_phase(final_values(learning, MIDDLE), MIDDLE, schedule, last_learning_trial=120, seed=3)
    both = pd.concat([learning, test], ignore_index=True)

    stats = session_stats(both)
    assert stats["points"] == int(learning["feedback"].eq(1).sum())
    assert stats["blocks"] == 2 and stats["trials"] == 120
    assert stats["timeouts"] == int(learning["timed_out"].sum())
    assert 0.0 <= stats["choose_A"] <= 1.0 and 0.0 <= stats["avoid_B"] <= 1.0
    summary = final_round_summary(both)
    assert set(summary["readout"]) == {"choose_A", "avoid_B"}
    classified = test["pair"].str.contains("A") ^ test["pair"].str.contains("B")
    assert summary["n"].sum() == int((classified & ~test["timed_out"]).sum())


def test_fit_table_balances_participants_after_different_exclusions():
    fast = {**MIDDLE, "m": 8.0, "bb": 0.3, "t": 0.2}
    df = simulate_players(fast, n_players=2, n_blocks=1, seed=12)
    df.loc[(df["participant_id"] == 0) & (df["trial"] == 1), ["timed_out", "rt", "response"]] = [
        True,
        np.nan,
        np.nan,
    ]
    table, info = fit_table(df)
    assert table.groupby("participant_id").size().tolist() == [59, 59]
    assert info["balancing"] == 1
    assert table.groupby("participant_id")["trial_id"].apply(list).map(lambda x: x == list(range(59))).all()


def test_fit_table_does_not_silently_drop_a_participant_with_no_valid_rows():
    df = simulate_players(MIDDLE, n_players=2, n_blocks=1, seed=13)
    mask = df["participant_id"].eq(0)
    df.loc[mask, "anticipated"] = True
    table, info = fit_table(df)
    assert table.empty
    assert info["balancing"] == 60


def test_final_round_summary_excludes_anticipations():
    schedule = make_schedule(seed=30)
    learning = simulate_players(MIDDLE, schedule=schedule, n_blocks=1, seed=30)
    test = simulate_test_phase(final_values(learning, MIDDLE), MIDDLE, schedule, last_learning_trial=60, seed=30)
    eligible = test["pair"].str.contains("A") ^ test["pair"].str.contains("B")
    index = test[eligible & ~test["timed_out"]].index[0]
    before = final_round_summary(test)["n"].sum()
    test.loc[index, "anticipated"] = True
    assert final_round_summary(test)["n"].sum() == before - 1


def test_near_ceiling_flags_only_near_perfect_players():
    strong = simulate_players({**MIDDLE, "eta_pos": 0.3, "eta_neg": 0.3, "m": 10.0}, n_blocks=4, seed=4)
    assert near_ceiling(strong)
    assert not near_ceiling(simulate_players(PRESETS["ADHD, off medication (Table 2)"], n_blocks=2, seed=4))
