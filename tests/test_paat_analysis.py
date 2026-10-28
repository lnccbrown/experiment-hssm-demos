import pytest

from analysis.paat_analysis import (
    MAX_TIMEOUT_RATE_FOR_UNCENSORED_FIT,
    condition_summary,
    data_fingerprint,
    drift_grid,
    fit_table,
    regression_design_diagnostics,
    require_defensible_uncensored_fit,
    session_stats,
)
from observers.paat_ssm import PAATDriftCoefficients, simulate_players


def test_analysis_summarizes_choices_and_produces_hssm_columns():
    df = simulate_players(profile="conference", n_players=2, seed=4)
    table, excluded = fit_table(df)
    summary = condition_summary(df)
    stats = session_stats(df)

    assert list(table.columns) == [
        "participant_id",
        "rt",
        "response",
        "rel_reward_z",
        "rel_aversive_z",
        "congruency_code",
    ]
    assert excluded == {
        "kept": 48,
        "total": 48,
        "timeouts": 0,
        "timeout_rate": 0.0,
        "participant_timeout_rates": {0: 0.0, 1: 0.0},
        "max_participant_timeout_rate": 0.0,
        "participants_above_timeout_limit": 0,
        "high_timeout_rate": False,
        "anticipations": 0,
    }
    assert set(summary["congruency"]) == {"congruent", "incongruent"}
    assert 0 <= stats["p_risky"] <= 1
    assert stats["median_rt"] > 0


def test_fit_table_drops_timeouts_and_responses_at_or_below_250_ms():
    df = simulate_players(profile="conference", n_players=1, seed=5)
    df.loc[0, ["timed_out", "response", "rt"]] = [True, None, None]
    df.loc[1, "rt"] = 0.25
    table, excluded = fit_table(df)
    assert len(table) == 22
    assert excluded["kept"] == 22
    assert excluded["timeouts"] == 1
    assert excluded["anticipations"] == 1
    assert excluded["timeout_rate"] == pytest.approx(1 / 24)
    assert excluded["max_participant_timeout_rate"] == pytest.approx(1 / 24)
    assert excluded["high_timeout_rate"] is False


def test_fit_table_flags_high_unmodeled_timeout_rates():
    df = simulate_players(profile="conference", n_players=1, seed=8)
    n_timeout = int(MAX_TIMEOUT_RATE_FOR_UNCENSORED_FIT * len(df)) + 1
    df.loc[: n_timeout - 1, ["timed_out", "response", "rt"]] = [True, None, None]
    _, excluded = fit_table(df)
    assert excluded["timeouts"] == n_timeout
    assert excluded["high_timeout_rate"] is True
    with pytest.raises(ValueError, match="right-censored"):
        require_defensible_uncensored_fit(excluded)


def test_fit_table_blocks_a_high_omission_participant_even_below_group_limit():
    df = simulate_players(profile="conference", n_players=3, seed=18)
    df.loc[df["participant_id"].eq(0).head(2).index, ["timed_out", "response", "rt"]] = [
        True,
        None,
        None,
    ]
    _, excluded = fit_table(df)
    assert excluded["timeout_rate"] < MAX_TIMEOUT_RATE_FOR_UNCENSORED_FIT
    assert excluded["max_participant_timeout_rate"] == pytest.approx(2 / 24)
    assert excluded["participants_above_timeout_limit"] == 1
    assert excluded["high_timeout_rate"] is True


def test_design_diagnostics_and_drift_grid_make_support_explicit():
    design = simulate_players(profile="published_study2", n_players=2, seed=9)
    diagnostics = regression_design_diagnostics(design)
    assert abs(diagnostics["reward_congruency_correlation"]) > 0.8
    assert diagnostics["reward_supports_overlap"] is False
    assert diagnostics["zero_reward_observed"] is False
    assert diagnostics["congruency_main_effect_supported"] is False

    grid = drift_grid(PAATDriftCoefficients(), design)
    for code in (0, 1):
        observed = {
            tuple(row)
            for row in design.loc[
                design["congruency_code"].eq(code),
                ["rel_reward_z", "rel_aversive_z"],
            ].drop_duplicates().to_numpy()
        }
        plotted = {
            tuple(row)
            for row in grid.loc[
                grid["congruency_code"].eq(code),
                ["rel_reward_z", "rel_aversive_z"],
            ].to_numpy()
        }
        assert plotted == observed


def test_fit_fingerprint_tracks_exact_data_and_order():
    table, _ = fit_table(simulate_players(profile="conference", n_players=1, seed=6))
    same = table.copy()
    changed = table.copy()
    changed.loc[0, "rt"] += 0.001
    reordered = table.iloc[::-1].reset_index(drop=True)

    assert data_fingerprint(table) == data_fingerprint(same)
    assert data_fingerprint(table) != data_fingerprint(changed)
    assert data_fingerprint(table) != data_fingerprint(reordered)
