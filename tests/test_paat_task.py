import numpy as np
import pandas as pd
import pytest

from schemas.tasks.paat import (
    DESIGN_PROFILES,
    PAAT_PREDICTOR_SCALE,
    RELATIVE_LEVELS,
    make_schedule,
    score_choice,
)


@pytest.mark.parametrize(
    ("profile", "incongruent", "congruent"),
    [
        ("conference", 12, 12),
        ("published_study2", 60, 36),
        ("osf_study2", 66, 30),
    ],
)
def test_named_profiles_have_their_declared_counts_and_balanced_sides(profile, incongruent, congruent):
    schedule = make_schedule(profile, seed=1)
    frame = schedule.to_frame()

    assert len(frame) == DESIGN_PROFILES[profile].n_trials
    assert frame["congruency"].value_counts().to_dict() == {
        "incongruent": incongruent,
        "congruent": congruent,
    }
    for _, block in frame.groupby("block"):
        assert block["risky_side"].value_counts().to_dict() == {"left": len(block) // 2, "right": len(block) // 2}


def test_probabilities_define_risk_congruency_and_use_the_shared_reference_scale():
    frame = make_schedule("published_study2", seed=2).to_frame()

    assert (frame["p_aversive_risky"] > frame["p_aversive_safe"]).all()
    assert (frame.loc[frame["congruency"] == "incongruent", "rel_reward"] > 0).all()
    assert (frame.loc[frame["congruency"] == "congruent", "rel_reward"] < 0).all()
    np.testing.assert_allclose(
        frame["rel_reward_z"],
        PAAT_PREDICTOR_SCALE.reward_z(frame["rel_reward"]),
    )
    np.testing.assert_allclose(
        frame["rel_aversive_z"],
        PAAT_PREDICTOR_SCALE.aversive_z(frame["rel_aversive"]),
    )
    assert set(frame["rel_reward"].abs()) == set(RELATIVE_LEVELS)
    assert set(frame["rel_aversive"]) == set(RELATIVE_LEVELS)


def test_predictor_scale_is_identical_across_profiles_participants_and_seeds():
    frames = [
        make_schedule(profile, seed=seed).to_frame()
        for profile in DESIGN_PROFILES
        for seed in (1, 19)
    ]
    combined = pd.concat(frames, ignore_index=True)
    assert combined.groupby("rel_reward")["rel_reward_z"].nunique().eq(1).all()
    assert combined.groupby("rel_aversive")["rel_aversive_z"].nunique().eq(1).all()


def test_probability_pair_means_vary_except_for_the_point_eight_difference():
    frames = [make_schedule("published_study2", seed=seed).to_frame() for seed in range(6)]
    data = pd.concat(frames, ignore_index=True)
    pairs = pd.concat(
        [
            pd.DataFrame(
                {
                    "difference": data["rel_reward"].abs().round(6),
                    "pair_mean": (data["p_reward_risky"] + data["p_reward_safe"]) / 2,
                }
            ),
            pd.DataFrame(
                {
                    "difference": data["rel_aversive"].round(6),
                    "pair_mean": (data["p_aversive_risky"] + data["p_aversive_safe"]) / 2,
                }
            ),
        ],
        ignore_index=True,
    )
    distinct_means = pairs.groupby("difference")["pair_mean"].nunique()
    assert distinct_means.loc[0.8] == 1
    assert pairs.loc[pairs["difference"].eq(0.8), "pair_mean"].iloc[0] == pytest.approx(0.5)
    assert (distinct_means.drop(0.8) > 1).all()


def test_study2_profiles_are_labeled_as_count_matched_synthetic_designs():
    assert DESIGN_PROFILES["published_study2"].label.startswith("Count-matched")
    assert DESIGN_PROFILES["osf_study2"].label.startswith("Count-matched")


def test_seed_reproduces_the_complete_schedule():
    first = make_schedule("conference", seed=42).to_frame()
    second = make_schedule("conference", seed=42).to_frame()
    other = make_schedule("conference", seed=43).to_frame()

    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_choice_scoring_is_risky_positive_independent_of_side():
    for trial in make_schedule("conference", seed=3).trials[:8]:
        risky = score_choice(trial, trial.risky_side)
        safe = score_choice(trial, trial.safe_side)
        assert risky["chosen_option"] == "risky" and risky["response"] == 1
        assert safe["chosen_option"] == "safe" and safe["response"] == -1
        expected_risky = int(trial.spin_risky < trial.probability("risky", trial.selected_wheel))
        expected_safe = int(trial.spin_safe < trial.probability("safe", trial.selected_wheel))
        assert risky["outcome"] == expected_risky
        assert safe["outcome"] == expected_safe


def test_timeout_forces_the_aversive_outcome():
    scored = score_choice(make_schedule(seed=4).trials[0], None)
    assert scored == {
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


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown PAAT profile"):
        make_schedule("not-a-design", seed=1)
