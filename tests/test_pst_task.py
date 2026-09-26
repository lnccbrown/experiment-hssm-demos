import numpy as np
import pandas as pd
import pytest

from schemas.tasks.pst import (
    SYMBOL_SETS,
    PSTConfig,
    PSTTrialSpec,
    block_accuracy,
    classify_test_pair,
    make_schedule,
    score_choice,
    should_stop_learning,
)


def test_learning_blocks_are_balanced_and_numbered():
    config = PSTConfig()
    frame = make_schedule(config, seed=1).to_frame("learning")

    assert len(frame) == config.max_blocks * config.block_length
    assert frame["trial"].tolist() == list(range(1, len(frame) + 1))
    assert (frame.groupby(["block", "pair"]).size() == config.trials_per_pair).all()
    better_left = (frame["left_symbol"] == frame["better_symbol"]).groupby([frame["block"], frame["pair"]]).sum()
    assert (better_left == config.trials_per_pair // 2).all()


def test_presampled_rewards_follow_probabilities():
    frames = [make_schedule(seed=s).to_frame("learning") for s in range(40)]
    frame = pd.concat(frames, ignore_index=True)
    better_left = frame["left_symbol"] == frame["better_symbol"]
    reward_better = np.where(better_left, frame["reward_left"], frame["reward_right"])
    reward_worse = np.where(better_left, frame["reward_right"], frame["reward_left"])

    for pair in PSTConfig().pairs:
        rows = frame["pair"] == pair.name  # 3,200 trials per pair -> SE ~ 0.008
        assert reward_better[rows].mean() == pytest.approx(pair.p_better, abs=0.03)
        assert reward_worse[rows].mean() == pytest.approx(pair.p_worse, abs=0.03)


def test_seed_reproduces_schedule():
    assert make_schedule(seed=7) == make_schedule(seed=7)
    assert make_schedule(seed=7).learning != make_schedule(seed=8).learning


@pytest.mark.parametrize("symbol_set", sorted(SYMBOL_SETS))
def test_glyphs_are_distinct_members_of_the_set(symbol_set):
    schedule = make_schedule(PSTConfig(symbol_set=symbol_set), seed=3)
    assert len(schedule.glyphs) == len(set(schedule.glyphs)) == 8
    assert set(schedule.glyphs) <= set(SYMBOL_SETS[symbol_set])
    practice = schedule.to_frame("practice")
    assert set(practice["left_symbol"]) | set(practice["right_symbol"]) == {6, 7}


def test_test_phase_covers_every_pairing_without_feedback():
    frame = make_schedule(PSTConfig(test_reps=2), seed=2).to_frame("test")

    assert len(frame) == 15 * 2 and frame["pair"].nunique() == 15
    assert (frame.groupby("pair")["left_symbol"].nunique() == 2).all()  # each order shown once
    assert frame["reward_left"].isna().all() and frame["reward_right"].isna().all()
    better = frame.set_index("pair")["better_symbol"].groupby(level=0).first()
    assert better["AC"] == 0 and better["DF"] == 5 and better["BD"] == 3
    categories = frame["pair"].map(classify_test_pair).value_counts()
    assert categories["choose_A"] == categories["avoid_B"] == 8


def test_default_final_round_uses_six_presentations_per_pairing():
    frame = make_schedule(seed=22).to_frame("test")
    assert len(frame) == 15 * 6
    categories = frame["pair"].map(classify_test_pair).value_counts()
    assert categories["choose_A"] == categories["avoid_B"] == 24


def test_score_choice_is_accuracy_coded():
    trial = PSTTrialSpec("learning", 1, 1, "AB", 0, left_symbol=1, right_symbol=0, better_symbol=0,
                         reward_left=1, reward_right=0)
    assert score_choice(trial, "right") == {"chosen_symbol": 0, "response": 1, "feedback": 0, "timed_out": False}
    assert score_choice(trial, "left") == {"chosen_symbol": 1, "response": -1, "feedback": 1, "timed_out": False}
    assert score_choice(trial, None)["timed_out"] is True


def test_adaptive_stopping_has_floor_and_ceiling():
    config = PSTConfig(min_blocks=2, max_blocks=4)
    passed = {"AB": 0.70, "CD": 0.65, "EF": 0.55}
    failed = {"AB": 0.90, "CD": 0.90, "EF": 0.40}

    assert not should_stop_learning(1, passed, config)
    assert should_stop_learning(2, passed, config)
    assert not should_stop_learning(3, failed, config)
    assert should_stop_learning(4, failed, config)


def test_block_accuracy_counts_timeouts_and_anticipations_as_incorrect():
    block = pd.DataFrame(
        {
            "pair": ["AB", "AB", "CD", "CD"],
            "response": [1, None, 1, -1],
            "anticipated": [True, False, False, False],
        }
    )
    assert block_accuracy(block) == {"AB": 0.0, "CD": 0.5}
