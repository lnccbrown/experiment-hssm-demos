# Probabilistic Selection Task (PST; Frank, Seeberger & O'Reilly, 2004): pairs, schedules, scoring.
# Pure Python and seeded, so the browser task and simulated players can share one schedule.
# Used by actors/pst_rl.py (simulated players) and experiments/pst_demo (timeline, export).

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd

# Glyph pools. Each participant gets a random subset: 6 task symbols plus 2 practice symbols.
HIRAGANA_GLYPHS = ("あ", "ぬ", "き", "ほ", "ゆ", "を", "ね", "む", "ふ", "や", "わ", "そ")
SHAPE_GLYPHS = ("circle", "triangle", "square", "diamond", "pentagon", "hexagon", "star", "ring")
SYMBOL_SETS: dict[str, tuple[str, ...]] = {"hiragana": HIRAGANA_GLYPHS, "shapes": SHAPE_GLYPHS}

# Shared behavioral-data rules. The browser, simulator, export adapter, and fitting pipeline all
# use these values so that an omission or anticipation has the same meaning everywhere.
PST_RESPONSE_DEADLINE_S = 4.0
PST_MIN_RT_S = 0.2

# Response table shared by the browser export and simulated players. ``response`` is
# accuracy-coded: +1 = chose the better symbol (upper DDM boundary), -1 = chose the worse one.
PST_RESPONSE_COLUMNS = (
    "participant_id",
    "phase",
    "block",
    "trial",
    "pair",
    "pair_id",
    "left_symbol",
    "right_symbol",
    "better_symbol",
    "worse_symbol",
    "chosen_symbol",
    "choice_side",
    "response",
    "feedback",
    "rt",
    "timed_out",
    "anticipated",
    "response_method",
)


@dataclass(frozen=True)
class PSTPair:
    """A training pair: two symbol indices and each symbol's reward probability."""

    name: str
    better: int
    worse: int
    p_better: float
    p_worse: float


DEFAULT_PAIRS = (
    PSTPair("AB", 0, 1, 0.8, 0.2),
    PSTPair("CD", 2, 3, 0.7, 0.3),
    PSTPair("EF", 4, 5, 0.6, 0.4),
)

# Original learning criterion: accuracy per pair within a block (Frank et al., 2004).
DEFAULT_CRITERION = (("AB", 0.65), ("CD", 0.60), ("EF", 0.50))

PRACTICE_REWARD_PROBS = (0.9, 0.1)


def symbol_label(symbol: int) -> str:
    """Role letter for a task symbol index (0 -> "A")."""
    return chr(ord("A") + int(symbol))


@dataclass(frozen=True)
class PSTConfig:
    """Task structure. Presentation timing (response window, feedback) lives in the timeline."""

    pairs: tuple[PSTPair, ...] = DEFAULT_PAIRS
    trials_per_pair: int = 20  # per block: 3 pairs x 20 = 60-trial blocks
    min_blocks: int = 2
    max_blocks: int = 4
    criterion: tuple[tuple[str, float], ...] = DEFAULT_CRITERION
    practice_trials: int = 6
    test_reps: int = 6  # Frank et al. supplement: six presentations of each test pairing
    symbol_set: str = "hiragana"

    def __post_init__(self) -> None:
        symbols = sorted(s for pair in self.pairs for s in (pair.better, pair.worse))
        if symbols != list(range(2 * len(self.pairs))):
            raise ValueError("pairs must use each symbol 0..2n-1 exactly once")
        if not 1 <= self.min_blocks <= self.max_blocks:
            raise ValueError("need 1 <= min_blocks <= max_blocks")
        if self.trials_per_pair < 1 or self.practice_trials < 0 or self.test_reps < 0:
            raise ValueError("trial counts must be positive (practice/test may be 0)")
        if self.symbol_set not in SYMBOL_SETS:
            raise ValueError(f"symbol_set must be one of {sorted(SYMBOL_SETS)}")
        if len(SYMBOL_SETS[self.symbol_set]) < self.n_symbols + 2:
            raise ValueError("symbol set has too few glyphs for the task and practice symbols")

    @property
    def n_symbols(self) -> int:
        return 2 * len(self.pairs)

    @property
    def block_length(self) -> int:
        return self.trials_per_pair * len(self.pairs)

    @property
    def practice_pair(self) -> PSTPair:
        return PSTPair("practice", self.n_symbols, self.n_symbols + 1, *PRACTICE_REWARD_PROBS)

    def reward_probabilities(self) -> dict[int, float]:
        """Reward probability of each task symbol."""
        probs: dict[int, float] = {}
        for pair in self.pairs:
            probs[pair.better] = pair.p_better
            probs[pair.worse] = pair.p_worse
        return probs


# The schema default remains the existing, paper-adjacent teaching schedule.  The app starts with
# the shorter profile so a first-time player can reach the modeling steps without completing more
# than 60 learning choices.  Keeping the profiles here (rather than only in the notebook) makes the
# duration contract testable and reusable by simulation.
PST_GAME_PROFILES: dict[str, dict[str, int]] = {
    "quick": {
        "trials_per_pair": 10,
        "min_blocks": 1,
        "max_blocks": 2,
        "practice_trials": 4,
        "test_reps": 2,
    },
    "thorough": {
        "trials_per_pair": 20,
        "min_blocks": 2,
        "max_blocks": 4,
        "practice_trials": 6,
        "test_reps": 6,
    },
}


def pst_config_for_profile(profile: str, *, symbol_set: str = "hiragana") -> PSTConfig:
    """Return one of the app's named duration profiles.

    ``quick`` is the app default; ``thorough`` preserves the contribution's original schedule.
    Both use the same pairs, reward probabilities, stopping criterion, timing, and model.
    """
    if profile not in PST_GAME_PROFILES:
        raise ValueError(f"profile must be one of {sorted(PST_GAME_PROFILES)}, got {profile!r}")
    return PSTConfig(symbol_set=symbol_set, **PST_GAME_PROFILES[profile])


@dataclass(frozen=True)
class PSTTrialSpec:
    """One scheduled trial. Symbols are indices; glyphs are looked up on the schedule."""

    phase: str  # "practice", "learning" or "test"
    block: int  # 1-based within the phase
    trial: int  # 1-based within the phase; the learning-phase count drives the boundary a(t)
    pair: str  # "AB", "practice", or a test pairing such as "AD"
    pair_id: int  # index into PSTConfig.pairs; -1 for practice and novel test pairings
    left_symbol: int
    right_symbol: int
    better_symbol: int  # the symbol with the higher reward probability
    reward_left: int | None  # pre-sampled outcome if the left symbol is chosen; None = no feedback
    reward_right: int | None

    @property
    def worse_symbol(self) -> int:
        return self.right_symbol if self.left_symbol == self.better_symbol else self.left_symbol

    def reward_for(self, symbol: int) -> int | None:
        """Pre-sampled outcome for choosing ``symbol`` (None when the phase gives no feedback)."""
        if symbol == self.left_symbol:
            return self.reward_left
        if symbol == self.right_symbol:
            return self.reward_right
        raise ValueError(f"symbol {symbol} is not shown on trial {self.trial} ({self.pair})")


@dataclass(frozen=True)
class PSTSchedule:
    """One participant's session: glyphs, practice, all learning blocks (up to max) and test."""

    config: PSTConfig
    seed: int | None
    glyphs: tuple[str, ...]  # glyph per symbol index: task symbols first, then 2 practice symbols
    practice: tuple[PSTTrialSpec, ...]
    learning: tuple[PSTTrialSpec, ...]
    test: tuple[PSTTrialSpec, ...]

    def learning_blocks(self, n_blocks: int) -> tuple[PSTTrialSpec, ...]:
        """The first ``n_blocks`` learning blocks (fixed-length runs, e.g. simulated players)."""
        if not 1 <= n_blocks <= self.config.max_blocks:
            raise ValueError(f"n_blocks must be between 1 and {self.config.max_blocks}")
        return self.learning[: n_blocks * self.config.block_length]

    def to_frame(self, phase: str = "learning") -> pd.DataFrame:
        trials = {"practice": self.practice, "learning": self.learning, "test": self.test}[phase]
        rows = [
            {
                **asdict(t),
                "worse_symbol": t.worse_symbol,
                "left_glyph": self.glyphs[t.left_symbol],
                "right_glyph": self.glyphs[t.right_symbol],
            }
            for t in trials
        ]
        return pd.DataFrame(rows)


def make_schedule(config: PSTConfig | None = None, seed: int | None = None) -> PSTSchedule:
    """Draw glyphs, practice, every learning block up to ``max_blocks``, and the test phase.

    Rewards are pre-sampled for both symbols on every trial, so the outcome of any choice is
    fixed in advance and the same schedule can be replayed by a human or a simulated player.
    """
    config = config or PSTConfig()
    glyph_rng, practice_rng, learning_rng, test_rng = np.random.default_rng(seed).spawn(4)
    pool = SYMBOL_SETS[config.symbol_set]
    order = glyph_rng.permutation(len(pool))[: config.n_symbols + 2]
    return PSTSchedule(
        config=config,
        seed=seed,
        glyphs=tuple(pool[i] for i in order),
        practice=_practice_trials(config, practice_rng),
        learning=_learning_trials(config, learning_rng),
        test=_test_trials(config, test_rng),
    )


def _balanced_sides(n: int, rng: np.random.Generator) -> np.ndarray:
    """Booleans (True = better/first symbol on the left), half each; a random side gets any odd one."""
    n_left = n // 2 + int(n % 2 == 1 and rng.random() < 0.5)
    return rng.permutation(np.arange(n) < n_left)


def _feedback_trial(
    phase: str,
    block: int,
    trial: int,
    pair: PSTPair,
    pair_id: int,
    better_on_left: bool,
    rng: np.random.Generator,
) -> PSTTrialSpec:
    reward_better = int(rng.random() < pair.p_better)
    reward_worse = int(rng.random() < pair.p_worse)
    if better_on_left:
        left, right, reward_left, reward_right = pair.better, pair.worse, reward_better, reward_worse
    else:
        left, right, reward_left, reward_right = pair.worse, pair.better, reward_worse, reward_better
    return PSTTrialSpec(phase, block, trial, pair.name, pair_id, left, right, pair.better, reward_left, reward_right)


def _practice_trials(config: PSTConfig, rng: np.random.Generator) -> tuple[PSTTrialSpec, ...]:
    sides = _balanced_sides(config.practice_trials, rng)
    return tuple(
        _feedback_trial("practice", 1, i, config.practice_pair, -1, bool(left), rng)
        for i, left in enumerate(sides, start=1)
    )


def _learning_trials(config: PSTConfig, rng: np.random.Generator) -> tuple[PSTTrialSpec, ...]:
    trials: list[PSTTrialSpec] = []
    n_pairs = len(config.pairs)
    for block in range(1, config.max_blocks + 1):
        order = rng.permutation(np.repeat(np.arange(n_pairs), config.trials_per_pair))
        sides = {k: iter(_balanced_sides(config.trials_per_pair, rng)) for k in range(n_pairs)}
        for k in order:
            pair = config.pairs[int(k)]
            trials.append(
                _feedback_trial("learning", block, len(trials) + 1, pair, int(k), bool(next(sides[int(k)])), rng)
            )
    return tuple(trials)


def _test_trials(config: PSTConfig, rng: np.random.Generator) -> tuple[PSTTrialSpec, ...]:
    """All pairings of the task symbols, no feedback; better = higher reward probability."""
    probs = config.reward_probabilities()
    training_ids = {frozenset((p.better, p.worse)): k for k, p in enumerate(config.pairs)}
    specs = []
    for first, second in combinations(sorted(probs), 2):
        better = first if probs[first] >= probs[second] else second
        name = symbol_label(first) + symbol_label(second)
        pair_id = training_ids.get(frozenset((first, second)), -1)
        for first_on_left in _balanced_sides(config.test_reps, rng):
            left, right = (first, second) if first_on_left else (second, first)
            specs.append((name, pair_id, left, right, better))
    return tuple(
        PSTTrialSpec("test", 1, n, name, pair_id, left, right, better, None, None)
        for n, (name, pair_id, left, right, better) in enumerate(
            (specs[i] for i in rng.permutation(len(specs))), start=1
        )
    )


def score_choice(trial: PSTTrialSpec, choice_side: str | None) -> dict[str, object]:
    """Accuracy-code one choice: response +1 = better symbol (upper boundary), -1 = worse.

    ``choice_side`` None means no response: no feedback, and nothing to learn from.
    """
    if choice_side is None:
        return {"chosen_symbol": None, "response": None, "feedback": None, "timed_out": True}
    if choice_side not in ("left", "right"):
        raise ValueError(f"choice_side must be 'left', 'right' or None, got {choice_side!r}")
    chosen = trial.left_symbol if choice_side == "left" else trial.right_symbol
    return {
        "chosen_symbol": chosen,
        "response": 1 if chosen == trial.better_symbol else -1,
        "feedback": trial.reward_for(chosen),
        "timed_out": False,
    }


def block_accuracy(block: pd.DataFrame) -> dict[str, float]:
    """Proportion of better choices per pair; omissions and anticipations count as incorrect."""
    anticipated = block.get("anticipated", pd.Series(False, index=block.index)).astype(bool)
    chose_better = block["response"].eq(1) & ~anticipated
    return {str(pair): float(v) for pair, v in chose_better.groupby(block["pair"]).mean().items()}


def should_stop_learning(block: int, accuracy: Mapping[str, float], config: PSTConfig) -> bool:
    """Adaptive length: stop at ``max_blocks``, or once ``min_blocks`` are done and the
    block just finished met every pair's criterion."""
    if block >= config.max_blocks:
        return True
    return block >= config.min_blocks and all(accuracy.get(pair, 0.0) >= level for pair, level in config.criterion)


def classify_test_pair(pair: str) -> str | None:
    """Classic test-phase readouts: 'choose_A' (A vs C/D/E/F) and 'avoid_B' (B vs C/D/E/F)."""
    has_a, has_b = "A" in pair, "B" in pair
    if has_a and not has_b:
        return "choose_A"
    if has_b and not has_a:
        return "avoid_B"
    return None
