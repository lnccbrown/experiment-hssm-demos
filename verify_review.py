"""Executable scientific and implementation gate for the PAAT demo.

Every section states the accepted contract and records a pass/fail result. The
process exits nonzero if any declared gate fails. Pass ``--hssm`` to include
the slower installed-ecosystem construction and API checks.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paat_analysis import (
    drift_grid,
    fit_table,
    regression_design_diagnostics,
    require_defensible_uncensored_fit,
)
from analysis.paat_fit import OFF_SUPPORT_TERMS
from observers.paat_ssm import (
    PRESETS,
    AngleParameters,
    PAATDriftCoefficients,
    drift_coefficients_from_mapping,
    simulate_players,
    simulate_schedule,
    trialwise_parameters,
)
from schemas.tasks.paat import DESIGN_PROFILES, PAAT_PREDICTOR_SCALE, make_schedule

RESULTS: list[tuple[str, bool]] = []


def gate(number: int, name: str, ok: bool, observed: str) -> None:
    RESULTS.append((name, bool(ok)))
    print("=" * 78)
    print(f"[{number}] GATE: {name}")
    print(f"    observed: {observed}")
    print(f"    verdict : {'PASS' if ok else 'FAIL'}")


# 1. One predictor scale for every schedule and participant.
frames = [
    make_schedule(profile, seed=seed).to_frame()
    for profile in DESIGN_PROFILES
    for seed in (1, 7, 19)
]
all_designs = pd.concat(frames, ignore_index=True)
reward_consistent = bool(
    all_designs.groupby("rel_reward")["rel_reward_z"].nunique().eq(1).all()
)
aversive_consistent = bool(
    all_designs.groupby("rel_aversive")["rel_aversive_z"].nunique().eq(1).all()
)
gate(
    1,
    "predictor scaling is fixed across profiles, seeds, and participants",
    reward_consistent and aversive_consistent,
    (
        f"reward mean/sd={PAAT_PREDICTOR_SCALE.reward_mean:.3f}/"
        f"{PAAT_PREDICTOR_SCALE.reward_sd:.3f}; aversive mean/sd="
        f"{PAAT_PREDICTOR_SCALE.aversive_mean:.3f}/"
        f"{PAAT_PREDICTOR_SCALE.aversive_sd:.3f}; unique mappings="
        f"{reward_consistent}/{aversive_consistent}"
    ),
)


# 2. The paper formula's support limitation must remain explicit and ungated
# recovery must not be claimed for its congruency intercept.
diagnostics = {
    profile: regression_design_diagnostics(make_schedule(profile, seed=7).to_frame())
    for profile in DESIGN_PROFILES
}
strong = all(
    abs(float(item["reward_congruency_correlation"])) > 0.8
    for item in diagnostics.values()
)
disjoint = all(not bool(item["reward_supports_overlap"]) for item in diagnostics.values())
unsupported = all(
    not bool(item["congruency_main_effect_supported"]) for item in diagnostics.values()
)
nearest = {
    profile: float(item["nearest_abs_reward_z"])
    for profile, item in diagnostics.items()
}
gate(
    2,
    "reward/congruency collinearity and the off-support main effect are explicit",
    strong and disjoint and unsupported and "v_congruency_code" in OFF_SUPPORT_TERMS,
    (
        "|correlations|="
        f"{[round(abs(float(x['reward_congruency_correlation'])), 3) for x in diagnostics.values()]}; "
        f"disjoint={disjoint}; nearest |z|={nearest}; recovery term withheld="
        f"{'v_congruency_code' in OFF_SUPPORT_TERMS}"
    ),
)


# 3. Reward magnitude and aversive difference are independently crossed in the
# synthetic evidence grid (small finite-schedule imbalance is tolerated).
orthogonal = []
for seed in range(8):
    design = make_schedule("published_study2", seed=seed).to_frame()
    orthogonal.append(
        float(np.corrcoef(design["rel_reward"].abs(), design["rel_aversive"])[0, 1])
    )
gate(
    3,
    "the synthetic evidence-difference grid is approximately orthogonal",
    max(abs(value) for value in orthogonal) < 0.25,
    f"corr range across seeds={min(orthogonal):+.3f} to {max(orthogonal):+.3f}",
)


# 4. Pair means are deliberately variable; only a difference of .8 fixes the
# single (.1, .9) pair.
pair_frames = [
    make_schedule("published_study2", seed=seed).to_frame() for seed in range(8)
]
pair_data = pd.concat(pair_frames, ignore_index=True)
pairs = pd.concat(
    [
        pd.DataFrame(
            {
                "difference": pair_data["rel_reward"].abs().round(6),
                "pair_mean": (
                    pair_data["p_reward_risky"] + pair_data["p_reward_safe"]
                )
                / 2,
            }
        ),
        pd.DataFrame(
            {
                "difference": pair_data["rel_aversive"].round(6),
                "pair_mean": (
                    pair_data["p_aversive_risky"] + pair_data["p_aversive_safe"]
                )
                / 2,
            }
        ),
    ],
    ignore_index=True,
)
distinct_means = pairs.groupby("difference")["pair_mean"].nunique()
eight_means = pairs.loc[pairs["difference"].eq(0.8), "pair_mean"]
pair_contract = bool(
    distinct_means.loc[0.8] == 1
    and np.allclose(eight_means, 0.5)
    and (distinct_means.drop(0.8) > 1).all()
)
gate(
    4,
    "probability-pair construction has no false fixed-mean claim",
    pair_contract,
    f"distinct pair means by difference={distinct_means.to_dict()}",
)


# 5. Published Study 1 slope pattern is represented on the frozen reference
# scale; only the unreported generative parameters are illustrative.
coefficients = PAATDriftCoefficients()
slope_contract = bool(
    np.isclose(coefficients.reward, 0.63)
    and np.isclose(coefficients.aversive, -0.30)
    and np.isclose(
        coefficients.reward + coefficients.reward_x_congruency,
        0.49,
    )
    and np.isclose(
        coefficients.aversive + coefficients.aversive_x_congruency,
        -0.04,
    )
)
gate(
    5,
    "default slopes reproduce the published Study 1 qualitative pattern",
    slope_contract,
    (
        f"incongruent reward/aversive={coefficients.reward:+.2f}/"
        f"{coefficients.aversive:+.2f}; congruent="
        f"{coefficients.reward + coefficients.reward_x_congruency:+.2f}/"
        f"{coefficients.aversive + coefficients.aversive_x_congruency:+.2f}"
    ),
)


# 6. Confirm the stock simulator's deadline sentinel behavior and our adapter.
schedule = make_schedule("published_study2", seed=21)
flat = AngleParameters(a=2.5, z=0.5, t=0.30, theta=0.0)
weak = PAATDriftCoefficients(
    reward=0.05,
    aversive=-0.02,
    congruency=0.0,
    reward_x_congruency=0.0,
    aversive_x_congruency=0.0,
)
theta = trialwise_parameters(schedule, weak, flat)
from ssms.basic_simulators import Simulator  # noqa: E402

raw = Simulator(model="angle").simulate(
    theta=theta,
    n_samples=1,
    delta_t=0.001,
    max_t=6.0,
    random_state=5,
    return_option="full",
    n_threads=1,
)
raw_rts = np.asarray(raw["rts"]).reshape(len(schedule.trials), -1)[:, 0]
raw_choices = np.asarray(raw["choices"]).reshape(len(schedule.trials), -1)[:, 0]
n_over = int((raw_rts > 6.0).sum())
simulated_flat = simulate_schedule(schedule, coefficients=weak, angle=flat, seed=5)
n_flagged = int(simulated_flat["timed_out"].astype(bool).sum())
gate(
    6,
    "stock angle deadline behavior maps exactly to explicit timeout rows",
    bool(
        (raw_rts == -999.0).sum() == 0
        and np.isin(raw_choices, (-1, 1)).all()
        and n_flagged == n_over
    ),
    f"raw rt>6={n_over}; adapter timeouts={n_flagged}; raw max={raw_rts.max():.3f}s",
)


# 7. High omission settings remain explorable but cannot silently reach the
# observed-response likelihood.
_, exclusions = fit_table(simulated_flat)
guarded = False
try:
    require_defensible_uncensored_fit(exclusions)
except ValueError:
    guarded = True
gate(
    7,
    "material deadline censoring is visible and blocks uncensored fitting",
    bool(
        exclusions["timeout_rate"] > 0.05
        and exclusions["high_timeout_rate"]
        and exclusions["participants_above_timeout_limit"] == 1
        and guarded
    ),
    (
        f"timeouts={exclusions['timeouts']}/{exclusions['total']} "
        f"({exclusions['timeout_rate']:.1%}); max participant="
        f"{exclusions['max_participant_timeout_rate']:.1%}; guard={guarded}"
    ),
)


# 8. UI/config reconstruction must preserve every truth coefficient.
preset = PRESETS["Study 1 pattern"]
probe = PAATDriftCoefficients(**{**preset.coefficients.__dict__, "intercept": 0.42})
rebuilt = drift_coefficients_from_mapping(probe.__dict__)
gate(
    8,
    "drift-coefficient reconstruction preserves the intercept and truth labels",
    rebuilt == probe,
    f"requested intercept={probe.intercept}; reconstructed={rebuilt.intercept}",
)


# 9. The explanatory surface contains only cells present in the design.
support_design = make_schedule("published_study2", seed=31).to_frame()
grid = drift_grid(PAATDriftCoefficients(), support_design)
off_support = 0
for code in (0, 1):
    observed = support_design.loc[
        support_design["congruency_code"].eq(code),
        ["rel_reward_z", "rel_aversive_z"],
    ].drop_duplicates()
    panel = grid.loc[
        grid["congruency_code"].eq(code),
        ["rel_reward_z", "rel_aversive_z"],
    ]
    membership = panel.merge(
        observed,
        on=["rel_reward_z", "rel_aversive_z"],
        how="left",
        indicator=True,
    )
    off_support += int(membership["_merge"].ne("both").sum())
gate(
    9,
    "drift surfaces contain no unmarked off-support predictor region",
    off_support == 0,
    f"off-support grid coordinates={off_support}; rows={len(grid)}",
)


# 10. Public profile names and scope remain unambiguous.
labels_ok = all(
    DESIGN_PROFILES[name].label.startswith("Count-matched")
    for name in ("published_study2", "osf_study2")
)
plan = Path(__file__).with_name("PAAT_PLAN.md").read_text(encoding="utf-8").lower()
scope_ok = "count-matched synthetic" in plan and "exact research replication requires" in plan
gate(
    10,
    "96-trial profiles are labeled synthetic rather than canonical replications",
    labels_ok and scope_ok,
    f"profile labels={labels_ok}; plan scope={scope_ok}",
)


if "--hssm" in sys.argv:
    # 11. HSSM and ssms must agree on the exact stock angle contract.
    from hssm.modelconfig.angle_config import get_angle_config
    from ssms.config import model_config

    hssm_config = get_angle_config()
    likelihood = hssm_config["likelihoods"]["approx_differentiable"]
    ssms_config = model_config["angle"]
    parameter_order_ok = list(hssm_config["list_params"]) == list(ssms_config["params"])
    choices_ok = list(hssm_config["choices"]) == list(ssms_config["choices"])
    bounds_ok = all(
        tuple(likelihood["bounds"][name])
        == tuple(ssms_config["param_bounds_dict"][name])
        for name in hssm_config["list_params"]
    )
    gate(
        11,
        "installed HSSM and ssm-simulators share the stock angle contract",
        parameter_order_ok
        and choices_ok
        and bounds_ok
        and likelihood["loglik"] == "angle.onnx",
        (
            f"params={hssm_config['list_params']}; choices={hssm_config['choices']}; "
            f"artifact={likelihood['loglik']}; bounds_match={bounds_ok}"
        ),
    )

    # 12. Construct both model shapes and verify that only v receives the
    # participant hierarchy. Capture HSSM 0.5's internal Bambi compatibility
    # warnings without treating them as PAAT API usage.
    from analysis.paat_fit import build_paat_model

    individual, _ = fit_table(
        simulate_players(profile="conference", n_players=1, seed=3)
    )
    group, _ = fit_table(simulate_players(profile="conference", n_players=3, seed=4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        individual_model = build_paat_model(individual, hierarchical=False)
        group_model = build_paat_model(group, hierarchical=True)
    group_names = set(group_model.pymc_model.named_vars)
    core_group_terms = {
        name
        for name in group_names
        if name.startswith(("a_", "z_", "t_", "theta_"))
        and "participant_id" in name
    }
    v_group_terms = {
        name
        for name in group_names
        if name.startswith("v_") and "participant_id" in name
    }
    paat_source = (
        Path(__file__)
        .with_name("analysis")
        .joinpath("paat_fit.py")
        .read_text(encoding="utf-8")
    )
    deprecated_warnings = [
        str(item.message)
        for item in caught
        if "distributional_components" in str(item.message)
        or "Model.components" in str(item.message)
    ]
    direct_deprecated_access = (
        ".distributional_components" in paat_source or ".components" in paat_source
    )
    gate(
        12,
        "individual and v-only hierarchical HSSM models use no deprecated PAAT API",
        bool(
            individual_model.model_name == group_model.model_name == "angle"
            and individual_model.loglik_kind
            == group_model.loglik_kind
            == "approx_differentiable"
            and not individual_model.has_lapse
            and not group_model.has_lapse
            and not core_group_terms
            and v_group_terms
            and not direct_deprecated_access
        ),
        (
            f"core-parameter group terms={sorted(core_group_terms)}; "
            f"v group terms={len(v_group_terms)}; lapse mixture="
            f"{individual_model.has_lapse}/{group_model.has_lapse}; direct deprecated access="
            f"{direct_deprecated_access}; captured upstream warnings="
            f"{len(deprecated_warnings)}"
        ),
    )
else:
    print("=" * 78)
    print("[11-12] HSSM construction checks skipped (run with --hssm)")


print("=" * 78)
failed = [name for name, ok in RESULTS if not ok]
for name, ok in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL':<4}  {name}")
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} gates passed")
if failed:
    print(f"FAILED GATES: {failed}")
raise SystemExit(1 if failed else 0)
