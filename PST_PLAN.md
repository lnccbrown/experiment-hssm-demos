# Probabilistic Selection Task: implemented design and scientific scope

Branch `feature/pst-task`, reviewed 2026-09-29.

The marimo app is a shortened, interactive teaching variant of the Probabilistic Selection Task (PST). Its learning model follows the RL-DDM equations in Pedersen, Frank & Biele (2017), and its software path deliberately uses the existing HSSM/ssm-simulators RLSSM tooling.

## End-to-end path

1. Python creates a seeded schedule, including the outcome that would follow either choice.
2. A sandboxed jsPsych iframe runs the task and returns rows over a nonce- and source-bound channel.
3. Python validates every returned trial against that exact schedule, re-scores the response, and
   verifies that a finished adaptive session ended at a permissible block and contains every enabled phase.
4. `PSTLearner` and `PSTEnvironment` define one `ssms.rl.ModelConfig` used by both simulation and fitting.
5. `ssms.rl.Simulator` generates virtual players. A thin task-rule wrapper turns total RTs over four
   seconds and RTs below 0.2 seconds into native omission sentinels before a generative learning update;
   output metadata distinguishes deadline omissions from anticipations and retains an anticipation's choice/RT.
6. `hssm.rl.RLSSMConfig.from_ssms_model(...)` translates the same model into a stock `hssm.RLSSM` fit.
7. HSSM's built-in differentiable DDM neural likelihood (LAN) supplies the likelihood. There is no locally registered DDM likelihood.
8. Posterior predictive data come from `Simulator(...).simulate(mode="ppc", observed_data=...)`, including replicated choices, RTs, deadline omissions and anticipations.

The former custom analytical/JAX DDM registration and custom posterior-prediction implementation were removed. `analysis/jax_ddm.py` and `register_exact_ddm()` no longer exist.

## Task

- Learning pairs: AB 80/20, CD 70/30 and EF 60/40.
- Sides are balanced within each pair/block and presentation order is shuffled.
- The browser uses the original accuracy thresholds after the configured minimum number of blocks.
- The app defaults to a **quick** profile: one-to-two 30-choice blocks (10 presentations per pair),
  four practice choices, and the all-pairs final round disabled. This is intentionally convenient but
  data-poor; a six-parameter individual fit can be weakly identified and prior-sensitive.
- A **thorough** profile preserves the contribution's original schedule: two-to-four 60-choice blocks
  (20 presentations per pair), six practice choices, and six presentations of each of the 15 pairings
  when the optional feedback-free final round is enabled.
- Responses have a four-second total-RT deadline.
- Responses faster than 0.2 seconds are anticipations. Browser and simulator both withhold their
  outcome and learning update, so removing them later does not erase an experienced learning event.
- Key and click input are both supported and retained in `response_method`.

The all-pair final round and frozen-value test simulator are app choices. The RL-DDM is fitted only to the learning phase.

## Learning and decision model

For the pair shown on learning trial `k`:

```text
v(k) = m × (V_better − V_worse)
a(k) = bb × (k / 10)^bp
V_chosen ← V_chosen + eta_sign × (reward − V_chosen)
```

`eta_pos` is used for a positive prediction error and `eta_neg` otherwise. The simpler selectable model uses one `eta` and a fixed `a`. Initial values are zero, response +1 denotes the better symbol / upper boundary, and `z` is fixed at 0.5. The local DDM parameter `a` is the distance from zero to either bound, one half of the full Wiener boundary separation used in the cited JAGS model; the paper-derived `bb` presets are converted accordingly.

HSSM's built-in DDM LAN is approximate and has finite validated support. Trial-wise `v` is clipped to [-3, 3] and `a` to [0.3, 2.5] in both Python and JAX learner paths. This is explicit in the notebook because it changes the unconstrained scientific model whenever a trajectory reaches a limit.

## Fitting

- One participant: intercept-only parameter formulas.
- Multiple participants: a hierarchical model with `(1 | participant_id)` for every free learning and decision parameter.
- `link_settings="log_logit"` keeps population and participant parameters within HSSM's registered bounds.
- App-specific regularizing priors are specified on the link scale. They are not Pedersen et al.'s JAGS hyperpriors.
- The upper bound for non-decision time is below the fastest retained RT across all participants.
- The fitted model has no lapse/outlier mixture (`p_outlier=0`); valid but atypical RTs remain in the fit.
- RLSSM requires a balanced panel. After omissions and anticipations are removed, each participant is truncated to the smallest retained count; the notebook reports that loss. A participant with no valid rows causes the fit to stop rather than being silently discarded.
- Simulator and exported data keep original trial numbers, so the changing boundary still uses elapsed task trial rather than the compact RLSSM row index.

The current HSSM RLSSM likelihood cannot model the task's right-censored deadline observations. Omissions are excluded, and the resulting likelihood does not correct for deadline truncation. This is a limitation, not a conditional-likelihood adjustment.

## Posterior checking and interpretation

The native SSMS PPC conditions learning-state updates on observed choices and outcomes, then generates a new decision response and RT at every retained trial. It is a conditional decision-layer check, not a fully generative replication of reward and learning histories. The app plots 90% across-draw bands for both choice proportions and mean RTs and reports replicated deadline-omission and anticipation rates.

Substantive panels are gated on:

- at least four chains;
- maximum rank-normalized r-hat no greater than 1.01;
- bulk and tail ESS of at least 400 total and 100 per chain;
- no divergences;
- BFMI of at least 0.30 when available;
- no reported maximum-tree-depth hits.

The session summary is deliberately model- and session-conditional. An uncertain boundary trend is labelled unresolved, not unchanged. Non-decision time is not interpreted when click and key responses are mixed. Near-ceiling data trigger an identifiability warning about learning rates versus value sensitivity.

The recovery panel compares a single simulated dataset with its known generating centre. It is an internal consistency illustration only. It does not establish estimator bias, interval calibration, parameter discriminability or general recoverability; those require repeated simulations across the parameter space. The paper-derived presets are group summaries/grid settings, not known truths for an individual user.

## Data integrity and browser isolation

- Every exported schedule field must match the Python schedule.
- Feedback and anticipation flags are independently recomputed.
- Duplicate, missing, reordered and over-deadline response rows are rejected. The app additionally
  checks complete practice/test phases and the exact adaptive learning endpoint before accepting a result.
- Every game receives a fresh unpredictable result-session identifier and iframe id.
- The bridge accepts only a matching nonce from that iframe's `contentWindow`.
- The iframe uses `sandbox="allow-scripts"` without `allow-same-origin`.
- Starting a new game creates an empty new bridge, so prior results cannot remain attached to the new schedule.

## Key files

```text
schemas/tasks/pst.py                 task, schedule, scoring and shared timing rules
actors/pst_rl.py                     SSMS learner/environment, simulation and latent replay
analysis/pst_analysis.py             preprocessing and descriptive summaries
analysis/pst_fit.py                  stock HSSM RLSSM fit, diagnostics and native PPC
analysis/pst_plots.py                observed and PPC charts
experiments/pst_demo/pst_timeline.py browser timeline
experiments/pst_demo/pst_export.py   validated browser export
experiments/pst_demo/pst_app.py      marimo application
runtime/                              generic iframe runner and secured result bridge
```

## Dependencies and verification

The supported API floor is HSSM 0.4 with ssm-simulators 0.13.2. On 2026-09-29 the lockfile resolves
HSSM 0.5.0 and ssm-simulators 0.14.0, which are also the current PyPI releases. The built-in DDM
LAN's ONNX artifact is fetched from Hugging Face on first use when it is not already cached.

```bash
uv run pytest            # fast Python tests plus Node browser-script tests
uv run pytest -m slow    # real single-participant and hierarchical HSSM fits + native PPC
uv run marimo check --strict experiments/pst_demo/pst_app.py
```

The slow suite is intentionally computationally intensive; it tests integration, not a formal recovery study.

### Consolidation validation (2026-09-29)

- `uv lock --check`: passed.
- `uv run pytest -q`: **52 passed, 2 deselected** in 6.47 s.
- Final review rerun, `.venv/bin/pytest -q`: **52 passed, 2 deselected** in 6.75 s.
- `uv run --isolated --with 'hssm==0.4.0' --with 'ssm-simulators==0.13.2' pytest -q`:
  **52 passed, 2 deselected** in 6.89 s, exercising the declared API floors.
- `uv run pytest -q -m slow`: **2 passed, 52 deselected** in 339.27 s. This ran both the real
  hierarchical fit/native PPC and the real single-participant fixed-boundary fit. The 16 warnings
  were upstream HSSM/Bambi deprecations and sequential-chain notices on a one-device host.
- `uv run marimo check --strict experiments/pst_demo/pst_app.py`: passed.
- `git diff --check`: passed.

The app server was launched successfully at its local URL, but a reliable manual Play → Simulate →
Fit → Check traversal was not completed: the available browser-control surface repeatedly requested
macOS automation permission and then blocked on Chrome's first-run profile UI. No manual browser
behavior is claimed from that attempt; the browser export, isolation, task semantics, fitting and PPC
paths are covered by the fast and marked integration tests above.
