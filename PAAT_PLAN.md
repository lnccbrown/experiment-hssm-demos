# Probabilistic Approach-Avoidance Task (PAAT) notebook

The canonical implementation lives on `feature/paat-task`, based on the
updated canonical PST/runtime commit `2e7f5f6`. Before migration, the complete dirty
legacy worktree was preserved as local snapshot commit `fc2bf1f` in the old
`experiment-simulator` repository. The legacy worktree has not been reset or
deleted.

Reference: Cheng et al. (2026), *The Probabilistic Approach-Avoidance Task: A
novel paradigm for assessing approach-avoidance conflict under uncertainty*,
Psychonomic Bulletin & Review. [Paper](https://doi.org/10.3758/s13423-026-02885-9)
· [OSF materials](https://osf.io/zx3cy/)

## Scope and product flow

This is a local Marimo conference notebook with the same delivery model as the
coherence and PST notebooks. Its guided flow is:

1. **Play** a short, two-block PAAT with deterministic replay of all random
   events.
2. **Explore** risky choices and response times by motivational congruency and
   relative evidence.
3. **Simulate** participants with the stock `ssm-simulators` linear-collapse
   (`angle`) model.
4. **Fit** the same built-in `angle` model with HSSM's packaged differentiable
   LAN likelihood.
5. **Check** sampler diagnostics, a single-simulation truth overlay,
   posterior predictions, the model cartoon, and a quantile-probability plot
   through native HSSM tools.

Deployment, hosting, analytics/consent, QR codes, and the shared accessibility
and mobile pass are outside this consolidation.

## Scientific contract

- Each option contains a reward wheel and an aversive wheel. The risky option
  always has greater aversive probability; reward differences determine
  congruency.
- `response = +1` is the risky option and `response = -1` is the safe option,
  matching the upper/lower boundary convention used by simulation and fitting.
- Predictors use one frozen reference scale across every profile, seed, and
  participant. The reference distribution crosses the four evidence
  magnitudes uniformly and uses the paper's Study 2 60/36 congruency ratio.
  Realized schedules are never standardized separately.
- Drift uses the published fixed-effect structure:

  `v ~ (rel_reward_z + rel_aversive_z) * congruency_code`

- The default numerical slopes retain the reported Study 1 qualitative
  reward/aversive pattern on the demo's fixed scale. They are not presented as
  a calibrated generative parameter vector. The intercept, threshold, starting
  point, non-decision time, and collapse parameter are illustrative.
- Probability-pair means are not fixed at 0.5. Available means vary for
  differences 0.2, 0.4, and 0.6; only the 0.8 difference has the single
  `(0.1, 0.9)` pair with mean 0.5.
- The public demo uses a labeled neutral aversive placeholder. Deploying actual
  aversive stimuli requires an appropriate protocol, ethics review, consent,
  safeguarding, and licensed stimulus handling.

### Regression support and interpretation

Congruency is defined by the sign of relative reward. Consequently, congruency
and relative reward are strongly associated and their observed reward supports
are disjoint. No trial occurs at `rel_reward_z == 0` (the closest designed
value is about 0.141 on the frozen scale). The raw congruency intercept is
therefore an off-support extrapolation, not an interpretable condition
difference. It remains in the published formula for completeness, but the UI
labels it as unsupported and the truth overlay withholds a coverage judgment.
Reward and aversive *slopes within condition* and their interactions are the
meaningful regression contrasts.

The drift heatmap evaluates only exact reward/aversive predictor cells present
for each congruency group. Blank cells are unsupported; the figure does not
fill a global rectangle with extrapolated predictions.

### Deadline omissions

The task has a six-second deadline. `ssm-simulators` can generate stock
`angle` decisions beyond that deadline, and the adapter records them as
explicit no-response rows. The installed HSSM stock `angle` LAN models observed
choices and RTs; it does not provide the right-censored omission likelihood
needed to treat those deadline rows as ordinary observations.

Every fit reports its omission count and is explicitly conditional on observed
responses. The notebook blocks fitting when omissions exceed 5% either overall
or for any participant. This threshold is a conservative demo safeguard, not a
statistical correction for censoring; lower nonzero omission rates remain a
stated limitation.

### Hierarchy

The optional teaching hierarchy adds participant intercepts and slopes to
drift `v` only. `a`, `z`, `t`, and `theta` remain pooled population parameters.
The simulation's heterogeneity control varies the same drift coefficients only,
so the advertised fit and data-generating process stay aligned.
This is deliberate: it focuses the short synthetic design on its primary
regression and avoids presenting a weakly identified all-parameter hierarchy as
a replication of the published HDDM analysis. It is not the full published
hierarchical specification. Research use would require a preregistered model,
adequate trials and participants, richer participant effects where justified,
and recovery checks for that exact design.

The notebook uses HSSM's `safe` prior settings rather than the paper's full
hierarchical prior specification. A truth-in-interval mark describes one
simulated dataset only; it is not a parameter-recovery study or evidence about
frequentist coverage.

## Named designs

The implementation does not claim that either 96-trial profile is a canonical
replication. Both are count-matched synthetic designs using this notebook's
evidence grid.

| Profile | Trials | Incongruent / congruent | Use |
|---|---:|---:|---|
| `conference` | 24 | 12 / 12 | Fast pedagogical interaction with compressed animation |
| `published_study2` | 96 | 60 / 36 | Synthetic grid matching totals reported in the final paper |
| `osf_study2` | 96 | 66 / 30 | Synthetic grid matching totals in complete released Study 2 sessions |

Every block has equal risky-left and risky-right counts. Wheel selection and
both possible option spins are pre-drawn from the schedule seed. Each browser
run receives a nonce, and its completed export must match the exact iframe
window, session, Python schedule, order, cardinality, immutable fields, and
response window before outcomes are reconstructed.

Exact research replication requires the intended canonical schedule and an
appropriate aversive-stimulus protocol; neither count-matched profile supplies
those pieces.

## Ecosystem boundary

PAAT is not registered as a task-specific model in HSSM or `ssm-simulators`,
and this notebook does not pretend otherwise. Its task-specific layer ends at
the experimental schedule, rendering/export validation, and regression mapping
from task predictors to drift.

- **Simulation:** `ssms.basic_simulators.Simulator(model="angle")` receives
  trial-wise `[v, a, z, t, theta]` values after bounds validation.
- **Inference:** `hssm.HSSM(model="angle",
  loglik_kind="approx_differentiable")` uses HSSM's packaged `angle.onnx`
  likelihood. The fixed lapse/outlier mixture is disabled (`p_outlier=0`) so
  the likelihood matches stock simulations that contain no contaminant
  process. The executable verifier checks parameter order, choices, and LAN
  bounds against the installed `ssm-simulators` model configuration.
- **Checks:** posterior predictive simulation and model plots use native HSSM
  APIs. R-hat, bulk/tail ESS, and divergences are surfaced; truth-in-interval
  marks are withheld when diagnostics fail.
- **Experiment shell:** PAAT reuses the canonical sandboxed jsPsych runner and
  its nonce- and iframe-source-bound Marimo result bridge.

No PAAT-specific sequential-sampling process or likelihood is implemented. No
upstream model change is required for the supported observed-response analysis:
the scientific model is the stock `angle` process with a PAAT regression on
`v`. A future claim to fit deadline-censored omissions would require upstream
likelihood support first.

HSSM 0.5 currently emits Bambi compatibility deprecation warnings internally
during model construction. PAAT code does not access the deprecated
`distributional_components`/`components` APIs; the verifier captures and
attributes those upstream warnings rather than hiding them.

## Validation status (2026-09-29)

- Fast Python and Node suite: **102 passed, 5 slow tests deselected**.
- Complete marked-slow suite: **5 passed, 102 fast tests deselected**. This
  includes real individual and v-only hierarchical PAAT sampling, native
  posterior prediction, posterior summaries, predictive plotting, model
  cartoon, and quantile-probability plotting, plus the canonical PST slow fits.
- Strict Marimo validation: pass with no output.
- Scientific/implementation gate: **12/12 passed** with `--hssm`.
- Ruff and `git diff --check`: pass.

The slow suite emits known warnings from HSSM 0.5's use of deprecated Bambi
compatibility properties, NumPyro chain scheduling on one CPU device, and
Numba/plotting internals. None originates from direct deprecated API use in
PAAT code, and none is a failed diagnostic assertion.

## Files

```text
experiments/paat_demo/
  paat_app.py          guided Marimo notebook
  paat_timeline.py     jsPsych instructions, trials, breaks, and finish screen
  paat_export.py       session/schedule-bound browser rows to validated responses
  paat_app.css
schemas/tasks/paat.py  named designs, schedules, fixed scale, response contract
renderers/paat_wheels/ wheel UI and browser session controller
observers/paat_ssm.py  drift mapping, bounds, and stock angle simulation
analysis/paat_*.py     summaries, fitting, diagnostics, and support-aware plots
tests/test_paat_*.py   Python contract and integration tests
tests/js/paat_session.test.mjs
verify_review.py       executable scientific gate with nonzero failure status
```

## Verification contract

`verify_review.py` is a gate, not a report-only audit. It exits nonzero if any
declared claim fails. `--hssm` additionally constructs individual and
hierarchical models and checks the installed HSSM/SSMS angle contract.

```bash
uv run pytest
uv run pytest -m slow
uv run marimo check --strict experiments/paat_demo/paat_app.py
uv run python verify_review.py --hssm
```

The slow tests are intentionally computationally expensive. They smoke-test
real sampling for both individual and v-only hierarchical PAAT models in
addition to the canonical PST slow tests.
