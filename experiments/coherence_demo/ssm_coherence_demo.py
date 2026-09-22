import marimo

# Marimo app: motion-coherence demo with forward ``DdmActor`` (ssm-simulators DDM).
# Wires sliders, participant demo iframe, simulation, and HSSM fitting.
# Entry point: ``marimo run experiments/coherence_demo/ssm_coherence_demo.py``.

# Ensure project root is importable when running via
# `marimo run experiments/coherence_demo/ssm_coherence_demo.py`.
import sys
from pathlib import Path

def resolve_repo_paths() -> tuple[Path, Path]:
    """Locate ``experiments/coherence_demo`` and repo root (marimo-safe)."""
    for candidate in (Path.cwd(), *Path.cwd().parents):
        app_dir = candidate / "experiments" / "coherence_demo"
        if (
            app_dir.is_dir()
            and (candidate / "schemas").is_dir()
            and (candidate / "actors").is_dir()
        ):
            return app_dir, candidate
    app_dir = Path(__file__).resolve().parent
    return app_dir, app_dir.parents[2]


_APP_DIR, _PROJECT_ROOT = resolve_repo_paths()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

app = marimo.App(width="full", css_file="coherence_demo.css")


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import marimo as mo

    for candidate in (Path.cwd(), *Path.cwd().parents):
        _app_dir = candidate / "experiments" / "coherence_demo"
        if (
            _app_dir.is_dir()
            and (candidate / "schemas").is_dir()
            and (candidate / "actors").is_dir()
        ):
            app_dir, project_root = _app_dir, candidate
            break
    else:
        raise RuntimeError(
            "Could not find repo root (need schemas/ and actors/). "
            "Run marimo from the simulator project directory."
        )

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    return mo, np, pd, app_dir, project_root


@app.cell
def _(mo, project_root):
    from actors.ssm_actor import DdmActor
    from analysis.hssm_pipeline import (
        fit_hssm_model,
        summarize_behavior,
        summarize_posterior,
    )
    from analysis.plotting import behavior_summary_charts
    from renderers.motion_coherence.motion_coherence_stimulus import (
        motion_coherence_preview_iframe_html,
    )
    from runtime.embed import render_srcdoc_iframe
    from motion_coherence_export import FIT_DF_COLUMNS, motion_trials_dataframe
    from runtime.jspsych_export import create_jspsych_marimo_bridge
    from runtime.jspsych_runner import build_jspsych_runner_html
    from schemas.experimentGenerator import ExperimentGenerator
    from schemas.trial_generator import FactorTrialGenerator
    from coherence_timeline import (
        build_coherence_demo_levels,
        build_coherence_timeline,
        coherence_runner_config,
    )

    return (
        ExperimentGenerator,
        FactorTrialGenerator,
        DdmActor,
        build_coherence_demo_levels,
        build_coherence_timeline,
        build_jspsych_runner_html,
        behavior_summary_charts,
        coherence_runner_config,
        create_jspsych_marimo_bridge,
        fit_hssm_model,
        motion_coherence_preview_iframe_html,
        project_root,
        FIT_DF_COLUMNS,
        motion_trials_dataframe,
        render_srcdoc_iframe,
        summarize_behavior,
        summarize_posterior,
    )


@app.cell
def _(app_dir, mo, project_root):
    import base64

    logo_b64 = base64.b64encode(
        (project_root / "assets" / "logo" / "hssm_white.png").read_bytes()
    ).decode("ascii")
    logo_block = mo.Html(
        f'<div class="coherence-demo-logo-block">'
        f'<img alt="HSSM" src="data:image/png;base64,{logo_b64}" />'
        f"</div>"
    )
    hook_b64 = base64.b64encode(
        (app_dir / "assets" / "coherenceToDDMHook.png").read_bytes()
    ).decode("ascii")
    hook_block = mo.Html(
        '<div class="coherence-demo-hook-block">'
        '<img class="coherence-demo-hook-img" '
        'alt="Random dot motion task linked to a drift diffusion model" '
        f'src="data:image/png;base64,{hook_b64}" />'
        "</div>"
    )
    intro_title = mo.md(
        "## Demonstration of an experiment-to-HSSM pipeline for a motion coherence task with a DDM actor"
    )
    intro_body = mo.md(
        r"""
This example simulates a **binary left/right motion task** at **three coherence levels** you set with the sliders (shown side-by-side).

- **Actor**: ``DdmActor`` — forward **DDM** via ssm-simulators; signed evidence maps to drift \(v\), boundary \(a\), starting point \(z\), and non-decision time \(t\).
- **Task**: trials are generated and run in the in-browser demonstration.
- **Fit**: fits a **HSSM** DDM with drift \(v\) regressed on coherence (`stim_level`, proportion).
"""
    )
    mo.vstack([logo_block, intro_title, hook_block, intro_body], gap=0.75)
    return


@app.cell
def _():
    # Motion stimulus display (coherence app only).
    MOTION_CANVAS_WIDTH = 500
    MOTION_CANVAS_HEIGHT = 260
    MOTION_PREVIEW_WIDTH = 220
    MOTION_PREVIEW_HEIGHT = 140
    MOTION_N_DOTS = 100
    MOTION_SPEED_PX_S = 120.0
    MOTION_SEED = 42
    return (
        MOTION_CANVAS_HEIGHT,
        MOTION_CANVAS_WIDTH,
        MOTION_N_DOTS,
        MOTION_PREVIEW_HEIGHT,
        MOTION_PREVIEW_WIDTH,
        MOTION_SEED,
        MOTION_SPEED_PX_S,
    )


@app.cell
def _(FactorTrialGenerator, np):
    def motion_stimulus_to_strengths(stimulus_factors: dict[str, object]) -> list[float]:
        coherence = float(
            stimulus_factors.get("coherence", stimulus_factors.get("stim_level", 0.0))
        )
        direction = str(stimulus_factors.get("motion_direction", "right"))
        if direction == "left":
            return [coherence, 0.0]
        return [0.0, coherence]

    def make_motion_coherence_trials(
        *,
        n_trials: int,
        coherence: float,
        display_params: dict[str, object] | None = None,
        presentation_duration_ms: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> FactorTrialGenerator:
        random = rng or np.random.default_rng()
        level = float(coherence)
        base_display = dict(display_params or {})
        generator = FactorTrialGenerator()
        for d in random.choice([-1, 1], size=int(n_trials)):
            motion = "left" if d < 0 else "right"
            correct_index = int(motion == "right")
            correct_key = "ArrowRight" if correct_index else "ArrowLeft"
            generator.add_trial(
                FactorTrialGenerator.generate_trials(
                    task="motion_coherence",
                    stimulus_factors={"coherence": level, "motion_direction": motion},
                    display_params=base_display,
                    presentation_duration_ms=presentation_duration_ms,
                    correct_index=correct_index,
                    choices=["ArrowLeft", "ArrowRight"],
                    correct_key=correct_key,
                    data={
                        "stim_level": level,
                        "motion_direction": motion,
                        "correct_response": correct_key,
                        "task": "motion_coherence",
                    },
                )
            )
        return generator

    return make_motion_coherence_trials, motion_stimulus_to_strengths


@app.cell
def _(mo):
    lvl1 = mo.ui.slider(0.0, 1.0, value=0.2, step=0.05, label="Coherence A (proportion)")
    lvl2 = mo.ui.slider(0.0, 1.0, value=0.5, step=0.05, label="Coherence B (proportion)")
    lvl3 = mo.ui.slider(0.0, 1.0, value=0.8, step=0.05, label="Coherence C (proportion)")
    return lvl1, lvl2, lvl3


@app.cell
def _(mo):
    dot_lifetime_s = mo.ui.number(
        start=0.01,
        stop=10.0,
        value=0.1,
        step=0.01,
        label="Dot lifetime (s)",
    )
    return (dot_lifetime_s,)


@app.cell
def _(
    MOTION_N_DOTS,
    MOTION_PREVIEW_HEIGHT,
    MOTION_PREVIEW_WIDTH,
    MOTION_SEED,
    MOTION_SPEED_PX_S,
    dot_lifetime_s,
    lvl1,
    lvl2,
    lvl3,
    mo,
    motion_coherence_preview_iframe_html,
):
    _lifetime = max(0.01, float(dot_lifetime_s.value or 0.1))

    def preview(stim_level: float, label: str):
        html = motion_coherence_preview_iframe_html(
            float(stim_level),
            instance_label=label,
            n_dots=MOTION_N_DOTS,
            width=MOTION_PREVIEW_WIDTH,
            height=MOTION_PREVIEW_HEIGHT,
            seed=MOTION_SEED,
            speed_px_s=MOTION_SPEED_PX_S,
            dot_lifetime_s=_lifetime,
        )
        return mo.Html(html)

    panel = mo.vstack(
        [
            mo.md("### Stimulus Selection for Simulation"),
            dot_lifetime_s,
            mo.hstack(
                [
                    mo.vstack([preview(lvl1.value, "A"), lvl1], gap=0.4),
                    mo.vstack([preview(lvl2.value, "B"), lvl2], gap=0.4),
                    mo.vstack([preview(lvl3.value, "C"), lvl3], gap=0.4),
                ],
                gap=1.2,
                justify="center",
            ),
        ],
        gap=0.5,
    )
    panel
    return


@app.cell
def _(mo):
    demo_trials_per_level = mo.ui.number(
        start=1,
        stop=20,
        value=5,
        label="Trials per level (count, demonstration)",
    )
    demo_restart = mo.ui.refresh(label="Restart demo")
    return demo_restart, demo_trials_per_level


@app.cell
def _(
    MOTION_CANVAS_HEIGHT,
    MOTION_CANVAS_WIDTH,
    MOTION_N_DOTS,
    MOTION_SEED,
    MOTION_SPEED_PX_S,
    build_coherence_demo_levels,
    build_coherence_timeline,
    build_jspsych_runner_html,
    demo_restart,
    demo_trials_per_level,
    dot_lifetime_s,
    lvl1,
    lvl2,
    lvl3,
    make_motion_coherence_trials,
    mo,
    coherence_runner_config,
    render_srcdoc_iframe,
):
    _ = demo_restart.value
    _lifetime = max(0.01, float(dot_lifetime_s.value or 0.1))
    a = max(0.0, min(1.0, float(lvl1.value)))
    b = max(0.0, min(1.0, float(lvl2.value)))
    c = max(0.0, min(1.0, float(lvl3.value)))
    reps = max(1, min(20, int(demo_trials_per_level.value or 5)))
    demo_levels = build_coherence_demo_levels(a, b, c, reps_per_level=reps)
    n_motion = len(demo_levels)
    _stimulus_params = {
        "n_dots": MOTION_N_DOTS,
        "speed_px_s": MOTION_SPEED_PX_S,
        "dot_lifetime_s": _lifetime,
        "canvas_width": MOTION_CANVAS_WIDTH,
        "canvas_height": MOTION_CANVAS_HEIGHT,
        "seed": MOTION_SEED,
    }
    demo_timeline = build_coherence_timeline(
        demo_levels,
        display_params=_stimulus_params,
        presentation_duration_ms=None,
        make_motion_coherence_trials=make_motion_coherence_trials,
    )
    demo_html = build_jspsych_runner_html(
        demo_timeline,
        config=coherence_runner_config(title="Motion coherence demo"),
    )
    demo_iframe = mo.Html(
        render_srcdoc_iframe(demo_html, title="Motion coherence demo", height=520)
    )
    mo.vstack(
        [
            mo.md("### Demonstration"),
            mo.md(
                f"_{reps} trials per stim level ({n_motion} motion trials across A, B, C). "
                "Motion direction is random each trial. "
                "When you finish, the **Plots** section below updates with your results "
                "(and summary charts also appear in this window)._"
            ),
            demo_trials_per_level,
            demo_iframe,
            demo_restart,
        ],
        gap=0.5,
    )
    return


@app.cell
def _(create_jspsych_marimo_bridge):
    demo_results = create_jspsych_marimo_bridge()
    demo_results
    return demo_results,


@app.cell
def _(demo_results, motion_trials_dataframe):
    demo_df = motion_trials_dataframe(demo_results.value["rows_json"])
    return demo_df,


@app.cell
def _(mo):
    n_trials = mo.ui.number(
        start=10,
        stop=300,
        value=100,
        label="Trials per level (count)",
    )
    n_actors = mo.ui.number(
        start=1,
        stop=30,
        value=3,
        label="Participants (count)",
    )

    v_intercept = mo.ui.number(
        start=-3.0,
        stop=3.0,
        value=0.0,
        step=0.1,
        label="v — drift intercept",
    )
    v_scale = mo.ui.number(
        start=0.0,
        stop=6.0,
        value=2.5,
        step=0.1,
        label="v scale — drift per signed evidence",
    )
    boundary_a = mo.ui.number(
        start=0.3,
        stop=2.5,
        value=1.2,
        step=0.05,
        label="a — boundary separation",
    )
    starting_z = mo.ui.number(
        start=0.1,
        stop=0.9,
        value=0.5,
        step=0.05,
        label="z — starting point (fraction of a)",
    )
    lapse = mo.ui.number(
        start=0.0,
        stop=0.2,
        value=0.0,
        step=0.005,
        label="Lapse rate (proportion)",
    )
    ndt = mo.ui.number(
        start=0.0,
        stop=2.0,
        value=0.30,
        step=0.01,
        label="t — non-decision time (s)",
    )
    lapse_rt_extra = mo.ui.number(
        start=0.0,
        stop=2.0,
        value=0.35,
        step=0.05,
        label="Lapse RT extra (s, added to t)",
    )

    run_sim = mo.ui.run_button(label="Run simulation")

    return (
        boundary_a,
        lapse,
        lapse_rt_extra,
        n_actors,
        n_trials,
        ndt,
        run_sim,
        starting_z,
        v_intercept,
        v_scale,
    )


@app.cell
def _(
    boundary_a,
    lapse,
    lapse_rt_extra,
    mo,
    n_actors,
    n_trials,
    ndt,
    run_sim,
    starting_z,
    v_intercept,
    v_scale,
):
    simulator_info_row = mo.Html(
        """
<div class="coherence-demo-simulator-info">
  <details>
    <summary aria-label="How the simulated actor works">
      <span>Simulator Info</span>
      <span class="coherence-demo-simulator-info__badge">?</span>
    </summary>
    <div class="coherence-demo-simulator-info__panel">
      <p><strong>Experiment.</strong> For each participant, the app
      runs every combination of the three coherence levels (A/B/C) and your chosen trial count.
      Each trial is a binary left/right motion discrimination with random direction, using the
      same side-strength encoding as the browser demo.</p>
      <p><strong>Actor.</strong> Responses are generated by
      <code>DdmActor</code>: alternative strengths map to signed evidence and trial drift
      <em>v = v_intercept + v_scale × (strength<sub>right</sub> − strength<sub>left</sub>)</em>;
      one forward draw from ssm-simulators (<code>model=ddm</code>) with boundary <code>a</code>,
      starting point <code>z</code>, and non-decision time <code>t</code>. Lapse trials use a
      random choice and <code>t + lapse_rt_extra</code>.</p>
      <p><strong>Output.</strong> Run simulation builds a trial-level table
      (accuracy, RT in s) used by the summary table, plots, and (after fitting) HSSM.</p>
    </div>
  </details>
</div>
"""
    )

    mo.vstack(
        [
            mo.md("### Simulation"),
            mo.accordion(
                {
                    "DDM simulator settings": mo.vstack(
                        [
                            mo.hstack([n_trials, n_actors], gap=1),
                            mo.hstack([v_intercept, v_scale, boundary_a], gap=1),
                            mo.hstack([starting_z, ndt, lapse], gap=1),
                            mo.hstack([lapse_rt_extra], gap=1),
                        ],
                        gap=0.6,
                    ),
                }
            ),
            run_sim,
            simulator_info_row,
        ],
        gap=0.5,
    )
    return


@app.cell
def _(
    DdmActor,
    ExperimentGenerator,
    FIT_DF_COLUMNS,
    boundary_a,
    lapse,
    lapse_rt_extra,
    make_motion_coherence_trials,
    motion_stimulus_to_strengths,
    lvl1,
    lvl2,
    lvl3,
    mo,
    n_actors,
    n_trials,
    ndt,
    np,
    pd,
    run_sim,
    starting_z,
    v_intercept,
    v_scale,
):
    if not run_sim.value:
        df = pd.DataFrame(columns=FIT_DF_COLUMNS)
        _sim_out = mo.md("_Click **Run simulation** to generate synthetic data._")
    else:
        condition_levels = [max(0.0, float(s.value)) for s in (lvl1, lvl2, lvl3)]

        nT = max(10, min(300, int(n_trials.value or 100)))
        nS = max(1, min(30, int(n_actors.value or 3)))

        rng = np.random.default_rng(12345)

        experiment = ExperimentGenerator()
        for level in condition_levels:
            experiment.add_trial_generator(
                make_motion_coherence_trials(
                    n_trials=nT,
                    coherence=level,
                    rng=rng,
                )
            )

        def build_actor(_subj: int):
            obs_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            return DdmActor(
                v_intercept=float(v_intercept.value),
                v_scale=float(v_scale.value),
                a=float(boundary_a.value),
                z=float(starting_z.value),
                lapse_rate=float(lapse.value),
                lapse_rt_extra=float(lapse_rt_extra.value),
                stimulus_to_strengths=motion_stimulus_to_strengths,
                rng=obs_rng,
            )

        rows = experiment.simulate(
            actor_factory=build_actor,
            n_subjects=nS,
            ndt=float(ndt.value),
        )

        df = pd.DataFrame(rows)
        _sim_out = df
    _sim_out
    return df


@app.cell
def _(df, mo, summarize_behavior):
    by = summarize_behavior(df)
    if df.empty:
        _summary = mo.md("_No simulation data yet._")
    else:
        by_labeled = by.rename(
            columns={
                "stim_level": "coherence (proportion)",
                "acc": "accuracy (proportion)",
                "rt_mean": "mean RT (s)",
                "rt_med": "median RT (s)",
                "n": "trials (count)",
            }
        )
        _summary = mo.vstack(
            [
                mo.md("### Simulated behavior summary"),
                mo.Html(by_labeled.to_html(index=False, classes="dataframe")),
            ],
            gap=0.5,
        )
    _summary
    return by


@app.cell
def _(
    behavior_summary_charts,
    by,
    demo_df,
    df,
    mo,
    summarize_behavior,
):
    import altair as alt

    sim_has = not df.empty
    demo_has = not demo_df.empty

    if not sim_has and not demo_has:
        _plots = mo.md("_No plots yet. Run the simulation or complete the demonstration._")
    else:
        columns: list[object] = []
        if sim_has:
            sim_charts = mo.ui.altair_chart(behavior_summary_charts(by, alt=alt))
            columns.append(
                mo.vstack([mo.md("**Simulation**"), sim_charts], gap=0.35, align="center")
            )
        if demo_has:
            by_demo = summarize_behavior(demo_df)
            demo_charts = mo.ui.altair_chart(behavior_summary_charts(by_demo, alt=alt))
            columns.append(
                mo.vstack([mo.md("**User Demo Data**"), demo_charts], gap=0.35, align="center")
            )
        _plots = mo.vstack(
            [mo.md("### Plots"), mo.hstack(columns, gap=1, align="start")],
            gap=0.5,
        )
    _plots
    return


@app.cell
def _(mo):
    hssm_data_source = mo.ui.radio(
        options=["Simulation", "Demo"],
        value="Simulation",
        label="HSSM data source",
    )
    return hssm_data_source,


@app.cell
def _(mo):
    fit_draws = mo.ui.number(
        start=100,
        stop=2000,
        value=600,
        step=100,
        label="HSSM posterior draws (samples)",
    )
    fit_tune = mo.ui.number(
        start=100,
        stop=2000,
        value=600,
        step=100,
        label="HSSM warmup tune (samples)",
    )
    fit_chains = mo.ui.number(
        start=1,
        stop=4,
        value=2,
        step=1,
        label="HSSM MCMC chains (count)",
    )
    return fit_chains, fit_draws, fit_tune


@app.cell
def _(
    demo_df,
    df,
    fit_chains,
    fit_draws,
    fit_tune,
    hssm_data_source,
    mo,
):
    _use_demo = hssm_data_source.value == "Demo"
    _has_data = not demo_df.empty if _use_demo else not df.empty
    run_fit = mo.ui.run_button(label="Run HSSM fit", disabled=not _has_data)
    mo.vstack(
        [
            mo.md("### HSSM fitting"),
            hssm_data_source,
            mo.accordion(
                {
                    "Fitting settings": mo.vstack(
                        [mo.hstack([fit_draws, fit_tune, fit_chains], gap=1)],
                        gap=0.6,
                    ),
                }
            ),
            run_fit,
        ],
        gap=0.5,
    )
    return run_fit


@app.cell
def _(
    demo_df,
    df,
    fit_chains,
    fit_draws,
    fit_tune,
    hssm_data_source,
    mo,
    fit_hssm_model,
    run_fit,
    summarize_posterior,
):
    mo.stop(not run_fit.value)

    fit_df = demo_df if hssm_data_source.value == "Demo" else df
    _source = hssm_data_source.value.lower()
    header = mo.md(f"### HSSM fit ({_source} data, DDM drift depends on coherence)")
    model, idata = fit_hssm_model(
        fit_df,
        draws=int(fit_draws.value),
        tune=int(fit_tune.value),
        chains=int(fit_chains.value),
    )
    summ = summarize_posterior(idata)
    _blocks = [mo.md("#### Posterior summary"), mo.Html(summ.to_html(classes="dataframe"))]
    if int(fit_chains.value) < 2:
        _blocks.append(
            mo.Html(
                '<div class="coherence-demo-hssm-chain-warning">'
                "Note: convergence diagnostics like <code>r_hat</code> require at least 2 chains; "
                "current fit used 1 chain."
                "</div>"
            )
        )
    summary_block = mo.vstack(_blocks, gap=0.5)
    mo.vstack([header, summary_block], gap=0.75)
    return fit_df, idata, model


@app.cell
def _(fit_df, idata, mo, model):
    import base64 as _base64
    import io as _io

    import hssm.plotting as _hplot
    _idata_pp = model.sample_posterior_predictive(
        idata=idata,
        inplace=False,
        include_group_specific=False,
        kind="response",
    )
    _ax_or_grid = _hplot.plot_model_cartoon(
        model,
        idata=_idata_pp,
        data=fit_df,
        predictive_group="posterior_predictive",
        plot_data=True,
        n_samples=20,
        plot_predictive_samples=True,
        bins=100,
        title="HSSM Model Cartoon",
        xlabel="Response time (s)",
    )

    _obj = _ax_or_grid[0] if isinstance(_ax_or_grid, list) and _ax_or_grid else _ax_or_grid
    _fig = getattr(_obj, "figure", None) or getattr(_obj, "fig", None)
    if _fig is None and hasattr(_obj, "get_figure"):
        _fig = _obj.get_figure()
    if _fig is None:
        raise TypeError(f"Unexpected plot object type: {type(_obj)}")
    _w, _h = _fig.get_size_inches()
    _scale = 2.0 / 3.0
    _fig.set_size_inches(max(1.5, _w * _scale), max(1.0, _h * _scale))
    for _ax in _fig.axes:
        _ax.title.set_fontsize(max(6, _ax.title.get_fontsize() * _scale))
        _ax.xaxis.label.set_fontsize(max(6, _ax.xaxis.label.get_fontsize() * _scale))
        _ax.yaxis.label.set_fontsize(max(6, _ax.yaxis.label.get_fontsize() * _scale))
        _ax.tick_params(axis="both", labelsize=max(6, 10 * _scale))
        _legend = _ax.get_legend()
        if _legend is not None:
            _legend.set_title(
                _legend.get_title().get_text(),
                prop={"size": max(6, 10 * _scale)},
            )
            for _txt in _legend.get_texts():
                _txt.set_fontsize(max(6, _txt.get_fontsize() * _scale))
            _legend.borderpad *= _scale
            _legend.labelspacing *= _scale
            _legend.handlelength *= _scale
            _legend.handletextpad *= _scale
            _legend.borderaxespad *= _scale

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _buf.seek(0)
    _b64 = _base64.b64encode(_buf.read()).decode("ascii")
    _img = mo.Html(
        f'<img class="coherence-demo-cartoon-img" alt="HSSM model cartoon" '
        f'src="data:image/png;base64,{_b64}" />'
    )
    _out = mo.vstack([mo.md("### HSSM model cartoon"), _img], gap=0.5)
    _out
    return


if __name__ == "__main__":
    app.run()
