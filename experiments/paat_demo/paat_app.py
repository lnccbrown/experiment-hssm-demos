import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    app_title="Probabilistic Approach-Avoidance Task",
    css_file="paat_app.css",
)


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    _notebook_dir = mo.notebook_dir()
    project_root = _notebook_dir.resolve().parents[1] if _notebook_dir else Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return mo, project_root


@app.cell
def _():
    import numpy as np
    import pandas as pd

    from analysis import paat_analysis as pa
    from analysis import paat_plots as plots
    from analysis.paat_fit import (
        fit_diagnostics,
        fit_paat,
        interval_coverage_table,
        posterior_table,
        sample_posterior_predictive,
    )
    from experiments.paat_demo.paat_export import paat_trials_dataframe
    from experiments.paat_demo.paat_timeline import (
        PAAT_MESSAGE_TYPE,
        PAATTiming,
        build_paat_timeline,
        paat_runner_config,
    )
    from observers.paat_ssm import (
        PRESETS,
        AngleParameters,
        drift_coefficients_from_mapping,
        simulate_players,
    )
    from renderers.paat_wheels import paat_css
    from runtime.embed import render_srcdoc_iframe
    from runtime.jspsych_export import create_jspsych_marimo_bridge
    from runtime.jspsych_runner import build_jspsych_runner_html
    from schemas.tasks.paat import DESIGN_PROFILES, make_schedule

    return (
        AngleParameters,
        DESIGN_PROFILES,
        PAATTiming,
        PAAT_MESSAGE_TYPE,
        PRESETS,
        build_jspsych_runner_html,
        build_paat_timeline,
        create_jspsych_marimo_bridge,
        drift_coefficients_from_mapping,
        fit_diagnostics,
        fit_paat,
        interval_coverage_table,
        make_schedule,
        np,
        pa,
        paat_css,
        paat_runner_config,
        paat_trials_dataframe,
        pd,
        plots,
        posterior_table,
        render_srcdoc_iframe,
        sample_posterior_predictive,
        simulate_players,
    )


@app.cell
def _(mo):
    import math as _math

    def step_header(number, title, lead, anchor):
        return mo.Html(
            f'<div class="paat-step" id="{anchor}"><span class="paat-step__number">{number}</span>'
            f"<div><h2>{title}</h2><p>{lead}</p></div></div>"
        )

    def note(text, kind="hint", label=None):
        prefix = f'<span class="paat-note__label">{label}</span> ' if label else ""
        return mo.Html(f'<div class="paat-note paat-note--{kind}">{prefix}{text}</div>')

    def tiles(items):
        body = "".join(
            '<div class="paat-stat">'
            f'<div class="paat-stat__label">{label}</div>'
            f'<div class="paat-stat__value">{value}</div>'
            f'<div class="paat-stat__caption">{caption}</div></div>'
            for label, value, caption in items
        )
        return mo.Html(f'<div class="paat-stats">{body}</div>')

    def table_html(table):
        truth = "true" in table.columns
        head = (
            "<th>Parameter</th><th>Estimate</th><th>94% interval</th>"
            "<th>r̂</th><th>ESS bulk</th><th>ESS tail</th>"
        )
        if truth:
            head += "<th>True / interval</th>"

        def metric(value, digits=2):
            value = float(value)
            return f"{value:.{digits}f}" if _math.isfinite(value) else "—"

        rows = []
        for row in table.itertuples(index=False):
            recovery = ""
            if truth:
                if row.true == row.true:
                    if row.covered is None or row.covered != row.covered:
                        mark = '<span class="paat-muted">not assessed</span>'
                    else:
                        mark = '<span class="paat-ok">✓</span>' if bool(row.covered) else '<span class="paat-miss">✕</span>'
                    recovery = f'<td class="num">{row.true:.3f} {mark}</td>'
                else:
                    recovery = '<td class="num">—</td>'
            support_mark = "" if bool(row.interpretation_supported) else " · do not interpret"
            rows.append(
                f"<tr><td>{row.parameter}{support_mark}<br><code>{row.symbol}</code></td>"
                f'<td class="num">{row.estimate:.3f}</td>'
                f'<td class="num">{row.low:.3f} to {row.high:.3f}</td>'
                f'<td class="num">{metric(row.r_hat)}</td>'
                f'<td class="num">{metric(row.ess_bulk, 0)}</td>'
                f'<td class="num">{metric(row.ess_tail, 0)}</td>{recovery}</tr>'
            )
        return mo.Html(
            f'<div class="paat-table-wrap"><table class="paat-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
        )

    def matplotlib_image(plot_object):
        import base64
        import io

        import matplotlib.pyplot as plt

        figure = getattr(plot_object, "figure", None)
        if figure is None:
            figure = getattr(plot_object, "fig", None)
        if figure is None:
            figure = plot_object
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=145, bbox_inches="tight")
        plt.close(figure)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return mo.Html(f'<div class="paat-figure"><img src="data:image/png;base64,{encoded}" /></div>')

    return matplotlib_image, note, step_header, table_html, tiles


@app.cell
def _(mo):
    get_sim, set_sim = mo.state(None)
    get_fit, set_fit = mo.state(None)
    return get_fit, get_sim, set_fit, set_sim


@app.cell
def _(mo):
    mo.sidebar(
        [
            mo.md("### PAAT · HSSM ecosystem"),
            mo.nav_menu(
                {
                    "#play": "1 · Play",
                    "#explore": "2 · Explore",
                    "#simulate": "3 · Simulate",
                    "#fit": "4 · Fit",
                    "#check": "5 · Check",
                },
                orientation="vertical",
            ),
        ],
        footer=mo.md("<small>Task: Cheng et al. (2026)<br>Simulation: ssm-simulators<br>Inference: HSSM</small>"),
    )
    return


@app.cell
def _(mo, paat_css, project_root):
    import base64 as _base64

    _logo = _base64.b64encode((project_root / "assets" / "logo" / "hssm_white.png").read_bytes()).decode("ascii")
    _path = "".join(
        f'<a href="#{anchor}">{label}</a>'
        for anchor, label in (
            ("play", "1 · Play"),
            ("explore", "2 · Explore"),
            ("simulate", "3 · Simulate"),
            ("fit", "4 · Fit"),
            ("check", "5 · Check"),
        )
    )
    mo.vstack(
        [
            mo.Html(f"<style>{paat_css()}</style>"),
            mo.Html(
                '<div class="paat-hero">'
                f'<img class="paat-hero__logo" alt="HSSM" src="data:image/png;base64,{_logo}" />'
                "<h1>Probabilistic Approach-Avoidance Task</h1>"
                "<p>Choose between uncertain reward and aversive outcomes, then follow the same data through "
                "<strong>ssm-simulators</strong> and the built-in HSSM <code>angle</code> model.</p>"
                f'<div class="paat-hero__path">{_path}</div></div>'
            ),
            mo.Html(
                '<div class="paat-model-strip">'
                "<div>Probabilities<br><strong>PAAT trials</strong></div>"
                "<div>Trial-wise drift<br><strong>regression</strong></div>"
                "<div>Collapsing bounds<br><strong>ssm-simulators</strong></div>"
                "<div>Bayesian inference<br><strong>HSSM</strong></div>"
                "</div>"
            ),
            mo.accordion(
                {
                    "What is being modeled?": mo.md(
                        """
    The **risky** option is the one with the higher aversive-outcome probability. Positive drift points
    toward that option; negative drift points toward the safer option. Relative reward, relative aversive
    probability, and motivational congruency jointly determine drift. The decision threshold collapses
    linearly through time (`theta`), exactly the built-in HSSM/ssm-simulators `angle` model.

    The conference profile shortens only the animation and trial count. The two Study 2 profiles are
    count-matched synthetic designs: they preserve 96 choices and original timing, but use a balanced
    evidence grid rather than the released canonical trial list. No disturbing IAPS images are distributed.
    """
                    )
                }
            ),
        ],
        gap=0.65,
    )
    return


@app.cell
def _(DESIGN_PROFILES, mo):
    task_profile = mo.ui.dropdown(
        {profile.label: name for name, profile in DESIGN_PROFILES.items()},
        value=DESIGN_PROFILES["conference"].label,
        label="Task profile",
    )
    new_session = mo.ui.button(value=0, on_click=lambda value: value + 1, label="New session")
    return new_session, task_profile


@app.cell
def _(
    PAATTiming,
    build_jspsych_runner_html,
    build_paat_timeline,
    make_schedule,
    mo,
    new_session,
    np,
    paat_runner_config,
    render_srcdoc_iframe,
    step_header,
    task_profile,
):
    import secrets as _secrets

    # Do not depend on exported results: completing a game must not rebuild its iframe.
    _ = new_session.value
    _seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    game_session_id = _secrets.token_hex(16)
    game_iframe_id = f"paat-game-{game_session_id}"
    game_schedule = make_schedule(task_profile.value, seed=_seed)
    _timing = PAATTiming() if task_profile.value == "conference" else PAATTiming.research()
    game_response_window_ms = _timing.response_window_ms
    _timeline = build_paat_timeline(game_schedule, timing=_timing)
    _html = build_jspsych_runner_html(
        _timeline,
        config=paat_runner_config(session_id=game_session_id),
    )
    mo.vstack(
        [
            step_header(
                1,
                "Play",
                "Choose an option with the arrow keys or by clicking it. Outcomes are probabilistic, so there is no always-correct answer.",
                "play",
            ),
            mo.hstack([task_profile, new_session], justify="start", align="end", gap=1.5, wrap=True),
            mo.Html(
                render_srcdoc_iframe(
                    _html,
                    title="Probabilistic Approach-Avoidance Task",
                    height=520,
                    style="display:block;border:1px solid rgba(100,116,139,.3);border-radius:14px;",
                    iframe_id=game_iframe_id,
                )
            ),
        ],
        gap=0.8,
    )
    return game_iframe_id, game_response_window_ms, game_schedule, game_session_id


@app.cell
def _(PAAT_MESSAGE_TYPE, create_jspsych_marimo_bridge, game_iframe_id, game_session_id):
    game_results = create_jspsych_marimo_bridge(
        message_type=PAAT_MESSAGE_TYPE,
        session_id=game_session_id,
        iframe_id=game_iframe_id,
    )
    game_results
    return (game_results,)


@app.cell
def _(game_response_window_ms, game_results, game_schedule, paat_trials_dataframe):
    try:
        human_df = paat_trials_dataframe(
            game_results.value["rows_json"],
            schedule=game_schedule,
            response_window_ms=game_response_window_ms,
            result_received=game_results.value["result_received"],
        ).assign(source="You")
        human_export_error = None
    except (TypeError, ValueError) as _error:
        human_df = paat_trials_dataframe(
            "[]",
            schedule=game_schedule,
            result_received=False,
        ).assign(source="You")
        human_export_error = str(_error)
    return human_df, human_export_error


@app.cell
def _(human_df, human_export_error, note, pa, tiles):
    import html as _html

    if human_export_error is not None:
        _out = note(
            _html.escape(human_export_error),
            kind="danger",
            label="Session export rejected:",
        )
    elif human_df.empty:
        _out = note("Your choices will appear here when the session finishes.", kind="empty")
    else:
        _stats = pa.session_stats(human_df)
        _unit = "points" if human_df["profile"].iloc[0] == "conference" else "dollars"
        _reward = f"{_stats['reward']:.0f}" if _unit == "points" else f"${_stats['reward']:.2f}"
        _out = tiles(
            [
                ("Risky choices", f"{_stats['p_risky']:.0%}", f"{_stats['answered']} answered trials"),
                ("Median response", f"{_stats['median_rt']:.2f} s", "choice phase"),
                ("Reward", _reward, _unit),
                ("Missed deadlines", str(_stats["timeouts"]), "six-second limit"),
            ]
        )
    _out
    return


@app.cell
def _(get_sim, human_df, mo, note, pa, pd, plots, step_header):
    _sim = get_sim()
    _frames = [frame for frame in (human_df, None if _sim is None else _sim["df"]) if frame is not None and not frame.empty]
    _header = step_header(
        2,
        "Explore",
        "See how relative reward, relative aversive probability, and motivational congruency shaped choices and response times.",
        "explore",
    )
    if not _frames:
        _body = note("Play the task above or simulate participants in step 3.", kind="empty")
    else:
        _data = pd.concat(_frames, ignore_index=True)
        _body = mo.ui.tabs(
            {
                "Trial type": mo.vstack(
                    [plots.condition_chart(pa.condition_summary(_data)), plots.rt_chart(pa.condition_summary(_data))],
                    gap=0.8,
                ),
                "Reward evidence": plots.evidence_chart(pa.evidence_summary(_data, "rel_reward"), "rel_reward"),
                "Aversive evidence": plots.evidence_chart(pa.evidence_summary(_data, "rel_aversive"), "rel_aversive"),
                "Choice + RT": plots.signed_rt_chart(pa.signed_rts(_data)),
            }
        )
    mo.vstack([_header, _body], gap=0.8)
    return


@app.cell
def _(PRESETS, mo):
    preset = mo.ui.dropdown(list(PRESETS), value="Study 1 pattern", label="Starting pattern")
    return (preset,)


@app.cell
def _(DESIGN_PROFILES, PRESETS, mo, preset):
    _preset = PRESETS[preset.value]
    _coefficients = _preset.coefficients
    _angle = _preset.angle
    sim_profile = mo.ui.dropdown(
        {DESIGN_PROFILES[name].label: name for name in ("published_study2", "osf_study2", "conference")},
        value=DESIGN_PROFILES["published_study2"].label,
        label="Simulation profile",
    )
    n_players = mo.ui.number(1, 20, 1, 5, label="Participants")
    spread = mo.ui.slider(0.0, 0.5, 0.05, 0.0, show_value=True, label="Drift heterogeneity")
    sim_params = mo.ui.dictionary(
        {
            "intercept": mo.ui.number(-1.5, 1.5, 0.01, _coefficients.intercept, label="Drift intercept"),
            "reward": mo.ui.number(-1.5, 1.5, 0.01, _coefficients.reward, label="Reward effect"),
            "aversive": mo.ui.number(-1.5, 1.5, 0.01, _coefficients.aversive, label="Aversive effect"),
            "congruency": mo.ui.number(-1.5, 1.5, 0.01, _coefficients.congruency, label="Congruency shift"),
            "reward_x_congruency": mo.ui.number(-1.0, 1.0, 0.01, _coefficients.reward_x_congruency, label="Reward × congruency"),
            "aversive_x_congruency": mo.ui.number(-1.0, 1.0, 0.01, _coefficients.aversive_x_congruency, label="Aversive × congruency"),
            "a": mo.ui.number(0.3, 2.5, 0.01, _angle.a, label="a · threshold"),
            "z": mo.ui.number(0.1, 0.9, 0.01, _angle.z, label="z · risky bias"),
            "t": mo.ui.number(0.05, 1.0, 0.01, _angle.t, label="t · non-decision time"),
            "theta": mo.ui.number(0.0, 0.7, 0.01, _angle.theta, label="θ · boundary collapse"),
        }
    )
    run_sim = mo.ui.run_button(label="Simulate with ssm-simulators", kind="success")
    return n_players, run_sim, sim_params, sim_profile, spread


@app.cell
def _(
    PRESETS,
    mo,
    n_players,
    preset,
    run_sim,
    sim_params,
    sim_profile,
    spread,
    step_header,
):
    mo.vstack(
        [
            step_header(
                3,
                "Simulate",
                "A trial-wise drift regression feeds the official ssm-simulators angle model; no custom decision process is used.",
                "simulate",
            ),
            mo.hstack([preset, sim_profile, n_players, spread], justify="start", align="end", gap=1.2, wrap=True),
            mo.md(f"_{PRESETS[preset.value].description}_"),
            mo.accordion({"Model parameters": sim_params.hstack(justify="start", gap=1, wrap=True)}),
            run_sim,
        ],
        gap=0.75,
    )
    return


@app.cell
def _(
    AngleParameters,
    drift_coefficients_from_mapping,
    mo,
    n_players,
    note,
    np,
    run_sim,
    set_sim,
    sim_params,
    sim_profile,
    simulate_players,
    spread,
):
    import html as _html

    mo.stop(not run_sim.value)
    set_sim(None)
    _values = {name: float(value) for name, value in sim_params.value.items()}
    _coefficients = drift_coefficients_from_mapping(_values)
    _angle = AngleParameters(a=_values["a"], z=_values["z"], t=_values["t"], theta=_values["theta"])
    _seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    try:
        with mo.status.spinner(title="Simulating PAAT participants with the angle model…"):
            _df = simulate_players(
                profile=sim_profile.value,
                coefficients=_coefficients,
                angle=_angle,
                n_players=int(n_players.value),
                spread=float(spread.value),
                seed=_seed,
            ).assign(source="Simulated")
    except ValueError as _error:
        mo.stop(True, note(_html.escape(str(_error)), kind="danger", label="Simulation stopped:"))
    set_sim(
        {
            "df": _df,
            "coefficients": _coefficients,
            "angle": _angle,
            "spread": float(spread.value),
            "profile": sim_profile.value,
            "n_players": int(n_players.value),
            "seed": _seed,
        }
    )
    return


@app.cell
def _(get_sim, mo, note, pa, plots):
    _sim = get_sim()
    if _sim is None:
        _out = note("No simulated cohort yet.", kind="empty")
    else:
        _stats = pa.session_stats(_sim["df"])
        _timeout_rate = _stats["timeouts"] / _stats["trials"] if _stats["trials"] else 0.0
        _timeout_note = (
            note(
                f"<strong>{_stats['timeouts']} / {_stats['trials']} ({_timeout_rate:.1%})</strong> trials missed the deadline. "
                "The stock angle LAN used below does not model these right-censored no-response trials; "
                "fits are blocked when omissions exceed 5% overall or for any participant.",
                kind="warn" if _stats["timeouts"] else "hint",
                label="Deadline audit:",
            )
        )
        _out = mo.vstack(
            [
                note(
                    f"Simulated <strong>{_sim['n_players']} participant(s)</strong> and {len(_sim['df'])} choices. "
                    f"Risky choice rate: <strong>{_stats['p_risky']:.0%}</strong>; median RT: "
                    f"<strong>{_stats['median_rt']:.2f} s</strong>. These rows also appear in step 2.",
                    kind="success",
                ),
                _timeout_note,
                plots.drift_surface_chart(pa.drift_grid(_sim["coefficients"], _sim["df"])),
            ],
            gap=0.7,
        )
    _out
    return


@app.cell
def _(mo):
    fit_source = mo.ui.dropdown(["Simulated participants", "Your session"], value="Simulated participants", label="Data")
    hierarchical = mo.ui.switch(value=False, label="Participant effects on drift")
    fit_settings = mo.ui.dictionary(
        {
            "draws": mo.ui.number(300, 1500, 100, 300, label="Posterior draws"),
            "tune": mo.ui.number(300, 1500, 100, 300, label="Warm-up draws"),
            "chains": mo.ui.number(2, 4, 1, 2, label="Chains"),
        }
    )
    run_fit = mo.ui.run_button(label="Fit HSSM angle model", kind="success")
    return fit_settings, fit_source, hierarchical, run_fit


@app.cell
def _(fit_settings, fit_source, hierarchical, mo, run_fit, step_header):
    mo.vstack(
        [
            step_header(
                4,
                "Fit",
                "Run an exploratory HSSM fit of the PAAT drift regression and angle-model parameters.",
                "fit",
            ),
            mo.hstack([fit_source, hierarchical], justify="start", align="end", gap=1.5, wrap=True),
            mo.md(
                "_Conference-session fits are pedagogical, not research inference. Leave hierarchy off for speed; "
                "multi-participant research fits need more draws, more chains, and full diagnostic review. "
                "The teaching hierarchy varies drift intercepts/slopes by participant while pooling a, z, t, and θ. "
                "Congruency is defined by the sign of relative reward, so their supports do not overlap: the "
                "congruency intercept is an off-support extrapolation and is not interpreted. Fits use observed "
                "responses only, add no lapse mixture, and are blocked above 5% deadline omissions overall "
                "or for any participant._"
            ),
            mo.accordion({"Sampler settings": fit_settings.hstack(justify="start", gap=1, wrap=True)}),
            run_fit,
        ],
        gap=0.75,
    )
    return


@app.cell
def _(
    fit_paat,
    fit_settings,
    fit_source,
    get_sim,
    hierarchical,
    human_df,
    mo,
    note,
    pa,
    run_fit,
    sample_posterior_predictive,
    set_fit,
):
    import html as _html

    mo.stop(not run_fit.value)
    set_fit(None)
    _sim = get_sim()
    _data = human_df if fit_source.value == "Your session" else (None if _sim is None else _sim["df"])
    mo.stop(_data is None or _data.empty, note("Play a session or simulate participants before fitting.", kind="warn"))
    _table, _excluded = pa.fit_table(_data)
    try:
        pa.require_defensible_uncensored_fit(_excluded)
    except ValueError as _error:
        mo.stop(True, note(_html.escape(str(_error)), kind="danger", label="Fit blocked:"))
    mo.stop(len(_table) < 20, note("Fewer than 20 usable choices remain; there is too little information to fit.", kind="warn"))
    _fingerprint = pa.data_fingerprint(_table)
    _hierarchical = bool(hierarchical.value and _table["participant_id"].nunique() > 1)
    _settings = {name: int(value) for name, value in fit_settings.value.items()}
    try:
        with mo.status.spinner(
            title="Fitting the built-in HSSM angle model…",
            subtitle=f"{len(_table)} choices; the first LAN/JAX compilation is the slowest",
        ):
            _fit = fit_paat(_table, hierarchical=_hierarchical, seed=1, **_settings)
            _ppc = sample_posterior_predictive(_fit, draws=min(100, _settings["draws"]))
    except Exception as _error:
        mo.stop(True, note(_html.escape(str(_error)), kind="danger", label="Fit failed:"))
    set_fit(
        {
            "fit": _fit,
            "ppc": _ppc,
            "excluded": _excluded,
            "source": fit_source.value,
            "truth": _sim if fit_source.value == "Simulated participants" else None,
            "fingerprint": _fingerprint,
            "settings": _settings,
            "hierarchical": _hierarchical,
            "profile": str(_data["profile"].iloc[0]),
        }
    )
    return


@app.cell
def _(
    fit_diagnostics,
    fit_source,
    get_fit,
    get_sim,
    hierarchical,
    human_df,
    matplotlib_image,
    mo,
    note,
    pa,
    posterior_table,
    interval_coverage_table,
    step_header,
    table_html,
):
    _state = get_fit()
    _header = step_header(
        5,
        "Check",
        "Inspect sampler diagnostics, a one-simulation truth overlay, posterior predictions, and the collapsing boundary.",
        "check",
    )
    if _state is None:
        _body = note("No fit yet. Choose data in step 4 and run HSSM.", kind="empty")
    else:
        _current_sim = get_sim()
        _current_data = (
            human_df
            if _state["source"] == "Your session"
            else (None if _current_sim is None else _current_sim["df"])
        )
        if _current_data is None or _current_data.empty:
            _current_fingerprint = None
            _current_hierarchical = None
        else:
            _current_table, _ = pa.fit_table(_current_data)
            _current_fingerprint = pa.data_fingerprint(_current_table)
            _current_hierarchical = bool(
                hierarchical.value and _current_table["participant_id"].nunique() > 1
            )
        _stale = (
            fit_source.value != _state["source"]
            or _current_fingerprint != _state["fingerprint"]
            or _current_hierarchical != _state["hierarchical"]
        )
        if _stale:
            _body = note(
                "The data or selected source changed after this fit. Run HSSM again before interpreting checks.",
                kind="warn",
                label="Fit is stale:",
            )
        else:
            _fit = _state["fit"]
            _truth = _state["truth"]
            _diagnostics = fit_diagnostics(_fit)
            if _truth is not None and _truth["spread"] == 0:
                _summary = interval_coverage_table(
                    _fit,
                    _truth["coefficients"],
                    _truth["angle"],
                )
            else:
                _summary = posterior_table(_fit)
            _info = _state["excluded"]
            _settings = _state["settings"]
            _model_shape = "participant-varying drift" if _state["hierarchical"] else "pooled parameters"
            _intro = mo.md(
                f"Fit snapshot: **{_state['source'].lower()}**, profile `{_state['profile']}`, "
                f"{_model_shape}, {_info['kept']} choices, {_settings['chains']} chain(s) × "
                f"{_settings['draws']} draws "
                f"in {_fit.elapsed_s:.1f} s. Excluded {_info['timeouts']} timeout(s) and "
                f"{_info['anticipations']} response(s) at or below 250 ms. Timeout rate: "
                f"{_info['timeout_rate']:.1%} overall, {_info['max_participant_timeout_rate']:.1%} for the "
                "highest-omission participant. The off-support congruency intercept is displayed for model "
                "completeness but is not interpreted or scored. A truth check only describes this simulated "
                "dataset; it is not evidence of frequentist coverage or general recoverability."
            )
            if _diagnostics["ok"]:
                _diagnostic_note = note(
                    f"Maximum r̂ <strong>{_diagnostics['max_r_hat']:.3f}</strong>; minimum bulk/tail ESS "
                    f"<strong>{min(_diagnostics['min_ess_bulk'], _diagnostics['min_ess_tail']):.0f}</strong>; "
                    "no divergences.",
                    kind="success",
                    label="Diagnostics passed:",
                )
            else:
                _diagnostic_note = note(
                    "; ".join(_diagnostics["issues"])
                    + ". Truth-in-interval marks are withheld; treat this fit as exploratory.",
                    kind="warn",
                    label="Diagnostics need attention:",
                )
            _views = {"Parameters": table_html(_summary)}
            try:
                _predictive = _fit.model.plot_predictive(
                    dt=_state["ppc"],
                    col="congruency_code",
                    uncertainty="band",
                )
                _views["Posterior predictive"] = matplotlib_image(_predictive)
            except Exception as _error:
                _views["Posterior predictive"] = note(str(_error), kind="warn", label="Plot unavailable:")
            try:
                import hssm as _hssm

                _cartoon = _hssm.plotting.plot_model_cartoon(
                    _fit.model,
                    dt=_state["ppc"],
                    obs=0,
                    n_samples=20,
                    uncertainty="band",
                    random_state=2,
                    title="HSSM angle model",
                )
                _views["Model cartoon"] = matplotlib_image(_cartoon)
            except Exception as _error:
                _views["Model cartoon"] = note(str(_error), kind="warn", label="Plot unavailable:")
            try:
                import hssm as _hssm

                _quantiles = _hssm.plotting.plot_quantile_probability(
                    _fit.model,
                    cond="congruency_code",
                    dt=_state["ppc"],
                    n_samples=20,
                )
                _views["Quantile probability"] = matplotlib_image(_quantiles)
            except Exception as _error:
                _views["Quantile probability"] = note(str(_error), kind="warn", label="Plot unavailable:")
            _body = mo.vstack([_intro, _diagnostic_note, mo.ui.tabs(_views)], gap=0.7)
    mo.vstack([_header, _body], gap=0.8)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    **Reproduce it.** The experiment code, seeded designs, simulator adapter, and HSSM regression are all ordinary
    Python modules beside this notebook. The paper's winning linear-collapsing-boundary family is mapped here to
    HSSM's built-in `angle` model and packaged approximate differentiable likelihood; simulations use the
    corresponding stock `ssm-simulators` implementation. The app uses HSSM's safe priors, not the paper's full
    hierarchical prior specification.

    [HSSM documentation](https://lnccbrown.github.io/HSSM/) ·
    [ssm-simulators documentation](https://lnccbrown.github.io/ssm-simulators/) ·
    [PAAT paper](https://doi.org/10.3758/s13423-026-02885-9) ·
    [authors' OSF materials](https://osf.io/zx3cy/)
    """)
    return


if __name__ == "__main__":
    app.run()
