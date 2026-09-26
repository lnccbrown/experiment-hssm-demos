import marimo

__generated_with = "0.21.1"

# Marimo app: Probabilistic Selection Task with an RL-DDM (Pedersen, Frank & Biele, 2017).
# Play in the browser, explore learning curves, simulate players, fit with HSSM, illustrate recovery once.
# Entry point: ``marimo run experiments/pst_demo/pst_app.py``.

app = marimo.App(width="medium", css_file="pst_app.css", app_title="Probabilistic Selection Task")


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
def _(project_root):
    import numpy as np
    import pandas as pd

    from analysis import pst_analysis as pa
    from analysis import pst_plots as plots
    from analysis.pst_fit import (
        fit_diagnostics,
        fit_pst,
        personal_readout,
        posterior_predictive_curves,
        posterior_table,
    )
    from experiments.pst_demo.pst_export import pst_trials_dataframe
    from experiments.pst_demo.pst_timeline import PST_MESSAGE_TYPE, build_pst_timeline, pst_runner_config
    from actors.pst_rl import (
        PRESETS,
        PSTLearner,
        final_values,
        replay_latents,
        simulate_players,
        simulate_test_phase,
        vary_players,
    )
    from renderers.pst_symbols import pst_css, symbol_gallery_html
    from runtime.embed import render_srcdoc_iframe
    from runtime.jspsych_export import create_jspsych_marimo_bridge
    from runtime.jspsych_runner import build_jspsych_runner_html
    from schemas.tasks.pst import PSTConfig, make_schedule, symbol_label

    return (
        PRESETS,
        PSTConfig,
        PSTLearner,
        PST_MESSAGE_TYPE,
        build_jspsych_runner_html,
        build_pst_timeline,
        create_jspsych_marimo_bridge,
        final_values,
        fit_diagnostics,
        fit_pst,
        make_schedule,
        np,
        pa,
        pd,
        personal_readout,
        plots,
        posterior_table,
        posterior_predictive_curves,
        pst_css,
        pst_runner_config,
        pst_trials_dataframe,
        render_srcdoc_iframe,
        replay_latents,
        simulate_players,
        simulate_test_phase,
        symbol_gallery_html,
        symbol_label,
        vary_players,
    )


@app.cell
def _(mo, np):
    def step_header(number, title, lead, anchor):
        return mo.Html(
            f'<div class="pst-step" id="{anchor}"><span class="pst-step__num">{number}</span>'
            f'<div><h2 class="pst-step__title">{title}</h2><p class="pst-step__lead">{lead}</p></div></div>'
        )

    def note(html, kind="hint", label=None):
        """Compact message; kind is hint, empty, warn, danger or success (styles in pst_app.css)."""
        label_html = f'<span class="pst-note__label">{label}</span> ' if label else ""
        return mo.Html(f'<div class="pst-note pst-note--{kind}">{label_html}{html}</div>')

    def tiles(items, *, sentences=False):
        """Number tiles from (label, value, caption); sentences=True gives wider tiles for a full sentence."""
        cells = "".join(
            f'<div class="pst-stat"><div class="pst-stat__label">{label}</div>'
            f'<div class="pst-stat__value">{value}</div><div class="pst-stat__caption">{caption}</div></div>'
            for label, value, caption in items
        )
        return mo.Html(f'<div class="pst-stats{" pst-stats--sentences" if sentences else ""}">{cells}</div>')

    def posterior_html(table):
        has_truth = "true" in table.columns
        head = "<th>Parameter</th><th>Estimate</th><th>94% interval</th>"
        head += "<th>True value</th>" if has_truth else ""
        head += "<th>r&#770;</th><th>Bulk ESS</th><th>Tail ESS</th><th>MCSE</th>"
        body = []
        for row in table.itertuples(index=False):
            truth = ""
            if has_truth:
                mark = '<span class="pst-ok">&#10003;</span>' if row.recovered else '<span class="pst-miss">&#10007;</span>'
                truth = f'<td class="num">{row.true:.3f} {mark}</td>'
            r_hat = "—" if not np.isfinite(row.r_hat) else f"{row.r_hat:.2f}"
            body.append(
                f"<tr><td>{row.parameter} <code>{row.symbol}</code>"
                f'<span class="meaning">{row.meaning}</span></td>'
                f'<td class="num">{row.estimate:.3f}</td>'
                f'<td class="num">{row.low:.3f} to {row.high:.3f}</td>{truth}'
                f'<td class="num">{r_hat}</td><td class="num">{row.ess_bulk:.0f}</td>'
                f'<td class="num">{row.ess_tail:.0f}</td>'
                f'<td class="num">{row.mcse_mean:.3f}</td></tr>'
            )
        return mo.vstack(
            [
                mo.Html(f'<table class="pst-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'),
                mo.md(
                    "<small>Estimate = posterior mean. With four chains, r&#770; should be ≤ 1.01; effective sample "
                    "size (ESS) and Monte Carlo standard error (MCSE) quantify numerical precision.</small>"
                ),
            ],
            gap=0.3,
        )

    return note, posterior_html, step_header, tiles


@app.cell
def _(mo):
    get_sim, set_sim = mo.state(None)
    get_fit, set_fit = mo.state(None)
    return get_fit, get_sim, set_fit, set_sim


@app.cell
def _(mo):
    mo.sidebar(
        [
            mo.md("### Probabilistic Selection Task"),
            mo.nav_menu(
                {
                    "#play": "1 · Play",
                    "#explore": "2 · Explore",
                    "#simulate": "3 · Simulate",
                    "#fit": "4 · Fit",
                    "#recovery": "5 · Recovery illustration",
                },
                orientation="vertical",
            ),
        ],
        footer=mo.md("<small>Model: Pedersen, Frank & Biele (2017). Simulation: ssm-simulators. Fitting: HSSM.</small>"),
    )
    return


@app.cell
def _(mo, project_root, pst_css):
    import base64 as _base64

    _logo = _base64.b64encode((project_root / "assets" / "logo" / "hssm_white.png").read_bytes()).decode("ascii")
    _steps = "".join(
        f'<a href="#{anchor}">{label}</a>'
        for anchor, label in [
            ("play", "1 · Play"),
            ("explore", "2 · Explore"),
            ("simulate", "3 · Simulate"),
            ("fit", "4 · Fit"),
            ("recovery", "5 · Recovery illustration"),
        ]
    )
    mo.vstack(
        [
            mo.Html(f"<style>{pst_css()}</style>"),
            mo.Html(
                '<div class="pst-hero">'
                f'<img class="pst-hero__logo" alt="HSSM" src="data:image/png;base64,{_logo}" />'
                "<h1>Probabilistic Selection Task · shortened variant</h1>"
                "<p>Two symbols appear; one of them is correct more often, but never always. Play for a few "
                "minutes, then see how a reinforcement-learning model explains both <strong>what</strong> you "
                "chose and <strong>how fast</strong> you chose it.</p>"
                f'<div class="pst-hero__steps">{_steps}</div>'
                "</div>"
            ),
            mo.accordion(
                {
                    "How the model works": mo.md(
                        """
**Values.** Every symbol has a value that starts at 0. After each choice, the chosen symbol's value
moves toward the outcome (1 for *Correct!*, 0 for *Incorrect*): value ← value + η × (outcome − value).
The step is η⁺ after a pleasant surprise and η⁻ after an unpleasant one.

**Decisions.** Evidence for the better symbol builds up at speed v = m × (its value − the other value),
with noise, until it reaches a boundary. More caution (a higher boundary) means slower but more accurate
answers, and caution can change during the game: a(t) = bb × (t / 10)^bp. A fixed t seconds are added
for processes outside evidence accumulation. Computed v and a are limited to the DDM likelihood network's
validated support (v: −3 to 3; a: 0.3 to 2.5).

**Why it helps.** One set of numbers explains the learning curve *and* the response times: as the values
separate, the drift grows, so answers become both more accurate and faster.

This is "Model 6" of Pedersen, Frank & Biele (2017), *The drift diffusion model as the choice rule in
reinforcement learning*, the best of eight variants for adults with ADHD. Players are simulated with
ssm-simulators; fits use HSSM's built-in differentiable DDM likelihood network. This interactive task uses
shorter default training than the cited protocol, so it is a teaching variant rather than an exact replication.
"""
                    )
                }
            ),
        ],
        gap=0.6,
    )
    return


@app.cell
def _(mo):
    symbol_set = mo.ui.dropdown(
        {"Hiragana (original task)": "hiragana", "Shapes": "shapes"},
        value="Hiragana (original task)",
        label="Symbols",
    )
    blocks = mo.ui.range_slider(1, 6, 1, [2, 4], show_value=True, label="Blocks of 60 choices (min to max)")
    with_practice = mo.ui.switch(value=True, label="Practice round")
    with_test = mo.ui.switch(value=True, label="Final round")
    new_game = mo.ui.button(value=0, on_click=lambda count: count + 1, label="New game")
    return blocks, new_game, symbol_set, with_practice, with_test


@app.cell
def _(
    PSTConfig,
    blocks,
    build_jspsych_runner_html,
    build_pst_timeline,
    make_schedule,
    mo,
    new_game,
    np,
    pst_runner_config,
    render_srcdoc_iframe,
    step_header,
    symbol_set,
    with_practice,
    with_test,
):
    import secrets as _secrets

    # This cell must not depend on the game's results: re-running it restarts the game.
    _ = new_game.value
    game_session_id = _secrets.token_urlsafe(18)
    game_iframe_id = f"pst-game-{game_session_id}"
    _low, _high = (int(b) for b in blocks.value)
    _config = PSTConfig(min_blocks=_low, max_blocks=_high, symbol_set=symbol_set.value)
    game_schedule = make_schedule(_config, seed=int(np.random.default_rng().integers(2**31 - 1)))
    _timeline = build_pst_timeline(game_schedule, include_practice=with_practice.value, include_test=with_test.value)
    _html = build_jspsych_runner_html(_timeline, config=pst_runner_config(session_id=game_session_id))
    _minutes = f"{round(_low * 2.4)}" if _low == _high else f"{round(_low * 2.4)} to {round(_high * 2.4)}"
    mo.vstack(
        [
            step_header(
                1,
                "Play",
                f"Shortened learning phase: about {_minutes} minutes, plus the final round if enabled. "
                "Choose with the &larr; and &rarr; keys (fastest) or by clicking a symbol.",
                "play",
            ),
            mo.hstack([symbol_set, blocks, with_practice, with_test, new_game], justify="start", align="center", gap=1.5, wrap=True),
            mo.Html(
                render_srcdoc_iframe(
                    _html,
                    title="Probabilistic Selection Task",
                    height=470,
                    style="display:block;border:1px solid rgba(100,116,139,0.3);border-radius:14px;",
                    iframe_id=game_iframe_id,
                )
            ),
        ],
        gap=0.8,
    )
    return game_iframe_id, game_schedule, game_session_id


@app.cell
def _(PST_MESSAGE_TYPE, create_jspsych_marimo_bridge, game_iframe_id, game_session_id):
    game_results = create_jspsych_marimo_bridge(
        message_type=PST_MESSAGE_TYPE,
        session_id=game_session_id,
        iframe_id=game_iframe_id,
    )
    game_results
    return (game_results,)


@app.cell
def _(game_results, game_schedule, pst_trials_dataframe):
    human_df = pst_trials_dataframe(game_results.value["rows_json"], schedule=game_schedule).assign(source="You")
    return (human_df,)


@app.cell
def _(human_df, mo, note, np, pa, tiles):
    if human_df.empty:
        _out = note("When you finish the game, your results appear here and in the steps below.", kind="empty")
    else:
        _s = pa.session_stats(human_df)
        _blocks = f"{_s['blocks']} block{'s' if _s['blocks'] != 1 else ''}"
        _items = [
            ("Points", _s["points"], f"{_blocks}, {_s['trials']} choices"),
            ("Better symbol chosen", f"{_s['accuracy']:.0%}", "learning phase"),
            ("Average response time", f"{_s['mean_rt']:.2f} s", "learning phase"),
        ]
        if not np.isnan(_s["choose_A"]):
            _items += [
                ("Best symbol chosen", f"{_s['choose_A']:.0%}", "final round"),
                ("Worst symbol avoided", f"{_s['avoid_B']:.0%}", "final round"),
            ]
        _out = mo.vstack([mo.Html('<h4 class="pst-subhead">Your game</h4>'), tiles(_items)], gap=0.5)
    _out
    return


@app.cell
def _(game_schedule, get_sim, human_df, mo, note, pa, pd, plots, step_header, symbol_gallery_html, symbol_label):
    _sim = get_sim()
    _frames = [f for f in (human_df, None if _sim is None else _sim["df"]) if f is not None and not f.empty]
    _header = step_header(
        2,
        "Explore",
        "How choices and response times changed while learning: your game and simulated players side by side.",
        "explore",
    )
    _config = game_schedule.config
    _probs = _config.reward_probabilities()
    _gallery = symbol_gallery_html(
        [(game_schedule.glyphs[i], symbol_label(i), f"correct {round(100 * _probs[i])}%") for i in range(_config.n_symbols)],
        _config.symbol_set,
    )
    if not _frames:
        _body = note(
            "Nothing to show yet: <strong>play the game</strong> in step 1, or <strong>simulate players</strong> in step 3.",
            kind="empty",
        )
    else:
        _data = pd.concat(_frames, ignore_index=True)
        _own = "You" if not human_df.empty else "Simulated"
        _tabs = {
            "Learning": mo.vstack(
                [
                    plots.learning_curve_chart(pa.learning_curves(_data), title="Better-symbol choices over the game"),
                    note(
                        "Better choices climb fastest for the easy 80/20 pair (AB) and slowest for the hard 60/40 pair "
                        "(EF). Dashed lines are simulated players: when they track the solid line, the model's settings "
                        "behave like you.",
                        label="What to look for:",
                    ),
                ]
            ),
            "Response times": mo.vstack(
                [
                    plots.rt_chart(pa.rt_curves(_data)),
                    plots.signed_rt_chart(
                        pa.signed_rts(_data[_data["source"] == _own]),
                        title=f"Response times by choice ({'your game' if _own == 'You' else 'simulated players'})",
                    ),
                    note(
                        "Answers usually speed up as the two symbols' values separate, and choices of the worse symbol "
                        "(negative side) become rarer.",
                        label="What to look for:",
                    ),
                ]
            ),
        }
        _final = pa.final_round_summary(_data)
        if not _final.empty:
            _tabs["Final round"] = mo.vstack(
                [
                    plots.test_phase_chart(_final),
                    note(
                        "The final round has no feedback. Choose-A and avoid-B are conventionally associated with "
                        "learning from positive and negative outcomes, respectively, but neither is a pure measure "
                        "of one learning process. Chance is 50%.",
                        label="What to look for:",
                    ),
                ]
            )
        _tabs["Symbols"] = mo.vstack([mo.md("Symbols in the current game, and how often each is correct:"), mo.Html(_gallery)])
        _body = mo.ui.tabs(_tabs)
    mo.vstack([_header, _body], gap=0.8)
    return


@app.cell
def _(PRESETS, mo):
    preset = mo.ui.dropdown(list(PRESETS), value="Recovery study, middle values", label="Start from")
    return (preset,)


@app.cell
def _(PRESETS, mo, preset):
    _p = PRESETS[preset.value]
    sim_params = mo.ui.dictionary(
        {
            "eta_pos": mo.ui.number(0.001, 0.999, 0.001, round(_p["eta_pos"], 3), label="η⁺ learning from wins"),
            "eta_neg": mo.ui.number(0.001, 0.999, 0.001, round(_p["eta_neg"], 3), label="η⁻ learning from losses"),
            "m": mo.ui.number(0.1, 10.0, 0.001, round(_p["m"], 3), label="m value sensitivity"),
            "bb": mo.ui.number(0.3, 2.5, 0.001, round(_p["bb"], 3), label="bb caution"),
            "bp": mo.ui.number(-0.5, 0.5, 0.001, round(_p["bp"], 3), label="bp change in caution"),
            "t": mo.ui.number(0.2, 1.0, 0.001, round(_p["t"], 3), label="t non-decision time (s)"),
        }
    )
    n_players = mo.ui.number(1, 20, 1, 5, label="Players")
    sim_blocks = mo.ui.number(1, 6, 1, 4, label="Blocks of 60 choices")
    spread = mo.ui.slider(0.0, 0.5, 0.05, 0.0, show_value=True, label="Individual differences")
    run_sim = mo.ui.run_button(label="Simulate players", kind="success")
    return n_players, run_sim, sim_blocks, sim_params, spread


@app.cell
def _(mo, n_players, preset, run_sim, sim_blocks, sim_params, spread, step_header):
    mo.vstack(
        [
            step_header(
                3,
                "Simulate",
                "Virtual players run the same task with the model. Start from the paper's values or set your own, "
                "then compare with your game in step 2.",
                "simulate",
            ),
            mo.hstack([preset, n_players, sim_blocks, spread], justify="start", align="end", gap=1.5, wrap=True),
            sim_params.hstack(justify="start", gap=1, wrap=True),
            run_sim,
        ],
        gap=0.8,
    )
    return


@app.cell
def _(
    PSTConfig,
    final_values,
    make_schedule,
    mo,
    n_players,
    np,
    pd,
    run_sim,
    set_sim,
    sim_blocks,
    sim_params,
    simulate_players,
    simulate_test_phase,
    spread,
    vary_players,
):
    mo.stop(not run_sim.value)
    _theta = {name: float(value) for name, value in sim_params.value.items()}
    _n, _blocks = int(n_players.value), int(sim_blocks.value)
    _seed = int(np.random.default_rng().integers(2**31 - 1))
    _players = vary_players(_theta, n_players=_n, spread=float(spread.value), seed=_seed)
    _config = PSTConfig(min_blocks=_blocks, max_blocks=_blocks)
    with mo.status.spinner(title="Simulating players…"):
        _learning = simulate_players(_players, config=_config, n_blocks=_blocks, seed=_seed)
        _final = [
            simulate_test_phase(
                final_values(_learning[_learning["participant_id"] == i], p),
                p,
                make_schedule(_config, seed=_seed + i),
                last_learning_trial=_blocks * _config.block_length,
                participant_id=i,
                seed=_seed + i,
            )
            for i, p in enumerate(_players)
        ]
    set_sim(
        {
            "df": pd.concat([_learning, *_final], ignore_index=True).assign(source="Simulated"),
            "theta": _theta,
            "players": _players,
            "spread": float(spread.value),
            "n": _n,
            "blocks": _blocks,
        }
    )
    return


@app.cell
def _(get_sim, mo, pa, plots):
    _sim = get_sim()
    if _sim is None:
        _out = None
    else:
        _stats = pa.session_stats(_sim["df"])
        _acc = " · ".join(f"{pair} {acc:.0%}" for pair, acc in sorted(_stats["last_block_accuracy"].items()))
        _variation = f", with individual differences (spread {_sim['spread']:.2f})" if _sim["spread"] > 0 else ""
        _out = mo.vstack(
            [
                mo.md(
                    f"Simulated **{_sim['n']} player(s)** × {_sim['blocks'] * 60} choices{_variation}. "
                    f"Last block: {_acc}; average response time {_stats['mean_rt']:.2f} s; "
                    f"{_stats['timeouts']} deadline omission(s). These players now also "
                    "appear in step 2."
                ),
                plots.learning_curve_chart(pa.learning_curves(_sim["df"]), title="Simulated players"),
            ]
        )
    _out
    return


@app.cell
def _(mo):
    fit_source = mo.ui.radio(["Your game", "Simulated players"], value="Your game", inline=True, label="Data to fit")
    fit_model = mo.ui.dropdown(
        {
            "Model 6: learning from wins and losses, changing caution": "model6",
            "Simple: one learning rate, fixed caution": "simple",
        },
        value="Model 6: learning from wins and losses, changing caution",
        label="Model",
    )
    fit_settings = mo.ui.dictionary(
        {
            "draws": mo.ui.number(100, 2000, 100, 300, label="Posterior draws"),
            "tune": mo.ui.number(100, 2000, 100, 300, label="Warm-up draws"),
            "chains": mo.ui.number(2, 4, 1, 4, label="Chains"),
        }
    )
    run_fit = mo.ui.run_button(label="Fit the model", kind="success")
    return fit_model, fit_settings, fit_source, run_fit


@app.cell
def _(fit_model, fit_settings, fit_source, mo, run_fit, step_header):
    mo.vstack(
        [
            step_header(
                4,
                "Fit",
                "HSSM estimates one session directly or uses a hierarchical, partially pooled model for multiple players.",
                "fit",
            ),
            mo.hstack([fit_source, fit_model], justify="start", align="end", gap=2, wrap=True),
            mo.accordion({"Sampler settings": fit_settings.hstack(justify="start", gap=1)}),
            run_fit,
        ],
        gap=0.8,
    )
    return


@app.cell
def _(
    PSTLearner,
    fit_diagnostics,
    fit_model,
    fit_pst,
    fit_settings,
    fit_source,
    get_sim,
    human_df,
    mo,
    note,
    pa,
    run_fit,
    set_fit,
):
    import html as _html

    mo.stop(not run_fit.value)
    _sim = get_sim()
    _data = human_df if fit_source.value == "Your game" else (None if _sim is None else _sim["df"])
    mo.stop(
        _data is None or _data.empty,
        note("No data to fit yet: play the game (step 1) or simulate players (step 3).", kind="warn"),
    )
    _learner = PSTLearner() if fit_model.value == "model6" else PSTLearner(dual_learning_rates=False, boundary="fixed")
    _table, _info = pa.fit_table(_data)
    _settings = {name: int(value) for name, value in fit_settings.value.items()}
    try:
        with mo.status.spinner(
            title="Fitting HSSM's built-in DDM LAN…",
            subtitle=f"{len(_table)} choices; usually one to several minutes",
        ):
            _fit = fit_pst(_table, learner=_learner, seed=0, **_settings)
    except Exception as _error:  # show data or sampler problems in the app, not as a traceback
        mo.stop(True, note(_html.escape(str(_error)), kind="danger", label="The fit failed:"))
    _truth = None
    if fit_source.value == "Simulated players" and fit_model.value == "model6":
        _truth = _sim["players"][0] if _sim["n"] == 1 else _sim["theta"]
    set_fit(
        {
            "fit": _fit,
            "data": _data,
            "rows": pa.fit_rows(_data),
            "info": _info,
            "source": fit_source.value,
            "truth": _truth,
            "spread": None if _sim is None else _sim["spread"],
            "near_ceiling": pa.near_ceiling(_data),
            "diagnostics": fit_diagnostics(_fit),
        }
    )
    return


@app.cell
def _(
    get_fit,
    mo,
    note,
    np,
    pa,
    personal_readout,
    plots,
    posterior_html,
    posterior_predictive_curves,
    posterior_table,
    replay_latents,
    tiles,
):
    _state = get_fit()
    if _state is None:
        _out = note(
            "No fit yet. Choose the data and press <strong>Fit the model</strong>; your game takes about half a minute.",
            kind="empty",
        )
    else:
        _fit, _info, _rows = _state["fit"], _state["info"], _state["rows"]
        _diag = _state["diagnostics"]
        _table = posterior_table(_fit)
        _note = f"Fitted **{_state['source'].lower()}**: {_info['kept']} choices in {_fit.seconds:.0f} s"
        if _info["timeouts"] or _info["anticipations"]:
            _note += (
                f"; excluded {_info['timeouts']} deadline omission(s) and {_info['anticipations']} anticipation(s)"
            )
        if _info["balancing"]:
            _note += f"; removed {_info['balancing']} trailing row(s) to meet RLSSM's balanced-panel requirement"
        _note += "; hierarchical partial pooling" if _fit.hierarchical else "; one-session model"
        _parts = [mo.md(_note + ".")]
        _parts.append(
            note(
                "Inference uses HSSM's built-in approximate DDM likelihood network. Deadline trials are explicit "
                "omissions in the task and simulator, but current RLSSM cannot fit censored trials. Omissions are "
                "therefore excluded; the fitted likelihood does not correct for deadline truncation. The regularizing "
                "priors are choices made for this teaching app, not the cited paper's hierarchical priors.",
                label="Model scope:",
            )
        )
        if _info["mixed_response_methods"]:
            _parts.append(
                note(
                    "This session mixes key and click responses. The method is retained in the data, but the model "
                    "does not include a method-specific motor-time effect; non-decision time is not interpreted.",
                    kind="warn",
                    label="Mixed input methods.",
                )
            )
        if _state["near_ceiling"]:
            _parts.append(
                note(
                    "Every pair was chosen correctly at least 95% of the time in the last block. The model then cannot "
                    "tell fast learning from strong value sensitivity, so read the learning rates and m with caution.",
                    kind="warn",
                    label="Near-perfect play.",
                )
            )
        if _diag["ok"]:
            _bfmi = f"{_diag['min_bfmi']:.2f}" if np.isfinite(_diag["min_bfmi"]) else "not available"
            _parts.append(
                note(
                    f"No divergences; maximum r̂ {_diag['max_r_hat']:.3f}; minimum bulk ESS "
                    f"{_diag['min_ess_bulk']:.0f}; minimum tail ESS {_diag['min_ess_tail']:.0f}; "
                    f"minimum BFMI {_bfmi}.",
                    kind="success",
                    label="Sampler checks passed.",
                )
            )
        else:
            _parts.append(
                note(
                    " ".join(_diag["issues"]) + " Predictive and personal interpretations are withheld.",
                    kind="danger",
                    label="Sampler checks did not pass.",
                )
            )
        _parts.append(posterior_html(_table))
        if _diag["ok"]:
            with mo.status.spinner(title="Generating native SSMS posterior-predictive datasets…"):
                _ppc = posterior_predictive_curves(_fit, _rows, n_draws=40)
            _first_id = _rows["participant_id"].iloc[0]
            _first = _rows[_rows["participant_id"] == _first_id]
            _participant_draws = _fit.participant_draws(np.arange(_fit.n_samples))
            _estimates = {name: float(np.median(values[:, 0])) for name, values in _participant_draws.items()}
            _whose = "your session" if _state["source"] == "Your game" else "first simulated player"
            _views = {
                "Posterior predictive check": mo.vstack(
                    [
                        plots.learning_curve_chart(
                            pa.learning_curves(_state["data"], rows=_rows.drop(columns="source", errors="ignore")),
                            _ppc["choice"],
                            title="Observed choices and replicated choices (90% PPC band)",
                        ),
                        plots.rt_chart(
                            pa.rt_curves(_state["data"], rows=_rows.drop(columns="source", errors="ignore")),
                            _ppc["rt"],
                            title="Observed RTs and replicated RTs (90% PPC band)",
                        ),
                        mo.md(
                            f"<small>Each dashed curve comes from replicated choices and RTs generated by "
                            f"`ssms.rl.Simulator(...).simulate(mode='ppc')`, conditional on observed learning "
                            f"history. Replicated deadline-omission rate: {_ppc['timeout_rate']:.1%}.</small>"
                        ),
                    ]
                ),
                "Learned values": plots.values_chart(
                    replay_latents(_first, _estimates, _fit.learner),
                    title=f"Symbol values implied by the fitted model ({_whose})",
                ),
            }
            if _state["source"] == "Your game":
                _views["Session-level summary"] = tiles(
                    [
                        (item["topic"], item["value"], item["text"])
                        for item in personal_readout(
                            _fit, mixed_response_methods=_info["mixed_response_methods"]
                        )
                    ],
                    sentences=True,
                )
            _parts.append(mo.ui.tabs(_views))
        _out = mo.vstack(_parts, gap=0.8)
    _out
    return


@app.cell
def _(get_fit, mo, note, plots, posterior_html, posterior_table, step_header):
    _state = get_fit()
    _header = step_header(
        5,
        "Single-dataset recovery illustration",
        "For simulated data only: compare one fitted dataset with its known generating settings.",
        "recovery",
    )
    if _state is None or _state["truth"] is None:
        _body = note(
            "To illustrate recovery, <strong>simulate players</strong> in step 3, then fit "
            "<strong>Simulated players</strong> with <strong>Model 6</strong> in step 4. One dataset is an internal "
            "consistency check, not evidence about bias, interval coverage, or general recoverability; those require "
            "many repeated simulations across a parameter grid.",
            kind="empty",
        )
    else:
        if not _state["diagnostics"]["ok"]:
            _body = mo.vstack(
                [
                    note(
                        "Recovery scoring is withheld because sampler diagnostics did not pass. "
                        + " ".join(_state["diagnostics"]["issues"]),
                        kind="danger",
                    ),
                    posterior_html(posterior_table(_state["fit"])),
                ],
                gap=0.8,
            )
        else:
            _table = posterior_table(_state["fit"], _state["truth"])
            _hits = int(_table["recovered"].sum())
            _notes = (
                "This is one stochastic fit under the app's regularizing priors, not a calibration study. Learning "
                "rates are usually the hardest parameters to identify; repeated simulations are needed to estimate "
                "bias and coverage."
            )
            if _state["fit"].hierarchical:
                _notes += (
                    " Multiple players were fit with participant-level random effects and partial pooling; red ticks "
                    "show the generating group centres, and intervals summarize the fitted group effect."
                )
            _body = mo.vstack(
                [
                    note(
                        "generating centre values fall inside this fit's 94% intervals. This count is descriptive "
                        "for this dataset only.",
                        kind="hint",
                        label=f"{_hits} of {len(_table)}",
                    ),
                    plots.recovery_chart(_table),
                    posterior_html(_table),
                    mo.md(_notes),
                ],
                gap=0.8,
            )
    mo.vstack([_header, _body], gap=0.8)
    return


if __name__ == "__main__":
    app.run()
