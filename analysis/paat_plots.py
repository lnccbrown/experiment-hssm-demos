"""Compact, conference-friendly Altair views for PAAT data."""

from __future__ import annotations

import pandas as pd

TRIAL_TYPES = ["Incongruent", "Congruent"]
TRIAL_COLORS = ["#c2410c", "#0f766e"]

RESPONSIVE_AUTOSIZE = {"type": "fit", "contains": "padding", "resize": True}


def _altair():
    import altair as alt

    return alt


def _display_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "source" not in data.columns:
        data["source"] = "Observed"
    if "congruency" in data.columns:
        data["trial_type"] = data["congruency"].astype(str).str.capitalize()
    if "response" in data.columns:
        data["choice"] = data["response"].map({-1: "Safe", 1: "Risky"}).fillna("No response")
    return data


def _trial_color(alt, *, legend=None):
    return alt.Color(
        "trial_type:N",
        title="Trial type",
        sort=TRIAL_TYPES,
        scale=alt.Scale(domain=TRIAL_TYPES, range=TRIAL_COLORS),
        legend=legend,
    )


def _style(chart):
    return (
        chart.configure_axis(
            domainColor="#cbd5e1",
            gridColor="#e2e8f0",
            gridOpacity=0.8,
            labelColor="#475569",
            labelFontSize=12,
            titleColor="#334155",
            titleFontSize=13,
            titlePadding=10,
        )
        .configure_legend(
            labelColor="#475569",
            labelFontSize=12,
            symbolSize=120,
            titleColor="#334155",
            titleFontSize=12,
        )
        .configure_title(
            anchor="start",
            color="#172033",
            fontSize=17,
            fontWeight=600,
            subtitleColor="#64748b",
            subtitleFontSize=12,
            subtitlePadding=5,
        )
        .configure_view(strokeOpacity=0)
    )


def _empty_chart(title: str):
    alt = _altair()
    chart = (
        alt.Chart(pd.DataFrame({"message": ["No answered trials yet"]}))
        .mark_text(color="#64748b", fontSize=13)
        .encode(text="message:N")
        .properties(width="container", height=145, title=title, autosize=RESPONSIVE_AUTOSIZE)
    )
    return _style(chart)


def _source_encodings(alt, data: pd.DataFrame) -> dict[str, object]:
    sources = list(dict.fromkeys(data["source"].astype(str)))
    if len(sources) <= 1:
        return {}
    return {
        "yOffset": alt.YOffset("source:N", sort=sources, title=None),
        "opacity": alt.Opacity(
            "source:N",
            title="Data",
            sort=sources,
            scale=alt.Scale(domain=sources, range=[1.0, 0.55]),
            legend=alt.Legend(orient="top", direction="horizontal"),
        ),
    }


def condition_chart(summary: pd.DataFrame):
    """Horizontal risky-choice bars that use the available panel width."""
    if summary.empty:
        return _empty_chart("Risky choice rate")
    alt = _altair()
    data = _display_data(summary)
    sources = list(dict.fromkeys(data["source"].astype(str)))
    common = {
        "y": alt.Y("trial_type:N", title=None, sort=TRIAL_TYPES, axis=alt.Axis(labelPadding=8)),
        "color": _trial_color(alt, legend=None),
        **_source_encodings(alt, data),
    }
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=6, size=22).encode(
        x=alt.X(
            "p_risky:Q",
            title="Proportion choosing the risky option",
            scale=alt.Scale(domain=[0, 1.05]),
            axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1.0]),
        ),
        tooltip=[
            alt.Tooltip("source:N", title="Data"),
            alt.Tooltip("trial_type:N", title="Trial type"),
            alt.Tooltip("p_risky:Q", title="Risky choices", format=".1%"),
            alt.Tooltip("n:Q", title="Trials"),
        ],
        **common,
    )
    labels = alt.Chart(data).mark_text(
        align="left",
        baseline="middle",
        dx=6,
        color="#334155",
        fontSize=12,
    ).encode(
        x=alt.X("p_risky:Q"),
        text=alt.Text("p_risky:Q", format=".0%"),
        **{key: value for key, value in common.items() if key != "opacity"},
    )
    midpoint = (
        alt.Chart(pd.DataFrame({"midpoint": [0.5]}))
        .mark_rule(color="#94a3b8", strokeDash=[5, 4], strokeWidth=1.2)
        .encode(x="midpoint:Q")
    )
    chart = (midpoint + bars + labels).properties(
        width="container",
        height=max(145, 54 * len(sources)),
        title=alt.TitleParams("Risky choice rate", subtitle="The dashed line marks no choice preference"),
        autosize=RESPONSIVE_AUTOSIZE,
    )
    return _style(chart)


def evidence_chart(summary: pd.DataFrame, predictor: str):
    """Full-width evidence-sensitivity line chart with consistent encodings."""
    if predictor not in ("rel_reward", "rel_aversive"):
        raise ValueError("predictor must be rel_reward or rel_aversive")
    title = "Reward evidence" if predictor == "rel_reward" else "Aversive evidence"
    if summary.empty:
        return _empty_chart(title)
    alt = _altair()
    data = _display_data(summary)
    sources = list(dict.fromkeys(data["source"].astype(str)))
    x_title = (
        "Risky − safe reward probability"
        if predictor == "rel_reward"
        else "Risky − safe aversive probability"
    )
    subtitle = (
        "Positive values mean more reward on the risky option"
        if predictor == "rel_reward"
        else "Larger values mean a stronger aversive disadvantage for the risky option"
    )
    encoding: dict[str, object] = {
        "x": alt.X(
            f"{predictor}:Q",
            title=x_title,
            axis=alt.Axis(format="+%", tickCount=9),
        ),
        "y": alt.Y(
            "p_risky:Q",
            title="Proportion choosing risky",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", tickCount=6),
        ),
        "color": _trial_color(
            alt,
            legend=alt.Legend(orient="top", direction="horizontal", title=None),
        ),
        "tooltip": [
            alt.Tooltip("source:N", title="Data"),
            alt.Tooltip("trial_type:N", title="Trial type"),
            alt.Tooltip(f"{predictor}:Q", title=x_title, format="+.0%"),
            alt.Tooltip("p_risky:Q", title="Risky choices", format=".1%"),
            alt.Tooltip("n:Q", title="Trials"),
        ],
    }
    if len(sources) > 1:
        encoding["strokeDash"] = alt.StrokeDash(
            "source:N",
            title="Data",
            sort=sources,
            legend=alt.Legend(orient="top", direction="horizontal"),
        )
    line = alt.Chart(data).mark_line(strokeWidth=3).encode(**encoding)
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=72, stroke="white", strokeWidth=1)
        .encode(**encoding)
    )
    midpoint = (
        alt.Chart(pd.DataFrame({"midpoint": [0.5]}))
        .mark_rule(color="#94a3b8", strokeDash=[5, 4], strokeWidth=1.2)
        .encode(y="midpoint:Q")
    )
    zero = (
        alt.Chart(pd.DataFrame({"zero": [0.0]}))
        .mark_rule(color="#cbd5e1", strokeWidth=1)
        .encode(x="zero:Q")
    )
    chart = (midpoint + zero + line + points).properties(
        width="container",
        height=285,
        title=alt.TitleParams(title, subtitle=subtitle),
        autosize=RESPONSIVE_AUTOSIZE,
    )
    return _style(chart)


def rt_chart(summary: pd.DataFrame):
    """Horizontal median-RT bars matched visually to the choice chart."""
    if summary.empty:
        return _empty_chart("Median response time")
    alt = _altair()
    data = _display_data(summary)
    sources = list(dict.fromkeys(data["source"].astype(str)))
    common = {
        "y": alt.Y("trial_type:N", title=None, sort=TRIAL_TYPES, axis=alt.Axis(labelPadding=8)),
        "color": _trial_color(alt, legend=None),
        **_source_encodings(alt, data),
    }
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=6, size=22).encode(
        x=alt.X(
            "rt_median:Q",
            title="Median response time (seconds)",
            scale=alt.Scale(zero=True, nice=True),
            axis=alt.Axis(format=".1f", tickCount=8),
        ),
        tooltip=[
            alt.Tooltip("source:N", title="Data"),
            alt.Tooltip("trial_type:N", title="Trial type"),
            alt.Tooltip("rt_median:Q", title="Median RT", format=".2f"),
            alt.Tooltip("n:Q", title="Trials"),
        ],
        **common,
    )
    labels = alt.Chart(data).mark_text(
        align="left",
        baseline="middle",
        dx=6,
        color="#334155",
        fontSize=12,
    ).encode(
        x=alt.X("rt_median:Q"),
        text=alt.Text("rt_median:Q", format=".2f"),
        **{key: value for key, value in common.items() if key != "opacity"},
    )
    chart = (bars + labels).properties(
        width="container",
        height=max(145, 54 * len(sources)),
        title=alt.TitleParams(
            "Median response time",
            subtitle="Choice time only; post-choice animation is excluded",
        ),
        autosize=RESPONSIVE_AUTOSIZE,
    )
    return _style(chart)


def signed_rt_chart(rows: pd.DataFrame):
    """Trial-level RT rug: safe choices extend left and risky choices right."""
    if rows.empty:
        return _empty_chart("Choice and response time")
    alt = _altair()
    data = _display_data(rows)
    sources = list(dict.fromkeys(data["source"].astype(str)))
    values = pd.to_numeric(data["signed_rt"], errors="coerce").dropna().abs()
    limit = max(0.5, float(values.max()) * 1.08) if not values.empty else 1.0
    common: dict[str, object] = {
        "x": alt.X(
            "signed_rt:Q",
            title="Response time (seconds): safe ← 0 → risky",
            scale=alt.Scale(domain=[-limit, limit]),
            axis=alt.Axis(format=".1f", tickCount=9),
        ),
        "y": alt.Y("trial_type:N", title=None, sort=TRIAL_TYPES, axis=alt.Axis(labelPadding=8)),
        "color": _trial_color(alt, legend=None),
    }
    if len(sources) > 1:
        common.update(_source_encodings(alt, data))
    ticks = alt.Chart(data).mark_tick(thickness=2, size=34, opacity=0.48).encode(
        tooltip=[
            alt.Tooltip("source:N", title="Data"),
            alt.Tooltip("trial_type:N", title="Trial type"),
            alt.Tooltip("choice:N", title="Choice"),
            alt.Tooltip("rt:Q", title="RT", format=".2f"),
        ],
        **common,
    )
    medians = data.groupby(["source", "trial_type", "congruency"], as_index=False)[
        "signed_rt"
    ].median()
    median_encoding = {
        key: value for key, value in common.items() if key not in ("x", "opacity")
    }
    median_points = (
        alt.Chart(medians)
        .mark_point(filled=True, shape="diamond", size=115, stroke="white", strokeWidth=1.2)
        .encode(
            x=alt.X("signed_rt:Q"),
            tooltip=[alt.Tooltip("signed_rt:Q", title="Median", format=".2f")],
            **median_encoding,
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"zero": [0.0]}))
        .mark_rule(color="#64748b", strokeWidth=1.4)
        .encode(x="zero:Q")
    )
    chart = (zero + ticks + median_points).properties(
        width="container",
        height=max(150, 54 * len(sources)),
        title=alt.TitleParams(
            "Choice and response time",
            subtitle="Each tick is a trial; diamonds mark medians",
        ),
        autosize=RESPONSIVE_AUTOSIZE,
    )
    return _style(chart)


def drift_surface_chart(grid: pd.DataFrame):
    """Heatmaps containing only predictor cells observed in each trial type."""
    if grid.empty:
        return _empty_chart("Trial-wise drift")
    alt = _altair()
    data = _display_data(grid)
    base = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X(
                "rel_reward_z:Q",
                title="Standardized relative reward",
                bin=alt.Bin(maxbins=31),
            ),
            y=alt.Y(
                "rel_aversive_z:Q",
                title="Standardized relative aversive",
                bin=alt.Bin(maxbins=31),
            ),
            color=alt.Color(
                "mean(v):Q",
                title="Drift toward risky",
                scale=alt.Scale(scheme="blueorange", domainMid=0),
            ),
            tooltip=[
                alt.Tooltip("trial_type:N", title="Trial type"),
                alt.Tooltip("rel_reward_z:Q", title="Reward evidence", format=".2f"),
                alt.Tooltip("rel_aversive_z:Q", title="Aversive evidence", format=".2f"),
                alt.Tooltip("mean(v):Q", title="Drift", format=".2f"),
                alt.Tooltip("sum(design_rows):Q", title="Design rows"),
            ],
        )
        .properties(width=300, height=255)
    )
    chart = (
        base.facet(
            column=alt.Column(
                "trial_type:N",
                title=None,
                sort=TRIAL_TYPES,
                header=alt.Header(
                    labelColor="#334155",
                    labelFontSize=13,
                    labelFontWeight=600,
                ),
            ),
            spacing=22,
        )
        .properties(
            title=alt.TitleParams(
                "Drift at observed simulated-design cells",
                subtitle=(
                    "Only predictor combinations present in each trial type are colored; blank cells are unsupported. "
                    "Warm colors favor risky; cool colors favor safe"
                ),
            )
        )
        .resolve_scale(color="shared")
    )
    return _style(chart)
