# Altair charts for the PST app: learning curves, RTs, test-phase readouts, recovery, values.
# One colour per pair everywhere; solid = your data, dashed = simulated players or the model.

from __future__ import annotations

import altair as alt
import pandas as pd

PAIR_COLORS = {"AB": "#0f766e", "CD": "#b45309", "EF": "#6d28d9"}
PAIR_TITLES = {"AB": "AB (80/20)", "CD": "CD (70/30)", "EF": "EF (60/40)"}
SOURCE_DASHES = {"You": [1, 0], "Simulated": [6, 4]}
WIDTH, HEIGHT = 420, 260

_pair_color = alt.Color(
    "pair:N",
    title="Pair",
    scale=alt.Scale(domain=list(PAIR_COLORS), range=list(PAIR_COLORS.values())),
    legend=alt.Legend(
        orient="bottom",
        labelLimit=0,
        labelExpr=" + ".join(f"(datum.label == '{p}' ? '{t}' : '')" for p, t in PAIR_TITLES.items()),
    ),
)


def _dash(df: pd.DataFrame):
    if "source" not in df.columns:
        return alt.value([1, 0])
    domain = [s for s in SOURCE_DASHES if s in set(df["source"])]
    return alt.StrokeDash(
        "source:N",
        title="Data",
        scale=alt.Scale(domain=domain, range=[SOURCE_DASHES[s] for s in domain]),
        legend=alt.Legend(orient="bottom"),
    )


def _chance_rule(y: float = 0.5):
    return alt.Chart(pd.DataFrame({"y": [y]})).mark_rule(color="#94a3b8", strokeDash=[3, 3]).encode(y="y:Q")


def learning_curve_chart(observed: pd.DataFrame, predicted: pd.DataFrame | None = None, *, title: str = "Learning curves"):
    """Share of better-symbol choices per pair over presentations of that pair."""
    x = alt.X("trials:Q", title="Times the pair was shown")
    y = alt.Y("p_better:Q", title="Chose the better symbol", scale=alt.Scale(domain=[0, 1]))
    tooltip = [alt.Tooltip("pair:N"), alt.Tooltip("trials:Q", title="Presentations (bin centre)"),
               alt.Tooltip("p_better:Q", title="Better choices", format=".0%")]
    if "source" in observed.columns:
        tooltip.insert(0, alt.Tooltip("source:N", title="Data"))
    lines = alt.Chart(observed).mark_line(point=True, strokeWidth=2).encode(
        x=x, y=y, color=_pair_color, strokeDash=_dash(observed), tooltip=tooltip
    )
    layers = [_chance_rule(), lines]
    if predicted is not None and not predicted.empty:
        band = alt.Chart(predicted).mark_area(opacity=0.16).encode(x=x, y="low:Q", y2="high:Q", color=_pair_color)
        model = alt.Chart(predicted).mark_line(strokeDash=[6, 4], strokeWidth=1.5).encode(
            x=x, y=y, color=_pair_color,
            tooltip=[alt.Tooltip("pair:N"), alt.Tooltip("p_better:Q", title="Model prediction", format=".0%")],
        )
        layers = [_chance_rule(), band, model, lines]
    return alt.layer(*layers).properties(width=WIDTH, height=HEIGHT, title=title)


def rt_chart(curves: pd.DataFrame, predicted: pd.DataFrame | None = None, *, title: str = "Response times"):
    """Mean RT per pair, optionally overlaid with a native PPC interval."""
    x = alt.X("trials:Q", title="Times the pair was shown")
    y = alt.Y("rt_mean:Q", title="Mean response time (s)", scale=alt.Scale(zero=False))
    observed = (
        alt.Chart(curves)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=x,
            y=y,
            color=_pair_color,
            strokeDash=_dash(curves),
            tooltip=[
                alt.Tooltip("pair:N"),
                alt.Tooltip("rt_mean:Q", title="Mean RT (s)", format=".2f"),
                alt.Tooltip("n:Q", title="Choices"),
            ],
        )
    )
    layers = [observed]
    if predicted is not None and not predicted.empty:
        band = alt.Chart(predicted).mark_area(opacity=0.16).encode(
            x=x, y=alt.Y("low:Q", title="Mean response time (s)", scale=alt.Scale(zero=False)),
            y2="high:Q", color=_pair_color
        )
        model = alt.Chart(predicted).mark_line(strokeDash=[6, 4], strokeWidth=1.5).encode(
            x=x,
            y=y,
            color=_pair_color,
            tooltip=[alt.Tooltip("pair:N"), alt.Tooltip("rt_mean:Q", title="PPC mean RT", format=".2f")],
        )
        layers = [band, model, observed]
    return alt.layer(*layers).properties(width=WIDTH, height=HEIGHT, title=title)


def signed_rt_chart(signed: pd.DataFrame, *, title: str = "Response times by choice"):
    """Histogram of RTs per pair; negative = chose the worse symbol (paper, Fig. 4)."""
    return (
        alt.Chart(signed)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("signed_rt:Q", bin=alt.Bin(maxbins=40), title="RT (s); negative = chose the worse symbol"),
            y=alt.Y("count():Q", title="Choices"),
            color=_pair_color,
        )
        .properties(width=250, height=160)
        .facet(column=alt.Column("pair:N", title=None, sort=list(PAIR_COLORS)))
        .properties(title=title)
    )


def test_phase_chart(summary: pd.DataFrame, *, title: str = "Final round (no feedback)"):
    """Choose-A and avoid-B, with chance at 50%."""
    color = (
        alt.Color("source:N", title="Data", legend=alt.Legend(orient="bottom"))
        if "source" in summary.columns
        else alt.value("#334155")
    )
    offset = alt.XOffset("source:N") if "source" in summary.columns else alt.XOffset()
    bars = alt.Chart(summary).mark_bar(size=34).encode(
        x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=offset,
        y=alt.Y("p_better:Q", title="Share of choices", scale=alt.Scale(domain=[0, 1])),
        color=color,
        tooltip=[alt.Tooltip("label:N", title="Readout"), alt.Tooltip("p_better:Q", format=".0%"), alt.Tooltip("n:Q")],
    )
    return alt.layer(_chance_rule(), bars).properties(width=320, height=HEIGHT, title=title)


def recovery_chart(table: pd.DataFrame, *, title: str = "Estimates (94% interval) vs the values used to simulate"):
    """Per parameter: posterior interval and mean, with the true value as a red tick."""
    base = alt.Chart(table)
    y = alt.Y("parameter:N", title=None, sort=list(table["parameter"]), axis=None)
    interval = base.mark_rule(strokeWidth=3, color="#64748b").encode(
        x=alt.X("low:Q", title=None, scale=alt.Scale(zero=False)), x2="high:Q", y=y
    )
    estimate = base.mark_point(filled=True, size=70, color="#0f172a").encode(
        x="estimate:Q", y=y, tooltip=["parameter", alt.Tooltip("estimate:Q", format=".3f"), alt.Tooltip("true:Q", format=".3f")]
    )
    truth = base.mark_tick(color="#dc2626", thickness=3, size=22).encode(x="true:Q", y=y)
    return (
        alt.layer(interval, estimate, truth)
        .properties(width=320, height=34)
        .facet(row=alt.Row("parameter:N", title=None, sort=list(table["parameter"]), header=alt.Header(labelAngle=0, labelAlign="left")))
        .resolve_scale(x="independent", y="independent")
        .properties(title=title)
    )


def values_chart(latents: pd.DataFrame, *, title: str = "What the model thinks each symbol is worth"):
    """Learned value of every symbol over the learning phase (better symbols solid, worse dashed)."""
    long = latents.melt(id_vars=["trial"], value_vars=[f"V_{s}" for s in "ABCDEF"], var_name="symbol", value_name="value")
    long["symbol"] = long["symbol"].str[-1]
    long["pair"] = long["symbol"].map({"A": "AB", "B": "AB", "C": "CD", "D": "CD", "E": "EF", "F": "EF"})
    long["kind"] = long["symbol"].map(lambda s: "better" if s in "ACE" else "worse")
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("trial:Q", title="Learning trial"),
            y=alt.Y("value:Q", title="Learned value", scale=alt.Scale(domain=[0, 1])),
            color=_pair_color,
            strokeDash=alt.StrokeDash(
                "kind:N",
                title="Symbol",
                scale=alt.Scale(domain=["better", "worse"], range=[[1, 0], [5, 3]]),
                legend=alt.Legend(orient="bottom"),
            ),
            detail="symbol:N",
            tooltip=["symbol:N", alt.Tooltip("value:Q", format=".2f"), "trial:Q"],
        )
        .properties(width=WIDTH, height=HEIGHT, title=title)
    )
