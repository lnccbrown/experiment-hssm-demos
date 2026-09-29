import pandas as pd

from analysis import paat_analysis as pa
from analysis import paat_plots as plots
from observers.paat_ssm import PAATDriftCoefficients, simulate_players


def _data(source: str, seed: int):
    return simulate_players(profile="conference", n_players=1, seed=seed).assign(source=source)


def test_overview_charts_fill_their_containers_without_source_facets():
    summary = pa.condition_summary(_data("You", 1))
    for chart in (plots.condition_chart(summary), plots.rt_chart(summary)):
        spec = chart.to_dict(validate=True)
        assert spec["width"] == "container"
        assert spec["autosize"]["resize"] is True
        assert "facet" not in spec
        assert spec["height"] <= 180


def test_evidence_chart_only_adds_source_styling_for_comparisons():
    human = _data("You", 2)
    single = plots.evidence_chart(pa.evidence_summary(human, "rel_reward"), "rel_reward").to_dict(validate=True)
    combined = pd.concat([human, _data("Simulated", 3)], ignore_index=True)
    multiple = plots.evidence_chart(
        pa.evidence_summary(combined, "rel_reward"), "rel_reward"
    ).to_dict(validate=True)

    assert single["width"] == multiple["width"] == "container"
    assert "strokeDash" not in single["layer"][-1]["encoding"]
    assert "strokeDash" in multiple["layer"][-1]["encoding"]


def test_evidence_summary_coalesces_equivalent_probability_differences():
    row = _data("You", 4).iloc[[0]]
    data = pd.concat([row, row], ignore_index=True)
    data["rel_reward"] = [0.4, 0.4000000000000001]
    summary = pa.evidence_summary(data, "rel_reward")

    assert len(summary) == 1
    assert summary.loc[0, "n"] == 2


def test_drift_chart_marks_unobserved_predictor_cells_as_blank():
    design = _data("Simulated", 5)
    grid = pa.drift_grid(PAATDriftCoefficients(), design)
    spec = plots.drift_surface_chart(grid).to_dict(validate=True)

    assert "facet" in spec
    assert "blank cells are unsupported" in spec["title"]["subtitle"]
    assert len(grid) == len(
        design[
            ["congruency_code", "rel_reward_z", "rel_aversive_z"]
        ].drop_duplicates()
    )
