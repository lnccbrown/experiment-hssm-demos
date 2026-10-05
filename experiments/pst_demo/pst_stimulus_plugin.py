# PST ``AFCStimulusPlugin``: two-symbol choice cards for the constant-stimuli n-AFC timeline.
# Registers ``pst_cards``; browser-side scoring (side, symbol, pre-drawn reward) is in pst_session.js.
# Imported for side effect by ``pst_timeline`` so the plugin is available by name.

from __future__ import annotations

from renderers.pst_symbols.pst_symbols import choice_screen_html
from schemas.contracts import Trial
from schemas.timelines.constant_stimuli_afc_timeline import (
    AFCStimulusPlugin,
    register_stimulus_plugin,
)

ARM_CARDS = "function(){ window.PSTSession.armCards(); }"
PST_ON_LOAD = "function(){ window.PSTSession.updatePoints(); window.PSTSession.armCards(); }"
PST_ON_FINISH = "function(data){ window.PSTSession.scoreTrial(data); }"


def render_pst_cards(trial: Trial, trial_index: int) -> str:
    display = trial["display_params"]
    return choice_screen_html(
        str(display["left_glyph"]),
        str(display["right_glyph"]),
        str(display["symbol_set"]),
        status_label=str(display.get("status_label", "")),
        show_points=trial["data"].get("phase") == "learning",
    )


PST_CARDS_PLUGIN = AFCStimulusPlugin(
    name="pst_cards",
    render_stimulus=render_pst_cards,
    on_load=PST_ON_LOAD,
    on_finish=PST_ON_FINISH,
)

register_stimulus_plugin(PST_CARDS_PLUGIN)
