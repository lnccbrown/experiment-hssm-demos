"""HTML primitives for PAAT reward and aversive probability wheels."""

from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path

from schemas.tasks.paat import PAATTrialSpec

_PACKAGE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def paat_css() -> str:
    return (_PACKAGE_DIR / "paat_wheels.css").read_text(encoding="utf-8")


def _wheel(probability: float, kind: str) -> str:
    pct = round(100 * probability)
    label = "Reward" if kind == "reward" else "Aversive"
    return (
        f'<div class="paat-wheel-card paat-wheel-card--{kind}">'
        f'<div class="paat-wheel paat-wheel--{kind}" style="--paat-arc:{probability * 360:.3f}deg" '
        f'role="img" aria-label="{label} probability {pct} percent">'
        '<div class="paat-wheel__disc"></div>'
        f'<div class="paat-wheel__number">{pct}%</div>'
        "</div>"
        f'<div class="paat-wheel__label">{label}</div>'
        "</div>"
    )


def _option(side: str, p_reward: float, p_aversive: float, *, clickable: bool = True) -> str:
    attrs = (
        f' data-side="{html.escape(side)}" role="button" tabindex="0" '
        f'aria-label="Choose the {html.escape(side)} option"'
        if clickable
        else ""
    )
    return (
        f'<div class="paat-option"{attrs}>'
        f'<div class="paat-option__side">{html.escape(side.title())} option</div>'
        '<div class="paat-option__wheels">'
        f"{_wheel(p_reward, 'reward')}{_wheel(p_aversive, 'aversive')}"
        "</div></div>"
    )


def choice_screen_html(trial: PAATTrialSpec, *, status_label: str = "") -> str:
    if trial.risky_side == "left":
        left = (trial.p_reward_risky, trial.p_aversive_risky)
        right = (trial.p_reward_safe, trial.p_aversive_safe)
    else:
        left = (trial.p_reward_safe, trial.p_aversive_safe)
        right = (trial.p_reward_risky, trial.p_aversive_risky)
    return (
        '<div class="paat-screen">'
        '<div class="paat-status">'
        f'<span>{html.escape(status_label)}</span><span class="paat-status__reward"></span>'
        "</div>"
        '<div class="paat-prompt">Which option do you prefer?</div>'
        '<div class="paat-options">'
        f"{_option('left', *left)}{_option('right', *right)}"
        "</div>"
        '<div class="paat-keys"><span><kbd>&larr;</kbd> left</span><span>right <kbd>&rarr;</kbd></span></div>'
        "</div>"
    )


def example_option_html(p_reward: float = 0.7, p_aversive: float = 0.3) -> str:
    return f'<div class="paat-example">{_option("example", p_reward, p_aversive, clickable=False)}</div>'
