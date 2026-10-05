# PST stimulus HTML: two-symbol choice cards (Hiragana or neutral shapes) for jsPsych and marimo.
# Browser-side scoring, points and the end screen live in pst_session.js; layout in pst_symbols.css.
# Paired with experiments/pst_demo/pst_stimulus_plugin.py for jsPsych trials.

from __future__ import annotations

import html
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

# Neutral shapes, all drawn in the same colour so colour cannot hint at "good" or "bad".
_SHAPE_MARKUP = {
    "circle": '<circle cx="50" cy="50" r="38"/>',
    "triangle": '<polygon points="50,10 91,84 9,84"/>',
    "square": '<rect x="15" y="15" width="70" height="70" rx="5"/>',
    "diamond": '<polygon points="50,6 94,50 50,94 6,50"/>',
    "pentagon": '<polygon points="50,7 92,38 76,89 24,89 8,38"/>',
    "hexagon": '<polygon points="28,11 72,11 94,50 72,89 28,89 6,50"/>',
    "star": '<polygon points="50,5 61,37 95,37 67,57 78,91 50,71 22,91 33,57 5,37 39,37"/>',
    "ring": '<circle cx="50" cy="50" r="31" fill="none" stroke="currentColor" stroke-width="13"/>',
}


@lru_cache(maxsize=1)
def pst_css() -> str:
    """Stylesheet shared by the jsPsych iframe and marimo views."""
    return (_PACKAGE_DIR / "pst_symbols.css").read_text(encoding="utf-8")


def glyph_html(glyph: str, symbol_set: str) -> str:
    """One symbol: a Hiragana character or an inline SVG shape."""
    if symbol_set == "shapes":
        return (
            f'<svg class="pst-shape" viewBox="0 0 100 100" fill="currentColor" role="img" '
            f'aria-label="{html.escape(glyph)}">{_SHAPE_MARKUP[glyph]}</svg>'
        )
    return f'<span class="pst-glyph">{html.escape(glyph)}</span>'


def _card(glyph: str, symbol_set: str, side: str | None = None) -> str:
    """A symbol card; cards with a side can be clicked instead of pressing that arrow key (pst_session.js)."""
    attrs = f' data-side="{side}" role="button" aria-label="Choose the {side} symbol"' if side else ""
    return f'<div class="pst-card"{attrs}>{glyph_html(glyph, symbol_set)}</div>'


def choice_screen_html(
    left_glyph: str,
    right_glyph: str,
    symbol_set: str,
    *,
    status_label: str = "",
    show_points: bool = False,
) -> str:
    """Two cards side by side, a status line (block, points) and the key hints."""
    points = '<span class="pst-status__points"></span>' if show_points else ""
    return (
        '<div class="pst-screen">'
        f'<div class="pst-status"><span class="pst-status__label">{html.escape(status_label)}</span>{points}</div>'
        '<div class="pst-cards">'
        f"{_card(left_glyph, symbol_set, 'left')}"
        f"{_card(right_glyph, symbol_set, 'right')}"
        "</div>"
        '<div class="pst-keys"><span><kbd>&larr;</kbd> left</span><span>right <kbd>&rarr;</kbd></span></div>'
        "</div>"
    )


def example_cards_html(glyphs: Iterable[str], symbol_set: str, *, clickable: bool = False) -> str:
    """Small cards for instruction screens (a clickable pair behaves like the choice screen)."""
    glyphs = list(glyphs)
    sides = ("left", "right") if clickable and len(glyphs) == 2 else (None,) * len(glyphs)
    cards = "".join(_card(g, symbol_set, side) for g, side in zip(glyphs, sides))
    return f'<div class="pst-example">{cards}</div>'


def symbol_gallery_html(items: Iterable[tuple[str, str, str]], symbol_set: str) -> str:
    """marimo view of a session's symbols: (glyph, label, caption) per card."""
    cards = "".join(
        '<div class="pst-gallery__item">'
        f"{glyph_html(glyph, symbol_set)}"
        f'<div class="pst-gallery__label">{html.escape(label)}</div>'
        f'<div class="pst-gallery__meta">{html.escape(caption)}</div>'
        "</div>"
        for glyph, label, caption in items
    )
    return f'<div class="pst-gallery">{cards}</div>'
