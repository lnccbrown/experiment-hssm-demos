# Small HTML embedding helpers for marimo and other Python UIs.
# Wraps standalone HTML in sandboxed ``srcdoc`` iframes for safe in-notebook preview.

from __future__ import annotations

DEFAULT_IFRAME_STYLE = "border:1px solid #ddd;border-radius:6px;"


def render_srcdoc_iframe(
    html: str,
    *,
    title: str,
    height: int,
    style: str = DEFAULT_IFRAME_STYLE,
    iframe_id: str | None = None,
) -> str:
    """Embed standalone HTML in a marimo-safe iframe srcdoc attribute.

    ``title`` becomes the frame's accessible name (``aria-label``): a ``title`` attribute would show as a
    tooltip over the whole frame in Safari.
    """
    srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
    label = title.replace("&", "&amp;").replace('"', "&quot;")
    id_attr = "" if iframe_id is None else f' id="{iframe_id.replace("&", "&amp;").replace(chr(34), "&quot;")}"'
    return (
        f'<iframe{id_attr} aria-label="{label}" sandbox="allow-scripts" srcdoc="{srcdoc}" '
        f'width="100%" height="{int(height)}" '
        f'style="{style}"></iframe>'
    )
