# Builds standalone jsPsych 7 runner HTML (CDN plugins, timeline JSON, custom glue scripts).
# ``RunnerConfig`` selects plugins, extra stimulus scripts, and iframe behavior for marimo embeds.
# Bridges Python-assembled timelines to in-browser ``jsPsych.run()`` via ``jspsych_runner_core.js``.

from __future__ import annotations

import base64
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from runtime.jspsych_plugins import (
    DEFAULT_PLUGINS,
    JSPSYCH_CORE_CSS,
    JSPSYCH_CORE_JS,
    JSPSYCH_PLUGIN_CDN,
)

_RUNTIME_DIR = Path(__file__).resolve().parent


_PROJECT_ROOT = _RUNTIME_DIR.parent


@dataclass(frozen=True)
class RunnerConfig:
    """Browser runner options injected as base64 JSON."""

    title: str = "jsPsych Runtime"
    display_element: str = "jspsych-target"
    plugins: tuple[str, ...] = DEFAULT_PLUGINS
    extra_scripts: tuple[str, ...] = ()  # runtime/ or renderers/ paths
    extra_styles: tuple[str, ...] = ()  # runtime/ or renderers/ paths
    input_arrow_keys: bool = False
    focus_guard: bool = False  # dim + pause while the iframe cannot receive key presses
    results_message_type: str = "jspsych-results"
    results_session_id: str | None = None  # nonce used to bind results to one iframe run
    show_results_charts: bool = False
    results_task_filter: str | None = None
    results_view: str = "JsPsychDemoCharts"  # global JS object whose mount(rows) draws the end screen
    load_vega: bool = True  # Vega-Lite scripts for chart-based results views
    revive_keys: tuple[str, ...] = field(
        default_factory=lambda: (
            "on_finish",
            "on_start",
            "on_load",
            "stimulus",
            "conditional_function",
            "loop_function",
            "on_timeline_start",
            "on_timeline_finish",
        )
    )


def _runner_template() -> str:
    return (_RUNTIME_DIR / "jspsych_runner.html").read_text(encoding="utf-8")


def _runner_css() -> str:
    return (_RUNTIME_DIR / "jspsych_runner.css").read_text(encoding="utf-8")


def _runner_core_js() -> str:
    return (_RUNTIME_DIR / "jspsych_runner_core.js").read_text(encoding="utf-8")


def _runner_boot_js() -> str:
    return (_RUNTIME_DIR / "jspsych_runner_boot.js").read_text(encoding="utf-8")


def _resolve_project_path(relative_path: str) -> Path:
    if relative_path.startswith("renderers/"):
        return _PROJECT_ROOT / relative_path
    return _RUNTIME_DIR / relative_path


def _load_project_asset(relative_path: str) -> str:
    path = _resolve_project_path(relative_path)
    return path.read_text(encoding="utf-8")


def _load_runtime_script(relative_path: str) -> str:
    return _load_project_asset(relative_path)


def _encode_json_b64(payload: object) -> str:
    timeline_json = json.dumps(payload, separators=(",", ":"))
    return base64.b64encode(timeline_json.encode("utf-8")).decode("ascii")


def _plugin_script_tags(plugins: tuple[str, ...]) -> str:
    lines: list[str] = []
    for name in plugins:
        url = JSPSYCH_PLUGIN_CDN.get(name)
        if url is None:
            raise ValueError(f"Unknown jsPsych plugin: {name}")
        lines.append(f'  <script src="{url}"></script>')
    return "\n".join(lines)


def _vega_script_tags() -> str:
    return (
        '  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>\n'
        '  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>\n'
        '  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>'
    )


def _extra_style_blocks(extra_styles: tuple[str, ...]) -> str:
    blocks: list[str] = []
    for rel in extra_styles:
        css = _load_project_asset(rel)
        blocks.append(css)
    return "\n".join(blocks)


def _extra_script_blocks(extra_scripts: tuple[str, ...]) -> str:
    blocks: list[str] = []
    for rel in extra_scripts:
        script = _load_project_asset(rel)
        blocks.append(f"  <script>\n{script}\n  </script>")
    return "\n".join(blocks)


def build_jspsych_runner_html(
    timeline: list[dict[str, object]],
    *,
    config: RunnerConfig | None = None,
    title: str | None = None,
) -> str:
    """Build a standalone jsPsych HTML runner page from template assets."""
    cfg = config or RunnerConfig()
    if title is not None:
        cfg = dataclasses.replace(cfg, title=title)

    timeline_b64 = _encode_json_b64(timeline)
    config_payload = {
        "display_element": cfg.display_element,
        "plugins": list(cfg.plugins),
        "input_arrow_keys": cfg.input_arrow_keys,
        "focus_guard": cfg.focus_guard,
        "results_message_type": cfg.results_message_type,
        "results_session_id": cfg.results_session_id,
        "show_results_charts": cfg.show_results_charts,
        "results_task_filter": cfg.results_task_filter,
        "results_view": cfg.results_view,
        "revive_keys": list(cfg.revive_keys),
    }
    config_b64 = _encode_json_b64(config_payload)
    title_safe = cfg.title.replace("<", "&lt;").replace(">", "&gt;")

    boot_js = (
        _runner_boot_js()
        .replace("__RUNNER_CONFIG_B64__", config_b64)
        .replace("__TIMELINE_B64__", timeline_b64)
    )

    return (
        _runner_template()
        .replace("__TITLE_SAFE__", title_safe)
        .replace("__JSPSYCH_CORE_CSS__", JSPSYCH_CORE_CSS)
        .replace("__JSPSYCH_CORE_JS__", JSPSYCH_CORE_JS)
        .replace(
            "__VEGA_SCRIPT_TAGS__",
            _vega_script_tags() if cfg.show_results_charts and cfg.load_vega else "",
        )
        .replace("__PLUGIN_SCRIPT_TAGS__", _plugin_script_tags(cfg.plugins))
        .replace("__RUNNER_CSS__", _runner_css())
        .replace("__EXTRA_STYLE_BLOCKS__", _extra_style_blocks(cfg.extra_styles))
        .replace("__RUNNER_CORE_JS__", _runner_core_js())
        .replace("__EXTRA_SCRIPT_BLOCKS__", _extra_script_blocks(cfg.extra_scripts))
        .replace("__RUNNER_BOOT_JS__", boot_js)
    )
