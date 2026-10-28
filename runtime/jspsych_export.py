# Task-agnostic jsPsych → marimo export: postMessage bridge and row → DataFrame pipeline.

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pandas as pd

DEFAULT_MESSAGE_TYPE = "jspsych-results"

RowPredicate = Callable[[dict[str, Any]], bool]
RowMapper = Callable[[dict[str, Any]], dict[str, object] | None]


def flatten_jspsych_row(row: object) -> dict[str, Any]:
    """Merge jsPsych ``data`` into the top-level dict (custom fields often live there)."""
    if not isinstance(row, dict):
        raise TypeError(f"jsPsych row must be a dict, got {type(row).__name__}")
    nested = row.get("data")
    if isinstance(nested, dict):
        flat = dict(nested)
        for key, val in row.items():
            if key == "data":
                continue
            if key not in flat or flat.get(key) in (None, ""):
                flat[key] = val
        return flat
    return dict(row)


def parse_rows_json(rows: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    """Parse jsPsych ``.json()`` export or an already-parsed row list."""
    if isinstance(rows, str):
        rows = json.loads(rows)
    if not isinstance(rows, list):
        raise TypeError(f"jsPsych rows must be a list, got {type(rows).__name__}")
    return rows


def jspsych_rows_to_dataframe(
    rows: list[dict[str, Any]] | str,
    *,
    include_row: RowPredicate,
    row_to_record: RowMapper,
    columns: list[str],
) -> pd.DataFrame:
    """Build a dataframe from jsPsych rows using task-specific filter and record mapping."""
    records: list[dict[str, object]] = []
    for row in parse_rows_json(rows):
        flat = flatten_jspsych_row(row)
        if not include_row(flat):
            continue
        record = row_to_record(flat)
        if record is not None:
            records.append(record)
    return pd.DataFrame(records, columns=columns)


# --- marimo / anywidget transport (optional dependency: marimo, anywidget) ---

_BRIDGE_ESM_TEMPLATE = """
export default {{
  render({{ model, el }}) {{
    el.style.display = "none";

    const onMessage = (event) => {{
      const payload = event.data;
      if (!payload || payload.type !== {message_type}) return;
      if (payload.session_id !== {session_id}) return;
      const frame = document.getElementById({iframe_id});
      if (!frame || event.source !== frame.contentWindow) return;
      if (typeof payload.rows_json !== "string") {{
        throw new Error(
          "jspsych-results message must include rows_json from JsPsychRunnerCore",
        );
      }}
      model.set("rows_json", payload.rows_json);
      model.set("result_received", true);
      model.save_changes();
    }};
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }},
}};
"""


def _bridge_esm(
    *,
    message_type: str = DEFAULT_MESSAGE_TYPE,
    session_id: str | None = None,
    iframe_id: str | None = None,
) -> str:
    if session_id is None or iframe_id is None:
        raise ValueError("session_id and iframe_id are required for a source-bound results bridge")
    return _BRIDGE_ESM_TEMPLATE.format(
        message_type=json.dumps(message_type),
        session_id=json.dumps(session_id),
        iframe_id=json.dumps(iframe_id),
    )


def create_jspsych_marimo_bridge(
    *,
    message_type: str = DEFAULT_MESSAGE_TYPE,
    session_id: str,
    iframe_id: str,
):
    """Return a source-bound hidden bridge that syncs one iframe's completed export."""
    import anywidget
    import traitlets
    import marimo as mo

    class JsPsychMarimoBridge(anywidget.AnyWidget):
        _esm = _bridge_esm(message_type=message_type, session_id=session_id, iframe_id=iframe_id)
        rows_json = traitlets.Unicode("[]").tag(sync=True)
        result_received = traitlets.Bool(False).tag(sync=True)

    return mo.ui.anywidget(JsPsychMarimoBridge())
