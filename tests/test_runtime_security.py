import base64
import json
import re

import pytest

from runtime.embed import render_srcdoc_iframe
from runtime.jspsych_export import _bridge_esm
from runtime.jspsych_runner import RunnerConfig, build_jspsych_runner_html


def test_srcdoc_iframe_is_sandboxed_and_identifiable():
    iframe = render_srcdoc_iframe("<script>ok()</script>", title="Task", height=400, iframe_id="task-abc")
    assert 'id="task-abc"' in iframe
    assert 'sandbox="allow-scripts"' in iframe
    assert "allow-same-origin" not in iframe


def test_results_bridge_requires_matching_nonce_and_iframe_source():
    with pytest.raises(ValueError, match="required"):
        _bridge_esm()
    source = _bridge_esm(message_type="pst-results", session_id="nonce-123", iframe_id="pst-frame")
    assert 'payload.session_id !== "nonce-123"' in source
    assert 'document.getElementById("pst-frame")' in source
    assert "event.source !== frame.contentWindow" in source


def test_runner_carries_session_nonce_in_encoded_config():
    html = build_jspsych_runner_html([], config=RunnerConfig(results_session_id="nonce-456"))
    encoded = re.search(r'decodeB64Json\("([A-Za-z0-9+/=]+)"\)', html)
    assert encoded is not None
    config = json.loads(base64.b64decode(encoded.group(1)))
    assert config["results_session_id"] == "nonce-456"
