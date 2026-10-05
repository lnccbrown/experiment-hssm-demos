import shutil
import subprocess
from pathlib import Path

import pytest

JS_TESTS = sorted((Path(__file__).parent / "js").glob("*.test.mjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("script", JS_TESTS, ids=lambda path: path.name)
def test_browser_scripts(script):
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
