"""Launch settings must reload the business package along with the UI script."""

import os
from pathlib import Path
import sys

import pytest
from streamlit import file_util

from rmbfund import cli


@pytest.mark.parametrize("pythonpath", [None, "/synthetic/existing:/synthetic/other"])
def test_ui_watches_business_modules_without_changing_workspace(tmp_path, monkeypatch, pythonpath):
    if pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", pythonpath)
    calls = []

    def launch(command, *, env):
        calls.append((command, env))
        return 7

    monkeypatch.setattr(cli.subprocess, "call", launch)
    root = tmp_path / "uncreated-workspace"
    assert cli.main(["--root", str(root), "--mode", "public", "ui", "--port", "8504"]) == 7
    command, env = calls[0]
    assert len(calls) == 1
    assert command == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(cli.__file__).parent / "ui/app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8504",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    assert env["RMBFUND_ROOT"] == str(root.resolve())
    assert env["RMBFUND_MODE"] == "public"
    assert env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
    source_root = str(Path(cli.__file__).resolve().parent.parent)
    existing_paths = pythonpath.split(os.pathsep) if pythonpath else []
    assert env["PYTHONPATH"].split(os.pathsep) == [source_root, *existing_paths]
    # Streamlit checks the explicit environment, not editable-install sys.path.
    monkeypatch.setenv("PYTHONPATH", env["PYTHONPATH"])
    assert file_util.file_in_pythonpath(str(Path(cli.__file__).parent / "portfolio.py"))
    assert file_util.file_in_pythonpath(str(Path(cli.__file__).parent / "service.py"))
    assert not root.exists()
