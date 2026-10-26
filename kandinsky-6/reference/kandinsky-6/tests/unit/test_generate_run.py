from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from kandinsky.cli import _bind_generate_run, _write_launch, _write_run_prompt
from kandinsky.runtime.home import logs_dir, open_generate_run, profile_dir


def test_generate_run_opens_a_stamped_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KANDINSKY_HOME", str(tmp_path))
    root = open_generate_run()
    assert root.parent == tmp_path / "outputs"
    assert root.name.startswith("generate_")
    for folder in ("generations", "logs", "profiles"):
        assert (root / folder).is_dir()


def test_generate_run_keeps_the_expanded_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KANDINSKY_HOME", str(tmp_path))
    run = open_generate_run()
    video = run / "generations" / "output.mp4"
    video.write_bytes(b"")
    video.with_suffix(".txt").write_text("A red dragon exhales a stream of flame.\n", encoding="utf-8")

    written = _write_run_prompt(run, video, ["unused"])

    assert written == run / "expanded_prompt.txt"
    assert written.read_text(encoding="utf-8") == "A red dragon exhales a stream of flame.\n"
    assert not video.with_suffix(".txt").exists()


def test_generate_run_records_launch_and_binds_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KANDINSKY_HOME", str(tmp_path))
    run = open_generate_run()
    _write_launch(run, prompt="a cat", config="h100.yaml", seed=1, output="generations/output.mp4")
    payload = json.loads((run / "launch.json").read_text(encoding="utf-8"))
    assert payload["name"] == "generate"
    assert payload["prompt"] == "a cat"
    assert payload["output"] == "generations/output.mp4"

    log = logging.getLogger("kandinsky")
    before = list(log.handlers)
    propagate = log.propagate
    previous_log = os.environ.get("KANDINSKY_LOG_DIR")
    previous_profile = os.environ.get("KANDINSKY_PROFILE_DIR")
    try:
        _bind_generate_run(run)
        assert logs_dir() == run / "logs"
        assert profile_dir() == run / "profiles"
        log.info("generate-run-marker")
        assert "generate-run-marker" in (run / "logs" / "kandinsky.log").read_text(encoding="utf-8")
    finally:
        for handler in list(log.handlers):
            if handler not in before:
                handler.close()
                log.removeHandler(handler)
        log.propagate = propagate
        _restore_env("KANDINSKY_LOG_DIR", previous_log)
        _restore_env("KANDINSKY_PROFILE_DIR", previous_profile)


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous
