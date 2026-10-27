from __future__ import annotations

import io

from kandinsky.pipeline.progress import denoising_progress

_LOG_LINE_WIDTH = 80
_STEPS = ("0/3", "1/3", "2/3", "3/3")


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_log_progress_is_one_short_line_per_step() -> None:
    buf = io.StringIO()
    bar = denoising_progress(3, file=buf)
    bar.update()
    bar.update()
    bar.update()
    bar.close()

    text = buf.getvalue()
    lines = [line for line in text.splitlines() if line.strip()]

    assert "\r" not in text
    assert len(lines) == len(_STEPS)
    for line, step in zip(lines, _STEPS, strict=True):
        assert step in line
        assert len(line) <= _LOG_LINE_WIDTH


def test_terminal_progress_redraws_in_place() -> None:
    buf = _TTY()
    bar = denoising_progress(2, file=buf)
    bar.update()
    bar.close()

    assert "\r" in buf.getvalue()
