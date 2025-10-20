from __future__ import annotations

from hud.cli.utils.logging import CaptureLogger, analyze_error_for_hints, is_port_free


def test_capture_logger_basic(capfd):
    logger = CaptureLogger(print_output=True)
    logger.success("done")
    logger.error("oops")
    logger.info("info")
    out = logger.get_output()
    assert "done" in out and "oops" in out and "info" in out


def test_analyze_error_for_hints_matches():
    hint = analyze_error_for_hints("ModuleNotFoundError: x")
    assert hint and "dependencies" in hint


def test_is_port_free_returns_bool():
    # Probe a high port; we only assert the function returns a boolean
    free = is_port_free(65500)
    assert isinstance(free, bool)
