"""Deploy build-summary rendering and duration formatting."""

from __future__ import annotations

from typing import Any

from hud.cli.deploy import (
    _format_duration,
    display_build_summary,
)


def test_format_duration() -> None:
    assert _format_duration(45) == "45s"
    assert _format_duration(125) == "2m 5s"
    assert _format_duration(3725) == "1h 2m"


def test_display_build_summary_succeeded_with_lock() -> None:
    status_response: dict[str, Any] = {
        "status": "SUCCEEDED",
        "version": "1.0.0",
        "duration_seconds": 125,
        "uri": "org/img:1.0.0",
        "lock": {
            "tasks": [{"slug": "solve-one", "task": "solve", "args": {"n": 1}}],
            "environment": {"variables": {"required": ["API_KEY"], "optional": ["DEBUG"]}},
            "capabilities": [{"name": "ssh"}, "browser"],
        },
    }
    display_build_summary(status_response, "org/img", env_name="demo")


def test_display_build_summary_failed() -> None:
    display_build_summary({"status": "FAILED", "version": "x"}, "org/img")


def test_display_build_summary_unknown_status() -> None:
    display_build_summary({"status": "BUILDING", "image_name": "img"}, "org/img")
