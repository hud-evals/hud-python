from __future__ import annotations

from hud.utils.naming import normalize_environment_name


def test_normalize_environment_name() -> None:
    assert normalize_environment_name("terminal-bench") == "terminal-bench"
    assert normalize_environment_name("My Cool_Bench") == "my-cool-bench"
    assert normalize_environment_name("bench@2.0!") == "bench20"
    assert normalize_environment_name("--hello--") == "hello"
    assert normalize_environment_name("a---b") == "a-b"
    assert normalize_environment_name("@#$") == "environment"
    assert normalize_environment_name("", default="converted") == "converted"
