from __future__ import annotations

import pytest

from hud.utils.env import resolve_env_vars


def test_resolve_env_vars_replaces_nested_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUD_TEST_VALUE", "resolved")

    assert resolve_env_vars({"a": "${HUD_TEST_VALUE}", "b": ["x-${HUD_TEST_VALUE}"]}) == {
        "a": "resolved",
        "b": ["x-resolved"],
    }


def test_resolve_env_vars_accepts_extra_mapping() -> None:
    assert resolve_env_vars("${VALUE}", {"VALUE": "resolved"}) == "resolved"


def test_resolve_env_vars_rejects_missing_placeholders() -> None:
    with pytest.raises(KeyError, match="MISSING_VALUE"):
        resolve_env_vars("${MISSING_VALUE}")
