from __future__ import annotations

import pytest

from hud.capabilities.base import Capability


def test_capability_from_manifest_validates_required_shape() -> None:
    with pytest.raises(ValueError, match="'name'"):
        Capability.from_manifest({"protocol": "mcp/2025-11-25", "url": "http://host/mcp"})

    with pytest.raises(TypeError, match="'params'"):
        Capability.from_manifest(
            {"name": "tools", "protocol": "mcp/2025-11-25", "url": "http://host/mcp", "params": []}
        )


def test_capability_from_manifest_validates_protocol_params() -> None:
    with pytest.raises(TypeError, match="auth_token"):
        Capability.from_manifest(
            {
                "name": "tools",
                "protocol": "mcp/2025-11-25",
                "url": "http://host/mcp",
                "params": {"auth_token": 123},
            }
        )

    with pytest.raises(TypeError, match="display"):
        Capability.from_manifest(
            {
                "name": "screen",
                "protocol": "rfb/3.8",
                "url": "rfb://host:5900",
                "params": {"display": False},
            }
        )


def test_capability_from_manifest_preserves_valid_manifest() -> None:
    cap = Capability.from_manifest(
        {
            "name": "tools",
            "protocol": "mcp/2025-11-25",
            "url": "http://host/mcp",
            "params": {"auth_token": "token"},
        }
    )

    assert cap.to_manifest() == {
        "name": "tools",
        "protocol": "mcp/2025-11-25",
        "url": "http://host/mcp",
        "params": {"auth_token": "token"},
    }
