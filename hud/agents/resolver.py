"""Model resolution - maps model strings to agent classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hud.agents.base import MCPAgent

__all__ = ["resolve_cls"]

_models_cache: list[dict[str, Any]] | None = None


def _fetch_gateway_models() -> list[dict[str, Any]]:
    """Fetch available models from HUD API (cached)."""
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    import httpx

    from hud.settings import settings

    if not settings.api_key:
        return []

    try:
        resp = httpx.get(
            f"{settings.hud_api_url}/models/",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models") or []
        _models_cache = models
        return models
    except Exception:
        return []


def resolve_cls(model: str) -> tuple[type[MCPAgent], dict[str, Any] | None]:
    """Resolve model string to (agent_class, gateway_info).

    Returns:
        (agent_class, None) for known AgentTypes
        (agent_class, gateway_model_info) for gateway models
    """
    from hud.types import AgentType

    # Known AgentType → no gateway info
    try:
        return AgentType(model).cls, None
    except ValueError:
        pass

    # Gateway lookup
    for m in _fetch_gateway_models():
        if model in (m.get("id"), m.get("name"), m.get("model_name")):
            agent_str = m.get("sdk_agent_type") or m["provider"]["default_sdk_agent_type"]
            return AgentType(agent_str).cls, m

    raise ValueError(f"Model '{model}' not found")
