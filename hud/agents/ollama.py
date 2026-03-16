"""Ollama Agent — local model support via OpenAI-compatible endpoint.

Connects to Ollama (and vLLM, LM Studio, llama.cpp) through their
OpenAI-compatible ``/v1/chat/completions`` endpoint by subclassing
:class:`OpenAIChatAgent`.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import httpx
from openai import AsyncOpenAI

from hud.settings import settings
from hud.types import AgentType, BaseAgentConfig
from hud.utils.types import with_signature

from .openai_chat import OpenAIChatAgent
from .types import OllamaConfig, OllamaCreateParams

logger = logging.getLogger(__name__)


class OllamaAgent(OpenAIChatAgent):
    """MCP-enabled agent for local models served by Ollama."""

    config_cls: ClassVar[type[BaseAgentConfig]] = OllamaConfig

    @classmethod
    def agent_type(cls) -> AgentType:
        return AgentType.OLLAMA

    @with_signature(OllamaCreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> OllamaAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        from .base import MCPAgent

        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: OllamaCreateParams | None = None, **kwargs: Any) -> None:
        # Resolve base_url from settings when not explicitly provided
        if params and params.base_url is None:
            params.base_url = settings.ollama_base_url
        elif not params:
            if kwargs.get("base_url") is None and kwargs.get("openai_client") is None:
                kwargs["base_url"] = settings.ollama_base_url

        # Ensure api_key defaults so AsyncOpenAI doesn't raise
        if params and params.api_key is None:
            params.api_key = "ollama"
        elif not params and kwargs.get("api_key") is None and kwargs.get("openai_client") is None:
            kwargs["api_key"] = "ollama"

        super().__init__(params, **kwargs)  # pyright: ignore[reportArgumentType]

    @property
    def _ollama_api_base(self) -> str:
        """Return the Ollama native API base (without /v1 suffix)."""
        base = self.config.base_url or settings.ollama_base_url
        return base.rstrip("/").removesuffix("/v1")

    async def check_health(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self._ollama_api_base}/api/tags", timeout=5.0)
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def list_models(self) -> list[str]:
        """List available models on the Ollama server."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self._ollama_api_base}/api/tags", timeout=5.0)
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
            return []
