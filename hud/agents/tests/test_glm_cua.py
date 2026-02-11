"""Tests for GLM backward compatibility aliases.

GLM models now work directly with OpenAIChatAgent + GLMComputerTool.
These tests verify the backward-compatible imports still work.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hud.agents.glm_cua import GLM_CUA_INSTRUCTIONS, GLMCUA, GLMCUAAgent
from hud.agents.openai_chat import OpenAIChatAgent

# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Test that old imports still work after removing GLMCUAAgent class."""

    def test_glmcua_is_openai_chat_agent(self) -> None:
        """GLMCUA should be an alias for OpenAIChatAgent."""
        assert GLMCUA is OpenAIChatAgent

    def test_glmcua_agent_is_openai_chat_agent(self) -> None:
        """GLMCUAAgent should be an alias for OpenAIChatAgent."""
        assert GLMCUAAgent is OpenAIChatAgent

    def test_glm_cua_instructions_importable(self) -> None:
        """GLM_CUA_INSTRUCTIONS should be importable from glm_cua module."""
        assert isinstance(GLM_CUA_INSTRUCTIONS, str)
        assert "GUI Agent" in GLM_CUA_INSTRUCTIONS

    def test_glm_cua_instructions_from_tool(self) -> None:
        """GLM_CUA_INSTRUCTIONS should also be importable from tool module."""
        from hud.tools.computer.glm import GLM_CUA_INSTRUCTIONS as tool_instructions

        assert tool_instructions == GLM_CUA_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Usage Pattern
# ---------------------------------------------------------------------------


class TestUsagePattern:
    """Test the recommended usage pattern works."""

    def test_create_with_system_prompt(self) -> None:
        """OpenAIChatAgent.create with GLM system prompt should work."""
        mock_client = AsyncMock()
        agent = OpenAIChatAgent.create(
            openai_client=mock_client,
            model="glm-4.5v",
            system_prompt=GLM_CUA_INSTRUCTIONS,
        )
        assert agent.config.model == "glm-4.5v"
        assert agent.config.system_prompt == GLM_CUA_INSTRUCTIONS

    def test_create_via_alias(self) -> None:
        """GLMCUA.create should work (it's OpenAIChatAgent.create)."""
        mock_client = AsyncMock()
        agent = GLMCUA.create(
            openai_client=mock_client,
            model="glm-4.6v",
            system_prompt=GLM_CUA_INSTRUCTIONS,
        )
        assert agent.config.model == "glm-4.6v"

    @pytest.mark.asyncio
    async def test_system_messages(self) -> None:
        """Agent should include the GLM system prompt in messages."""
        mock_client = AsyncMock()
        agent = OpenAIChatAgent.create(
            openai_client=mock_client,
            model="glm-4.5v",
            system_prompt=GLM_CUA_INSTRUCTIONS,
        )
        messages = await agent.get_system_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "GUI Agent" in messages[0]["content"]
