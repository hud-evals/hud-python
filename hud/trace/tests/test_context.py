"""Tests for hud.trace.context module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.trace.context import (
    TraceContext,
    _httpx_request_hook,
    _is_hud_url,
    get_current_trace_headers,
)


class TestIsHudUrl:
    """Tests for _is_hud_url helper."""

    def test_inference_hud_ai_is_hud(self) -> None:
        """inference.hud.ai is a HUD URL."""
        assert _is_hud_url("https://inference.hud.ai/v1/chat") is True
        assert _is_hud_url("http://inference.hud.ai/v1/chat") is True

    def test_mcp_hud_ai_is_hud(self) -> None:
        """mcp.hud.ai is a HUD URL."""
        assert _is_hud_url("https://mcp.hud.ai/browser") is True
        assert _is_hud_url("http://mcp.hud.ai/some/path") is True

    def test_mcp_hud_so_is_hud(self) -> None:
        """mcp.hud.so is a HUD URL."""
        assert _is_hud_url("https://mcp.hud.so/browser") is True

    def test_other_urls_are_not_hud(self) -> None:
        """Other URLs are not HUD URLs."""
        assert _is_hud_url("https://example.com") is False
        assert _is_hud_url("https://api.openai.com") is False
        assert _is_hud_url("https://notinference.hud.ai.fake.com") is False


class TestHttpxRequestHook:
    """Tests for _httpx_request_hook."""

    def test_injects_trace_headers_for_hud_urls(self) -> None:
        """Hook injects trace headers for HUD URLs when in trace context."""
        mock_request = MagicMock()
        mock_request.url = "https://inference.hud.ai/v1/chat"
        mock_request.headers = {}
        
        # Set up trace context
        from hud.trace.context import _current_trace_headers
        token = _current_trace_headers.set({"Trace-Id": "test-trace-123"})
        
        try:
            _httpx_request_hook(mock_request)
            
            assert mock_request.headers["Trace-Id"] == "test-trace-123"
        finally:
            _current_trace_headers.reset(token)

    def test_injects_api_key_for_hud_urls(self) -> None:
        """Hook injects API key for HUD URLs when no auth present."""
        mock_request = MagicMock()
        mock_request.url = "https://mcp.hud.ai/browser"
        mock_request.headers = {}
        
        with patch("hud.trace.context.settings") as mock_settings:
            mock_settings.api_key = "test-api-key"
            
            _httpx_request_hook(mock_request)
            
            assert mock_request.headers["Authorization"] == "Bearer test-api-key"

    def test_does_not_override_existing_auth(self) -> None:
        """Hook does not override existing Authorization header."""
        mock_request = MagicMock()
        mock_request.url = "https://mcp.hud.ai/browser"
        mock_request.headers = {"Authorization": "Bearer existing-token"}
        
        with patch("hud.trace.context.settings") as mock_settings:
            mock_settings.api_key = "test-api-key"
            
            _httpx_request_hook(mock_request)
            
            assert mock_request.headers["Authorization"] == "Bearer existing-token"

    def test_ignores_non_hud_urls(self) -> None:
        """Hook ignores non-HUD URLs."""
        mock_request = MagicMock()
        mock_request.url = "https://api.openai.com/v1/chat"
        mock_request.headers = {}
        
        # Set up trace context
        from hud.trace.context import _current_trace_headers
        token = _current_trace_headers.set({"Trace-Id": "test-trace-123"})
        
        try:
            with patch("hud.trace.context.settings") as mock_settings:
                mock_settings.api_key = "test-api-key"
                
                _httpx_request_hook(mock_request)
                
                # No headers should be added
                assert "Trace-Id" not in mock_request.headers
                assert "Authorization" not in mock_request.headers
        finally:
            _current_trace_headers.reset(token)


class TestTraceContext:
    """Tests for TraceContext."""

    def test_init_generates_trace_id(self) -> None:
        """TraceContext generates trace_id if not provided."""
        mock_env = MagicMock()
        tc = TraceContext(env=mock_env, name="test-task")
        
        assert tc.trace_id is not None
        assert len(tc.trace_id) == 36  # UUID format

    def test_init_uses_provided_trace_id(self) -> None:
        """TraceContext uses provided trace_id."""
        mock_env = MagicMock()
        tc = TraceContext(env=mock_env, name="test-task", trace_id="custom-id")
        
        assert tc.trace_id == "custom-id"

    def test_headers_contains_trace_id(self) -> None:
        """headers property returns dict with trace ID."""
        mock_env = MagicMock()
        tc = TraceContext(env=mock_env, name="test-task", trace_id="test-123")
        
        assert tc.headers == {"Trace-Id": "test-123"}

    def test_success_true_when_no_error(self) -> None:
        """success property returns True when no error."""
        mock_env = MagicMock()
        tc = TraceContext(env=mock_env, name="test-task")
        
        assert tc.success is True

    def test_success_false_when_error(self) -> None:
        """success property returns False when error is set."""
        mock_env = MagicMock()
        tc = TraceContext(env=mock_env, name="test-task")
        tc.error = ValueError("test error")
        
        assert tc.success is False

    def test_done_false_initially(self) -> None:
        """done property returns False initially."""
        mock_env = MagicMock()
        tc = TraceContext(env=mock_env, name="test-task")
        
        assert tc.done is False

    def test_variants_empty_by_default(self) -> None:
        """variants is empty dict by default."""
        mock_env = MagicMock()
        tc = TraceContext(env=mock_env, name="test-task")
        
        assert tc.variants == {}

    def test_variants_set_from_init(self) -> None:
        """variants set from _variants parameter."""
        mock_env = MagicMock()
        tc = TraceContext(
            env=mock_env,
            name="test-task",
            _variants={"model": "gpt-4o", "temp": 0.7},
        )
        
        assert tc.variants == {"model": "gpt-4o", "temp": 0.7}

    @pytest.mark.asyncio
    async def test_context_manager_sets_headers(self) -> None:
        """Context manager sets trace headers in contextvar."""
        mock_env = MagicMock()
        tc = TraceContext(env=mock_env, name="test-task", trace_id="test-123")
        
        # Mock telemetry calls
        with patch.object(tc, "_trace_enter", new_callable=AsyncMock):
            with patch.object(tc, "_trace_exit", new_callable=AsyncMock):
                assert get_current_trace_headers() is None
                
                async with tc:
                    headers = get_current_trace_headers()
                    assert headers is not None
                    assert headers["Trace-Id"] == "test-123"
                
                assert get_current_trace_headers() is None

    @pytest.mark.asyncio
    async def test_context_manager_captures_error(self) -> None:
        """Context manager captures exception in error field."""
        mock_env = MagicMock()
        tc = TraceContext(env=mock_env, name="test-task")
        
        with patch.object(tc, "_trace_enter", new_callable=AsyncMock):
            with patch.object(tc, "_trace_exit", new_callable=AsyncMock):
                with pytest.raises(ValueError):
                    async with tc:
                        raise ValueError("test error")
                
                assert tc.error is not None
                assert str(tc.error) == "test error"
                assert tc.success is False

    @pytest.mark.asyncio
    async def test_call_tool_delegates_to_env(self) -> None:
        """call_tool delegates to environment."""
        mock_env = MagicMock()
        mock_env.call_tool = AsyncMock(return_value="result")
        
        tc = TraceContext(env=mock_env, name="test-task")
        result = await tc.call_tool("my_tool", {"arg": "value"})
        
        mock_env.call_tool.assert_called_once_with("my_tool", {"arg": "value"})
        assert result == "result"

    def test_repr(self) -> None:
        """__repr__ shows useful info."""
        mock_env = MagicMock()
        tc = TraceContext(env=mock_env, name="test-task", trace_id="abc12345-6789-0000-0000-000000000000")
        tc.reward = 0.95
        
        repr_str = repr(tc)
        assert "abc12345" in repr_str
        assert "test-task" in repr_str
        assert "0.95" in repr_str


class TestTraceContextPrompt:
    """Tests for TraceContext.prompt feature."""

    def test_prompt_defaults_from_env(self) -> None:
        """TraceContext.prompt defaults from env.prompt."""
        mock_env = MagicMock()
        mock_env.prompt = "Task prompt from environment"
        
        tc = TraceContext(
            env=mock_env,
            name="test-task",
            trace_id="test-123",
        )
        
        assert tc.prompt == "Task prompt from environment"

    def test_prompt_none_when_env_has_no_prompt(self) -> None:
        """TraceContext.prompt is None when env has no prompt."""
        mock_env = MagicMock(spec=[])  # No prompt attribute
        
        tc = TraceContext(
            env=mock_env,
            name="test-task",
            trace_id="test-123",
        )
        
        assert tc.prompt is None

    def test_prompt_can_be_overridden(self) -> None:
        """TraceContext.prompt can be set to override env default."""
        mock_env = MagicMock()
        mock_env.prompt = "Original prompt"
        
        tc = TraceContext(
            env=mock_env,
            name="test-task",
            trace_id="test-123",
        )
        
        tc.prompt = "Overridden prompt"
        assert tc.prompt == "Overridden prompt"

    def test_prompt_included_in_payload(self) -> None:
        """Prompt is included in trace payload."""
        mock_env = MagicMock()
        mock_env.prompt = "Test prompt"
        mock_env._all_hubs = False
        
        tc = TraceContext(
            env=mock_env,
            name="test-task",
            trace_id="test-123",
        )
        
        payload = tc._build_base_payload()
        assert payload.prompt == "Test prompt"
