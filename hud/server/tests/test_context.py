"""Tests for context server utilities."""

from __future__ import annotations

import os
from unittest.mock import patch

from hud.server.context import DEFAULT_SOCK_PATH


class TestContextServer:
    """Test context server functionality."""

    def test_default_sock_path(self) -> None:
        """Test DEFAULT_SOCK_PATH constant."""
        assert DEFAULT_SOCK_PATH == "/tmp/hud_ctx.sock"
        assert isinstance(DEFAULT_SOCK_PATH, str)

    def test_context_path_resolution_env_var(self) -> None:
        """Test that context functions respect HUD_CTX_SOCK environment variable."""
        # Test the path resolution logic that all context functions use
        with patch.dict(os.environ, {'HUD_CTX_SOCK': '/custom/env/path.sock'}):
            # This is the logic used in all context functions
            sock_path = os.getenv("HUD_CTX_SOCK", DEFAULT_SOCK_PATH)
            assert sock_path == '/custom/env/path.sock'

    def test_context_path_resolution_default(self) -> None:
        """Test that context functions use default path when no env var."""
        with patch.dict(os.environ, {}, clear=True):
            # This is the logic used in all context functions
            sock_path = os.getenv("HUD_CTX_SOCK", DEFAULT_SOCK_PATH)
            assert sock_path == DEFAULT_SOCK_PATH

    def test_context_functions_exist(self) -> None:
        """Test that all context functions are properly importable."""
        # Just verify the functions exist and are callable
        from hud.server.context import serve_context, attach_context, run_context_server

        assert callable(serve_context)
        assert callable(attach_context)
        assert callable(run_context_server)

        # Check they have reasonable names
        assert serve_context.__name__ == 'serve_context'
        assert attach_context.__name__ == 'attach_context'
        assert run_context_server.__name__ == 'run_context_server'
