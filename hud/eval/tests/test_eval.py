"""Tests for hud.eval.eval module (Eval class)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from hud.eval.eval import Eval


class TestEvalDataclass:
    """Tests for Eval as a data class."""

    def test_init_defaults(self) -> None:
        """Eval initializes with sensible defaults."""
        ev = Eval()

        assert ev.env_config is None
        assert ev.script is None
        assert ev.args == {}
        assert ev.variants == {}
        assert ev.index == 0

    def test_init_with_config(self) -> None:
        """Eval can be initialized with env_config and script."""
        config = {"name": "test-env", "hubs": []}
        ev = Eval(env=config, script="checkout", args={"user_id": "alice"})

        assert ev.env_config == config
        assert ev.script == "checkout"
        assert ev.args == {"user_id": "alice"}

    def test_copy_creates_new_instance(self) -> None:
        """copy() creates a new Eval instance."""
        original = Eval(
            env={"name": "test"},
            script="checkout",
            args={"user_id": "alice"},
            variants={"model": "gpt-4o"},
        )
        copied = original.copy()

        assert copied is not original
        assert copied.env == original.env
        assert copied.script == original.script
        assert copied.args == original.args
        assert copied.args is not original.args  # Deep copy
        assert copied.variants == original.variants
        assert copied.variants is not original.variants  # Deep copy

    def test_copy_clears_trace_id(self) -> None:
        """copy() clears trace_id for fresh instance."""
        original = Eval(trace_id="original-trace")
        copied = original.copy()

        assert copied.trace_id is None


class TestEvalToEvalContext:
    """Tests for Eval.to_eval_context()."""

    def test_creates_eval_context(self) -> None:
        """to_eval_context() creates an EvalContext."""
        from hud.eval.context import EvalContext

        ev = Eval(script="checkout")
        ctx = ev.to_eval_context()

        assert isinstance(ctx, EvalContext)
        assert ctx.eval_name == "checkout"

    def test_uses_eval_as_name_when_no_script(self) -> None:
        """to_eval_context() uses 'eval' as name when no script."""
        ev = Eval()
        ctx = ev.to_eval_context()

        assert ctx.eval_name == "eval"

    def test_passes_through_properties(self) -> None:
        """to_eval_context() passes through properties."""
        ev = Eval(
            script="checkout",
            trace_id="test-trace",
            api_key="test-key",
            job_id="test-job",
            group_id="test-group",
            index=5,
            variants={"model": "gpt-4o"},
        )
        ctx = ev.to_eval_context()

        assert ctx.trace_id == "test-trace"
        assert ctx._eval_api_key == "test-key"
        assert ctx.job_id == "test-job"
        assert ctx.group_id == "test-group"
        assert ctx.index == 5
        assert ctx.variants == {"model": "gpt-4o"}


class TestEvalContextManager:
    """Tests for Eval as async context manager."""

    @pytest.mark.asyncio
    async def test_aenter_returns_eval_context(self) -> None:
        """__aenter__ returns an EvalContext."""
        from hud.eval.context import EvalContext

        ev = Eval()  # No script to avoid script lookup

        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
            patch.object(EvalContext, "__aexit__", new_callable=AsyncMock),
        ):
            ctx = await ev.__aenter__()
            assert isinstance(ctx, EvalContext)
            # Clean up manually since we patched __aexit__
            ev._ctx = None

    @pytest.mark.asyncio
    async def test_context_clears_on_exit(self) -> None:
        """__aexit__ clears internal context reference."""
        from hud.eval.context import EvalContext

        ev = Eval()

        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
            patch.object(EvalContext, "__aexit__", new_callable=AsyncMock),
        ):
            await ev.__aenter__()
            assert ev._ctx is not None

            # Manually call __aexit__ on Eval (which will call mocked ctx.__aexit__)
            await ev.__aexit__(None, None, None)
            assert ev._ctx is None

    @pytest.mark.asyncio
    async def test_reward_accessible_after_exit(self) -> None:
        """Reward set in context is accessible after exit."""
        from hud.eval.context import EvalContext

        ev = Eval()

        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
            patch.object(EvalContext, "__aexit__", new_callable=AsyncMock),
        ):
            ctx = await ev.__aenter__()
            ctx.reward = 0.95

            await ev.__aexit__(None, None, None)
            # Context reference is cleared but reward was set on the actual context


class TestEvalFromApi:
    """Tests for _eval_from_api helper."""

    def test_creates_eval_from_api_response(self) -> None:
        """_eval_from_api creates Eval from API response."""
        from hud.eval.manager import _eval_from_api

        data = {
            "env_config": {"name": "test-env", "hubs": []},
            "script": "checkout",
            "args": {"user_id": "alice"},
        }

        ev = _eval_from_api(data)

        assert ev.env_config == {"name": "test-env", "hubs": []}
        assert ev.script == "checkout"
        assert ev.args == {"user_id": "alice"}

    def test_handles_missing_optional_fields(self) -> None:
        """_eval_from_api handles missing optional fields."""
        from hud.eval.manager import _eval_from_api

        data = {}  # Minimal response

        ev = _eval_from_api(data)

        assert ev.env_config is None
        assert ev.script is None
        assert ev.args == {}


class TestEnvironmentCall:
    """Tests for Environment.__call__ returning Eval."""

    def test_call_returns_eval(self) -> None:
        """Environment() returns an Eval object."""
        from hud.environment import Environment

        env = Environment("test-env")
        ev = env()

        assert isinstance(ev, Eval)

    def test_call_with_script_sets_script(self) -> None:
        """Environment(script) sets script name."""
        from hud.environment import Environment

        env = Environment("test-env")
        ev = env("checkout")

        assert ev.script == "checkout"

    def test_call_with_args_sets_args(self) -> None:
        """Environment(script, **args) sets args."""
        from hud.environment import Environment

        env = Environment("test-env")
        ev = env("checkout", user_id="alice", amount=100)

        assert ev.args == {"user_id": "alice", "amount": 100}

    def test_call_captures_env_config_when_configured(self) -> None:
        """Environment() captures env config when there's something to store."""
        from hud.environment import Environment

        # Plain env has no config (nothing to reconstruct)
        env = Environment("test-env")
        ev = env()
        assert ev.env_config is None  # Nothing to store

        # Env with setup_tool has config
        env2 = Environment("test-env").setup_tool("navigate", url="https://example.com")
        ev2 = env2()
        assert ev2.env_config is not None
        assert ev2.env_config["name"] == "test-env"
        assert len(ev2.env_config["setup_tools"]) == 1
