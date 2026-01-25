from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from hud.agents.base import MCPAgent
from hud.types import AgentResponse, BaseAgentConfig, Trace

if TYPE_CHECKING:
    from hud.eval.context import EvalContext


class IntegrationTestRunner(MCPAgent):
    """Special agent that runs integration tests by executing tools directly.

    Unlike regular agents, this doesn't run an LLM loop - it executes
    integration_test_tool and evaluate_tool in sequence to verify tool behavior.
    """

    metadata: ClassVar[dict[str, Any] | None] = {}
    config_cls: ClassVar[type[BaseAgentConfig]] = BaseAgentConfig

    def __init__(self, **kwargs: Any) -> None:
        kwargs["auto_trace"] = False
        super().__init__(**kwargs)

    async def run(
        self,
        ctx: EvalContext,
        *,
        max_steps: int = 10,
    ) -> Trace:
        """Run integration test by executing tools directly.

        The EvalContext should have integration_test_tool and evaluate_tool
        configured in its metadata or environment setup.
        """
        from hud.eval.context import EvalContext

        if not isinstance(ctx, EvalContext):
            raise TypeError(f"ctx must be EvalContext, got {type(ctx).__name__}")

        self.ctx = ctx

        try:
            # Initialize tools from context
            if not self._initialized:
                await self._initialize_from_ctx(ctx)

            self.console.info(f"Full system prompt: {self.system_prompt}")

            # For integration tests, we expect the context's environment to have
            # _setup_calls, _integration_test_calls, and _evaluate_calls configured
            env = ctx

            # Run integration test tool (stored in environment metadata or separate list)
            integration_test_calls = getattr(env, "_integration_test_calls", [])
            if not integration_test_calls:
                raise ValueError(
                    "--integration-test requires integration_test_tool to be configured"
                )

            for name, args in integration_test_calls:
                await ctx.call_tool((name, args))

            # The evaluate phase runs automatically when ctx exits,
            # but we can also get the reward from ctx.reward after
            return Trace(done=True, reward=ctx.reward or 0.0, info={})

        finally:
            await self._cleanup()

    # Stub implementations to satisfy abstract base class; not used in --integration-test path
    async def get_system_messages(self) -> list[Any]:
        return []

    async def get_response(self, messages: list[Any]) -> AgentResponse:
        raise NotImplementedError("IntegrationTestRunner does not implement agent loop")

    async def format_blocks(self, blocks: list[Any]) -> list[Any]:
        return []

    async def format_tool_results(
        self,
        tool_calls: list[Any],
        tool_results: list[Any],
    ) -> list[Any]:
        return []
