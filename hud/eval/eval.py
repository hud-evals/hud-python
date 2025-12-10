"""Eval - A runnable evaluation unit (data class).

An Eval holds the configuration needed to run an evaluation:
- Environment configuration (how to create/connect)
- Optional script name and args

When entered as a context manager, it creates an EvalContext.

Usage:
    env = Environment("my-env").connect_hub("browser")

    # Empty - just env
    async with env() as ctx:
        await ctx.call_tool("navigate", url="...")

    # With script
    async with env("checkout", user_id="alice") as ctx:
        await agent.run(ctx.prompt)

    # Orchestrated via hud.eval
    evals = [env("checkout", user_id="alice"), env("checkout", user_id="bob")]
    async with hud.eval(evals, variants={"model": ["gpt-4o"]}, group=4) as ctx:
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

    from hud.eval.context import EvalContext

__all__ = ["Eval"]

logger = logging.getLogger(__name__)


@dataclass
class Eval:
    """A runnable evaluation unit (data class).

    Holds the configuration to create an EvalContext:
    - env_config: How to create/connect the environment
    - script: Optional script name to run (from @env.script)
    - args: Arguments for the script

    When entered as a context manager, creates an EvalContext.

    Attributes:
        env_config: Serializable environment configuration
        script: Script name to run (None for env-only)
        args: Script arguments
    """

    # Core config
    env_config: dict[str, Any] | None = None
    script: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    # EvalContext creation params (set by hud.eval for parallel execution)
    trace_id: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    job_id: str | None = field(default=None, repr=False)
    group_id: str | None = field(default=None, repr=False)
    index: int = field(default=0, repr=False)
    variants: dict[str, Any] = field(default_factory=dict, repr=False)
    code_snippet: str | None = field(default=None, repr=False)
    _suppress_link: bool = field(default=False, repr=False)

    # Runtime state
    _ctx: EvalContext | None = field(default=None, repr=False)

    def copy(self) -> Eval:
        """Create a copy of this Eval for parallel execution."""
        return Eval(
            env_config=self.env_config,
            script=self.script,
            args=self.args.copy(),
            trace_id=None,  # Each copy gets unique trace_id
            api_key=self.api_key,
            job_id=self.job_id,
            group_id=self.group_id,
            index=self.index,
            variants=self.variants.copy(),
            code_snippet=self.code_snippet,
            _suppress_link=self._suppress_link,
        )

    def to_eval_context(self) -> EvalContext:
        """Convert this Eval to an EvalContext.

        Creates an EvalContext with environment from env_config and
        script info stored for setup/evaluate phases.
        """
        from hud.environment import Environment
        from hud.eval.context import EvalContext

        # Create environment from config
        env = Environment.from_config(self.env_config) if self.env_config else Environment("eval")

        # Create EvalContext from environment
        ctx = EvalContext.from_environment(
            env=env,
            name=self.script or "eval",
            trace_id=self.trace_id,
            api_key=self.api_key,
            job_id=self.job_id,
            group_id=self.group_id,
            index=self.index,
            variants=self.variants,
            code_snippet=self.code_snippet,
            env_config=self.env_config,
        )
        ctx._suppress_link = self._suppress_link

        return ctx

    async def __aenter__(self) -> EvalContext:
        """Enter eval context - create EvalContext and enter it."""
        self._ctx = self.to_eval_context()
        await self._ctx.__aenter__()

        # If we have a script, run its setup phase
        if self.script:
            await self._run_script_setup()

        return self._ctx

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit eval context - run script evaluate and exit EvalContext."""
        if self._ctx is None:
            return

        # If we have a script and no error, run its evaluate phase
        if self.script and exc_type is None:
            await self._run_script_evaluate()

        # Exit the EvalContext
        await self._ctx.__aexit__(exc_type, exc_val, exc_tb)
        self._ctx = None

    async def _run_script_setup(self) -> None:
        """Run the script's setup phase (get prompt)."""
        if self._ctx is None or self.script is None:
            return

        # Check if script is registered locally
        scripts = getattr(self._ctx, "_scripts", {})
        if self.script in scripts:
            # Local script - run setup via generator
            import uuid

            script_fn = scripts[self.script]
            gen = script_fn(**self.args)

            # Run setup phase (code before first yield)
            prompt = await gen.__anext__()

            # Store generator for evaluate phase
            session_id = uuid.uuid4().hex[:8]
            script_sessions = getattr(self._ctx, "_script_sessions", {})
            script_latest = getattr(self._ctx, "_script_latest", {})
            script_sessions[session_id] = gen
            script_latest[self.script] = session_id

            # Set prompt on context
            self._ctx.prompt = str(prompt)

            logger.debug(
                "Script %s setup complete, session=%s",
                self.script,
                session_id,
            )
        else:
            # Remote script - call via MCP prompt
            # Format: {env_name}:{script_name}
            env_name = self._ctx.name if self._ctx else "eval"
            prompt_id = f"{env_name}:{self.script}"
            try:
                result = await self._ctx.get_prompt(prompt_id, self.args)
                if result.messages:
                    # Extract prompt from first message
                    first_msg = result.messages[0]
                    content = first_msg.content
                    # Handle TextContent which has .text attribute
                    if hasattr(content, "text") and isinstance(content.text, str):  # type: ignore[union-attr]
                        self._ctx.prompt = content.text  # type: ignore[union-attr]
                    elif isinstance(content, str):
                        self._ctx.prompt = content
            except Exception as e:
                logger.warning("Failed to get script prompt: %s", e)

    async def _run_script_evaluate(self) -> None:
        """Run the script's evaluate phase (get reward)."""
        if self._ctx is None or self.script is None:
            return

        # Check if we have a stored generator (local script)
        script_latest = getattr(self._ctx, "_script_latest", {})
        session_id = script_latest.get(self.script)
        if session_id:
            script_sessions = getattr(self._ctx, "_script_sessions", {})
            gen = script_sessions.pop(session_id, None)
            if gen:
                try:
                    reward = await gen.__anext__()
                    self._ctx.reward = float(reward)
                    logger.debug(
                        "Script %s evaluate complete, reward=%s",
                        self.script,
                        reward,
                    )
                except StopAsyncIteration:
                    # Generator ended without second yield - assume success
                    self._ctx.reward = 1.0

                # Clean up latest pointer
                if script_latest.get(self.script) == session_id:
                    del script_latest[self.script]
                return

        # Remote script - read via MCP resource
        # Format: {env_name}:{script_name}
        env_name = self._ctx.name if self._ctx else "eval"
        resource_id = f"{env_name}:{self.script}"
        try:
            import json

            contents = await self._ctx.read_resource(resource_id)
            if contents:
                first_content = contents[0]
                # Handle TextResourceContents which has .text attribute
                if hasattr(first_content, "text") and isinstance(first_content.text, str):  # type: ignore[union-attr]
                    data = json.loads(first_content.text)  # type: ignore[union-attr]
                    if "reward" in data:
                        self._ctx.reward = float(data["reward"])
        except Exception as e:
            logger.warning("Failed to get script reward: %s", e)
