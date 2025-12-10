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

__all__ = ["Eval", "build_eval_name"]

logger = logging.getLogger(__name__)


def build_eval_name(script: str | None, args: dict[str, Any] | None) -> str:
    """Build descriptive name: 'script with val1, val2, ...'"""
    if not script:
        return "eval"
    if not args:
        return script
    
    val_parts = []
    for v in list(args.values())[:3]:  # Max 3 values
        v_str = repr(v) if isinstance(v, str) else str(v)
        if len(v_str) > 25:
            v_str = v_str[:22] + "..."
        val_parts.append(v_str)
    
    if val_parts:
        return f"{script} with {', '.join(val_parts)}"
    return script


@dataclass
class Eval:
    """A runnable evaluation unit (data class).

    Holds the configuration to create an EvalContext:
    - env: The environment (live instance or serialized config)
    - script: Optional script name to run (from @env.script)
    - args: Arguments for the script

    When entered as a context manager, creates an EvalContext.

    Attributes:
        env: Environment instance (local) or EnvConfig dict (remote) or None (blank)
        script: Script name to run (None for env-only)
        args: Script arguments
    """

    # Core config - env can be live Environment or serialized config
    env: Any = None  # Environment | dict[str, Any] | None
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
    _trace: bool = field(default=True, repr=False)
    _quiet: bool = field(default=False, repr=False)

    # Runtime state
    _ctx: EvalContext | None = field(default=None, repr=False)

    # Backwards compat alias
    @property
    def env_config(self) -> dict[str, Any] | None:
        """Get serializable env config (for backwards compat and backend)."""
        from hud.environment import Environment

        if isinstance(self.env, Environment):
            return self.env._get_env_config()
        elif isinstance(self.env, dict):
            return self.env
        return None

    def copy(self) -> Eval:
        """Create a copy of this Eval for parallel execution."""
        return Eval(
            env=self.env,  # Share reference - from_environment handles copying
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
            _trace=self._trace,
            _quiet=self._quiet,
        )

    def to_eval_context(self) -> EvalContext:
        """Convert this Eval to an EvalContext.

        Creates an EvalContext from the environment (live or from config).
        Also handles deprecated Task objects stored in _task attribute.
        """
        from hud.environment import Environment
        from hud.eval.context import EvalContext

        # Check for deprecated Task (backwards compat)
        task = getattr(self, "_task", None)
        if task is not None:
            import warnings
            warnings.warn(
                "Task objects are deprecated. Use Eval from env() instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            ctx = EvalContext.from_task(
                task=task,
                api_key=self.api_key,
                job_id=self.job_id,
                group_id=self.group_id,
                index=self.index,
                variants=self.variants,
                code_snippet=self.code_snippet,
                trace=self._trace,
                quiet=self._quiet,
            )
            ctx._suppress_link = self._suppress_link
            return ctx

        # Get or create environment
        if isinstance(self.env, Environment):
            # Local - use live environment (from_environment handles copying)
            source_env = self.env
        elif isinstance(self.env, dict):
            # Remote/config - create fresh from config
            source_env = Environment.from_config(self.env)
        else:
            # Blank
            source_env = Environment("eval")

        eval_name = build_eval_name(self.script, self.args)

        # Create EvalContext from environment
        ctx = EvalContext.from_environment(
            env=source_env,
            name=eval_name,
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
        ctx._trace_enabled = self._trace

        return ctx

    async def __aenter__(self) -> EvalContext:
        """Enter eval context.
        
        Order of operations:
        1. Create EvalContext from environment config
        2. Connect environment (MCP servers, etc.)
        3. Run script setup (if script) → sets ctx.prompt
        4. Notify backend (with prompt now set)
        5. Print trace link
        """
        self._ctx = self.to_eval_context()
        await self._ctx.__aenter__()  # Connect env, set trace headers

        # Run script setup (sets prompt)
        if self.script:
            await self._run_script_setup()

        # Notify backend with prompt included
        await self._ctx._eval_enter()
        self._ctx._print_eval_link()

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

        # Store script name on context for ctx.submit()
        self._ctx._script_name = self.script

        # Delegate to ScriptMixin.run_script_setup
        prompt = await self._ctx.run_script_setup(self.script, self.args)
        if prompt:
            self._ctx.prompt = prompt

    async def _run_script_evaluate(self) -> None:
        """Run the script's evaluate phase (get reward)."""
        if self._ctx is None or self.script is None:
            return

        # Delegate to ScriptMixin.run_script_evaluate
        reward = await self._ctx.run_script_evaluate(self.script)
        if reward is not None:
            self._ctx.reward = reward
