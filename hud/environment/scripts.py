"""Script decorator for Environment - defines setup/evaluate phases."""

from __future__ import annotations

import inspect
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from fastmcp.prompts import PromptManager
    from fastmcp.resources import ResourceManager

__all__ = ["ScriptMixin"]

logger = logging.getLogger(__name__)


class ScriptMixin:
    """Mixin providing @env.script decorator for setup/evaluate phases.

    Scripts are async generators that yield twice:
    - First yield: prompt string (setup phase)
    - Second yield: reward float (evaluate phase)

    The decorator registers both an MCP prompt and resource with the same
    identifier (script:{name}), linked by session state.

    Example:
        @env.script()
        async def search_cats(url: str):
            await env.call_tool("navigate", url=url)
            yield "Find all cat images on the page"
            result = await env.call_tool("count_cats")
            yield float(result > 0)
    """

    # These come from Environment/MCPServer
    name: str
    _prompt_manager: PromptManager
    _resource_manager: ResourceManager

    # Script state
    _scripts: dict[str, Callable[..., AsyncGenerator[Any, None]]]
    _script_sessions: dict[str, AsyncGenerator[Any, None]]  # session_id -> generator
    _script_latest: dict[str, str]  # script_name -> latest session_id

    def _init_scripts(self) -> None:
        """Initialize script state. Called from Environment.__init__."""
        self._scripts = {}
        self._script_sessions = {}
        self._script_latest = {}

    def script(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[
        [Callable[..., AsyncGenerator[Any, None]]],
        Callable[..., AsyncGenerator[Any, None]],
    ]:
        """Decorator to register a script with setup and evaluate phases.

        Creates both a prompt and resource with identifier script:{name}.
        The script function should yield twice:
        - First yield: the prompt string (returned from prompt)
        - Second yield: the reward float (returned from resource)

        Args:
            name: Optional name for the script (defaults to function name)
            description: Optional description of what the script does

        Example:
            @env.script()
            async def search_cats(url: str):
                await env.call_tool("navigate", url=url)
                yield "Find cat images"
                result = await env.call_tool("count_cats")
                yield float(result > 0)

            # MCP client usage:
            # 1. get_prompt("{env_name}:search_cats", {url: "..."}) -> prompt messages
            # 2. agent runs...
            # 3. read_resource("{env_name}:search_cats") -> {"reward": 0.95}
        """

        def decorator(
            fn: Callable[..., AsyncGenerator[Any, None]],
        ) -> Callable[..., AsyncGenerator[Any, None]]:
            script_name = name or fn.__name__
            script_id = f"{self.name}:{script_name}"
            script_desc = description or fn.__doc__ or f"Script: {script_name}"

            # Store the generator function
            self._scripts[script_name] = fn

            # Get function signature for prompt arguments
            sig = inspect.signature(fn)
            prompt_args = [
                {"name": p.name, "required": p.default is inspect.Parameter.empty}
                for p in sig.parameters.values()
            ]

            # Register PROMPT - runs setup, returns prompt messages
            # We need a reference to self and the outer variables
            script_self = self
            script_fn = fn
            script_name_ref = script_name

            async def prompt_handler(**handler_args: Any) -> list[dict[str, Any]]:
                # Create generator instance
                gen = script_fn(**handler_args)

                # Run setup phase (code before first yield)
                prompt_text = await gen.__anext__()

                # Store generator with session ID
                session_id = uuid.uuid4().hex[:8]
                script_self._script_sessions[session_id] = gen
                script_self._script_latest[script_name_ref] = session_id

                logger.debug(
                    "Script %s setup complete, session=%s, prompt=%s",
                    script_name_ref,
                    session_id,
                    prompt_text[:50] if isinstance(prompt_text, str) else prompt_text,
                )

                return [{"role": "user", "content": str(prompt_text)}]

            # Register prompt using FastMCP - create FunctionPrompt directly
            # to bypass the **kwargs validation in from_function()
            from fastmcp.prompts.prompt import FunctionPrompt, PromptArgument

            prompt = FunctionPrompt(
                name=script_id,
                description=f"[Setup] {script_desc}",
                arguments=[
                    PromptArgument(name=arg["name"], required=arg["required"])
                    for arg in prompt_args
                ],
                fn=prompt_handler,
            )
            self._prompt_manager.add_prompt(prompt)

            # Register RESOURCE - runs evaluate, returns reward
            async def resource_handler() -> str:
                # Get latest session for this script
                session_id = self._script_latest.get(script_name)
                if not session_id:
                    raise ValueError(
                        f"No active session for script '{script_name}'. "
                        "Call the prompt first to run setup."
                    )

                gen = self._script_sessions.pop(session_id, None)
                if gen is None:
                    raise ValueError(
                        f"Session '{session_id}' not found or already evaluated."
                    )

                # Run evaluate phase (code after first yield)
                try:
                    reward = await gen.__anext__()
                except StopAsyncIteration:
                    # Generator ended without second yield - assume success
                    reward = 1.0

                logger.debug(
                    "Script %s evaluate complete, session=%s, reward=%s",
                    script_name,
                    session_id,
                    reward,
                )

                # Clean up latest pointer if it matches
                if self._script_latest.get(script_name) == session_id:
                    del self._script_latest[script_name]

                return json.dumps({"reward": float(reward)})

            # Register as resource with same script: URI
            from fastmcp.resources.resource import FunctionResource

            resource = FunctionResource.from_function(
                fn=resource_handler,
                uri=script_id,
                name=script_name,
                description=f"[Evaluate] {script_desc}",
                mime_type="application/json",
            )
            self._resource_manager.add_resource(resource)

            logger.debug(
                "Registered script '%s' as prompt and resource: %s",
                script_name,
                script_id,
            )

            return fn

        return decorator

