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
    from fastmcp.tools import ToolManager

__all__ = ["ScriptMixin"]

logger = logging.getLogger(__name__)


class ScriptMixin:
    """Mixin providing @env.script decorator for setup/evaluate phases.

    Scripts are async generators that yield twice:
    - First yield: prompt string (setup phase)
    - Second yield: reward float (evaluate phase)

    The script can receive the agent's answer via yield:
        answer = yield "Do the task"
        yield 1.0 if "success" in answer else 0.0

    The answer is passed via the hud_submit tool or ctx.submit().

    The decorator registers both an MCP prompt and resource with the same
    identifier ({env_name}:{script_name}), linked by session state.

    Example:
        @env.script()
        async def search_cats(url: str):
            await env.call_tool("navigate", url=url)
            answer = yield "Find all cat images on the page"
            result = await env.call_tool("count_cats")
            yield float(result > 0 or "found" in answer.lower())
    """

    # These come from Environment/MCPServer
    name: str
    _prompt_manager: PromptManager
    _resource_manager: ResourceManager
    _tool_manager: ToolManager

    # Script state
    _scripts: dict[str, Callable[..., AsyncGenerator[Any, Any]]]
    _script_sessions: dict[str, AsyncGenerator[Any, Any]]  # session_id -> generator
    _script_latest: dict[str, str]  # script_name -> latest session_id
    _script_answers: dict[str, str]  # script_name -> submitted answer

    def _init_scripts(self) -> None:
        """Initialize script state. Called from Environment.__init__."""
        self._scripts = {}
        self._script_sessions = {}
        self._script_latest = {}
        self._script_answers = {}
        
        # Register _hud_submit tool (underscore = hidden from agent)
        self._register_hud_submit_tool()

    async def submit(self, script: str, answer: str) -> None:
        """Submit the agent's answer for a script's evaluate phase.

        This stores the answer locally and broadcasts to connected hubs
        that have the _hud_submit tool (auto-detected by Environment).

        Args:
            script: Name of the script (without env prefix)
            answer: The agent's answer/result to submit

        Example:
            # Direct call with script name
            await env.submit("checkout", "Order completed successfully")
            
            # Or via EvalContext (knows its own script)
            await ctx.submit("Order completed successfully")
        """
        # Store locally for our scripts
        self._script_answers[script] = answer
        logger.debug("Stored answer for script '%s': %s...",
                    script, answer[:50] if len(answer) > 50 else answer)

        # Broadcast to connections that have _hud_submit
        # Environment._broadcast_tool auto-filters to connections with the tool
        await self._broadcast_tool(  # type: ignore[attr-defined]
            "_hud_submit",
            script=script,
            answer=answer,
        )

    def _register_hud_submit_tool(self) -> None:
        """Register the _hud_submit tool for receiving agent answers.
        
        Named with underscore prefix to hide from agent tool listings.
        """
        from fastmcp.tools import Tool

        script_self = self

        async def _hud_submit(script: str, answer: str) -> str:
            """Submit the agent's answer for a script's evaluate phase.

            Internal tool - called by Environment.submit() on connected hubs.

            Args:
                script: Name of the script (without env prefix)
                answer: The agent's answer/result to submit
            """
            # Store locally (don't broadcast - we ARE the target)
            script_self._script_answers[script] = answer
            logger.debug("_hud_submit received answer for script '%s': %s...",
                        script, answer[:50] if len(answer) > 50 else answer)
            return f"Answer submitted for script '{script}'"

        # Register the tool with underscore name
        tool = Tool.from_function(_hud_submit)
        self._tool_manager.add_tool(tool)
        logger.debug("Registered _hud_submit tool")

    async def run_script_setup(self, script_name: str, args: dict[str, Any]) -> str | None:
        """Run a script's setup phase and return the prompt.
        
        Handles both local scripts (registered via @env.script) and remote
        scripts (via MCP prompt).
        
        Args:
            script_name: Name of the script to run
            args: Arguments to pass to the script
            
        Returns:
            The prompt string from the script's setup phase, or None if failed
        """
        # Check if script is registered locally
        if script_name in self._scripts:
            # Local script - run setup via generator
            script_fn = self._scripts[script_name]
            gen = script_fn(**args)

            # Run setup phase (code before first yield)
            prompt = await gen.__anext__()

            # Store generator for evaluate phase
            session_id = uuid.uuid4().hex[:8]
            self._script_sessions[session_id] = gen
            self._script_latest[script_name] = session_id

            logger.debug(
                "Script %s setup complete, session=%s",
                script_name,
                session_id,
            )
            return str(prompt)
        else:
            # Remote script - call via MCP prompt
            # Format: {env_name}:{script_name} (use source env name if available)
            env_name = getattr(self, "_source_env_name", None) or self.name
            safe_env_name = env_name.replace("_", "-")
            prompt_id = f"{safe_env_name}:{script_name}"
            try:
                result = await self.get_prompt(prompt_id, args)  # type: ignore[attr-defined]
                if result.messages:
                    first_msg = result.messages[0]
                    content = first_msg.content
                    if hasattr(content, "text") and isinstance(content.text, str):  # type: ignore[union-attr]
                        return content.text  # type: ignore[union-attr]
                    elif isinstance(content, str):
                        return content
            except Exception as e:
                logger.warning("Failed to get script prompt: %s", e)
            return None

    async def run_script_evaluate(self, script_name: str) -> float | None:
        """Run a script's evaluate phase and return the reward.
        
        Uses the submitted answer (if any) via gen.asend().
        Handles both local and remote scripts.
        
        Args:
            script_name: Name of the script to evaluate
            
        Returns:
            The reward from the script's evaluate phase, or None if failed
        """
        # Check if we have a stored generator (local script)
        session_id = self._script_latest.get(script_name)
        if session_id:
            gen = self._script_sessions.pop(session_id, None)
            if gen:
                # Get submitted answer (if any)
                answer = self._script_answers.pop(script_name, None)

                try:
                    # Use asend to pass the answer to the script
                    reward = await gen.asend(answer)
                    logger.debug(
                        "Script %s evaluate complete, answer=%s, reward=%s",
                        script_name,
                        answer[:50] if answer and len(answer) > 50 else answer,
                        reward,
                    )
                    return float(reward)
                except StopAsyncIteration:
                    # Generator ended without second yield - assume success
                    return 1.0
                finally:
                    # Clean up latest pointer
                    if self._script_latest.get(script_name) == session_id:
                        del self._script_latest[script_name]

        # Remote script - read via MCP resource (use source env name if available)
        env_name = getattr(self, "_source_env_name", None) or self.name
        safe_env_name = env_name.replace("_", "-")
        resource_id = f"{safe_env_name}:{script_name}"
        try:
            contents = await self.read_resource(resource_id)  # type: ignore[attr-defined]
            if contents:
                first_content = contents[0]
                if hasattr(first_content, "text") and isinstance(first_content.text, str):  # type: ignore[union-attr]
                    data = json.loads(first_content.text)  # type: ignore[union-attr]
                    if "reward" in data:
                        return float(data["reward"])
        except Exception as e:
            logger.warning("Failed to get script reward: %s", e)
        return None

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
            # Sanitize env name for URI scheme (no underscores allowed)
            safe_env_name = self.name.replace("_", "-")
            script_id = f"{safe_env_name}:{script_name}"
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
                session_id = script_self._script_latest.get(script_name_ref)
                if not session_id:
                    raise ValueError(
                        f"No active session for script '{script_name_ref}'. "
                        "Call the prompt first to run setup."
                    )

                gen = script_self._script_sessions.pop(session_id, None)
                if gen is None:
                    raise ValueError(
                        f"Session '{session_id}' not found or already evaluated."
                    )

                # Get submitted answer (if any)
                answer = script_self._script_answers.pop(script_name_ref, None)

                # Run evaluate phase (code after first yield)
                # Use asend to pass the answer (or None if not submitted)
                try:
                    reward = await gen.asend(answer)
                except StopAsyncIteration:
                    # Generator ended without second yield - assume success
                    reward = 1.0

                logger.debug(
                    "Script %s evaluate complete, session=%s, answer=%s, reward=%s",
                    script_name_ref,
                    session_id,
                    answer[:50] if answer and len(answer) > 50 else answer,
                    reward,
                )

                # Clean up latest pointer if it matches
                if script_self._script_latest.get(script_name_ref) == session_id:
                    del script_self._script_latest[script_name_ref]

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

