"""Scenario decorator for Environment - defines setup/evaluate phases."""

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

__all__ = ["ScenarioMixin"]

logger = logging.getLogger(__name__)


class ScenarioMixin:
    """Mixin providing @env.scenario decorator for setup/evaluate phases.

    Scenarios are async generators that yield twice:
    - First yield: prompt string (setup phase)
    - Second yield: reward float (evaluate phase)

    The scenario can receive the agent's answer via yield:
        answer = yield "Do the task"
        yield 1.0 if "success" in answer else 0.0

    The answer is passed via the hud_submit tool or ctx.submit().

    The decorator registers both an MCP prompt and resource with the same
    identifier ({env_name}:{scenario_name}), linked by session state.

    Example:
        @env.scenario()
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

    # Scenario state
    _scenarios: dict[str, Callable[..., AsyncGenerator[Any, Any]]]
    _scenario_sessions: dict[str, AsyncGenerator[Any, Any]]  # session_id -> generator
    _scenario_latest: dict[str, str]  # scenario_name -> latest session_id
    _scenario_answers: dict[str, str]  # scenario_name -> submitted answer

    def _init_scenarios(self) -> None:
        """Initialize scenario state. Called from Environment.__init__."""
        self._scenarios = {}
        self._scenario_sessions = {}
        self._scenario_latest = {}
        self._scenario_answers = {}

        # Register _hud_submit tool (underscore = hidden from agent)
        self._register_hud_submit_tool()

    async def submit(self, scenario: str, answer: str) -> None:
        """Submit the agent's answer for a scenario's evaluate phase.

        This stores the answer locally and broadcasts to connected hubs
        that have the _hud_submit tool (auto-detected by Environment).

        Args:
            scenario: Name of the scenario (without env prefix)
            answer: The agent's answer/result to submit

        Example:
            # Direct call with scenario name
            await env.submit("checkout", "Order completed successfully")

            # Or via EvalContext (knows its own scenario)
            await ctx.submit("Order completed successfully")
        """
        # Store locally for our scenarios
        self._scenario_answers[scenario] = answer
        logger.debug(
            "Stored answer for scenario '%s': %s...",
            scenario,
            answer[:50] if len(answer) > 50 else answer,
        )

        # Broadcast to connections that have _hud_submit
        # Environment._broadcast_tool auto-filters to connections with the tool
        await self._broadcast_tool(  # type: ignore[attr-defined]
            "_hud_submit",
            scenario=scenario,
            answer=answer,
        )

    def _register_hud_submit_tool(self) -> None:
        """Register the _hud_submit tool for receiving agent answers.

        Named with underscore prefix to hide from agent tool listings.
        """
        from fastmcp.tools import Tool

        scenario_self = self

        async def _hud_submit(scenario: str, answer: str) -> str:
            """Submit the agent's answer for a scenario's evaluate phase.

            Internal tool - called by Environment.submit() on connected hubs.

            Args:
                scenario: Name of the scenario (without env prefix)
                answer: The agent's answer/result to submit
            """
            # Store locally (don't broadcast - we ARE the target)
            scenario_self._scenario_answers[scenario] = answer
            logger.debug(
                "_hud_submit received answer for scenario '%s': %s...",
                scenario,
                answer[:50] if len(answer) > 50 else answer,
            )
            return f"Answer submitted for scenario '{scenario}'"

        # Register the tool with underscore name
        tool = Tool.from_function(_hud_submit)
        self._tool_manager.add_tool(tool)
        logger.debug("Registered _hud_submit tool")

    async def run_scenario_setup(self, scenario_name: str, args: dict[str, Any]) -> str | None:
        """Run a scenario's setup phase and return the prompt.

        Handles both local scenarios (registered via @env.scenario) and remote
        scenarios (via MCP prompt).

        Args:
            scenario_name: Name of the scenario to run
            args: Arguments to pass to the scenario

        Returns:
            The prompt string from the scenario's setup phase, or None if failed
        """
        # Check if scenario is registered locally
        if scenario_name in self._scenarios:
            # Local scenario - run setup via generator
            scenario_fn = self._scenarios[scenario_name]
            gen = scenario_fn(**args)

            # Run setup phase (code before first yield)
            prompt = await gen.__anext__()

            # Store generator for evaluate phase
            session_id = uuid.uuid4().hex[:8]
            self._scenario_sessions[session_id] = gen
            self._scenario_latest[scenario_name] = session_id

            logger.debug(
                "Scenario %s setup complete, session=%s",
                scenario_name,
                session_id,
            )
            return str(prompt)
        else:
            # Remote scenario - call via MCP prompt
            # If scenario_name already contains ":", it's already namespaced - use directly
            # Otherwise, prefix with env name: {env_name}:{scenario_name}
            if ":" in scenario_name:
                prompt_id = scenario_name
                logger.debug("Remote scenario (already namespaced): prompt_id=%s", prompt_id)
            else:
                env_name = getattr(self, "_source_env_name", None) or self.name
                safe_env_name = env_name.replace("_", "-")
                prompt_id = f"{safe_env_name}:{scenario_name}"
                logger.debug("Remote scenario (adding namespace): prompt_id=%s", prompt_id)
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
                logger.warning("Failed to get scenario prompt: %s", e)
            return None

    async def run_scenario_evaluate(self, scenario_name: str) -> float | None:
        """Run a scenario's evaluate phase and return the reward.

        Uses the submitted answer (if any) via gen.asend().
        Handles both local and remote scenarios.

        Args:
            scenario_name: Name of the scenario to evaluate

        Returns:
            The reward from the scenario's evaluate phase, or None if failed
        """
        # Check if we have a stored generator (local scenario)
        session_id = self._scenario_latest.get(scenario_name)
        if session_id:
            gen = self._scenario_sessions.pop(session_id, None)
            if gen:
                # Get submitted answer (if any)
                answer = self._scenario_answers.pop(scenario_name, None)

                try:
                    # Use asend to pass the answer to the scenario
                    reward = await gen.asend(answer)
                    logger.debug(
                        "Scenario %s evaluate complete, answer=%s, reward=%s",
                        scenario_name,
                        answer[:50] if answer and len(answer) > 50 else answer,
                        reward,
                    )
                    return float(reward)
                except StopAsyncIteration:
                    # Generator ended without second yield - assume success
                    return 1.0
                finally:
                    # Clean up latest pointer
                    if self._scenario_latest.get(scenario_name) == session_id:
                        del self._scenario_latest[scenario_name]

        # Remote scenario - read via MCP resource
        # If scenario_name already contains ":", it's already namespaced - use directly
        if ":" in scenario_name:
            resource_id = scenario_name
        else:
            env_name = getattr(self, "_source_env_name", None) or self.name
            safe_env_name = env_name.replace("_", "-")
            resource_id = f"{safe_env_name}:{scenario_name}"
        try:
            contents = await self.read_resource(resource_id)  # type: ignore[attr-defined]
            if contents:
                first_content = contents[0]
                if hasattr(first_content, "text") and isinstance(first_content.text, str):  # type: ignore[union-attr]
                    data = json.loads(first_content.text)  # type: ignore[union-attr]
                    if "reward" in data:
                        return float(data["reward"])
        except Exception as e:
            logger.warning("Failed to get scenario reward: %s", e)
        return None

    def scenario(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[
        [Callable[..., AsyncGenerator[Any, None]]],
        Callable[..., AsyncGenerator[Any, None]],
    ]:
        """Decorator to register a scenario with setup and evaluate phases.

        Creates both a prompt and resource with identifier scenario:{name}.
        The scenario function should yield twice:
        - First yield: the prompt string (returned from prompt)
        - Second yield: the reward float (returned from resource)

        Args:
            name: Optional name for the scenario (defaults to function name)
            description: Optional description of what the scenario does

        Example:
            @env.scenario()
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
            scenario_name = name or fn.__name__
            # Sanitize env name for URI scheme (no underscores allowed)
            safe_env_name = self.name.replace("_", "-")
            scenario_id = f"{safe_env_name}:{scenario_name}"
            scenario_desc = description or fn.__doc__ or f"Scenario: {scenario_name}"

            # Capture source code for reproducibility
            try:
                source_code = inspect.getsource(fn)
            except (OSError, TypeError) as e:
                logger.warning(
                    "Could not capture source code for scenario '%s': %s",
                    scenario_name,
                    e,
                )
                source_code = None

            # Store the generator function
            self._scenarios[scenario_name] = fn

            # Get function signature for prompt arguments with type info
            sig = inspect.signature(fn)
            prompt_args: list[dict[str, Any]] = []
            for p in sig.parameters.values():
                is_required = p.default is inspect.Parameter.empty
                arg_info: dict[str, Any] = {"name": p.name, "required": is_required}

                # Include default value if present
                if not is_required:
                    # Only include JSON-serializable defaults
                    default_val = p.default
                    if default_val is None or isinstance(
                        default_val, (str, int, float, bool, list, dict)
                    ):
                        arg_info["default"] = default_val

                # Extract type annotation
                if p.annotation is not inspect.Parameter.empty:
                    try:
                        # Use pydantic to convert annotation to JSON schema
                        from pydantic import TypeAdapter

                        adapter = TypeAdapter(p.annotation)
                        param_schema = adapter.json_schema()
                        # Extract type from schema (could be "string", "integer", etc.)
                        if "type" in param_schema:
                            arg_info["type"] = param_schema["type"]
                        elif "$ref" in param_schema or "anyOf" in param_schema:
                            # Complex type - store the full schema
                            arg_info["inputSchema"] = param_schema
                    except Exception:
                        arg_info["type"] = "string"
                else:
                    arg_info["type"] = "string"

                prompt_args.append(arg_info)

            # Register PROMPT - runs setup, returns prompt messages
            # We need a reference to self and the outer variables
            scenario_self = self
            scenario_fn = fn
            scenario_name_ref = scenario_name

            async def prompt_handler(**handler_args: Any) -> list[str]:
                # Create generator instance
                gen = scenario_fn(**handler_args)

                # Run setup phase (code before first yield)
                prompt_text = await gen.__anext__()

                # Store generator with session ID
                session_id = uuid.uuid4().hex[:8]
                scenario_self._scenario_sessions[session_id] = gen
                scenario_self._scenario_latest[scenario_name_ref] = session_id

                logger.debug(
                    "Scenario %s setup complete, session=%s, prompt=%s",
                    scenario_name_ref,
                    session_id,
                    prompt_text[:50] if isinstance(prompt_text, str) else prompt_text,
                )

                # Return just the string - FastMCP wraps it in PromptMessage
                # Don't return dict or it gets JSON-serialized as text content
                return [str(prompt_text)]

            # Register prompt using FastMCP - create FunctionPrompt directly
            # to bypass the **kwargs validation in from_function()
            from fastmcp.prompts.prompt import FunctionPrompt, PromptArgument

            # Build meta with source code and full arguments info (with types/defaults)
            scenario_meta: dict[str, Any] = {}
            if source_code:
                scenario_meta["code"] = source_code
            if prompt_args:
                scenario_meta["arguments"] = prompt_args

            prompt = FunctionPrompt(
                name=scenario_id,
                description=f"[Setup] {scenario_desc}",
                arguments=[
                    PromptArgument(name=arg["name"], required=arg["required"])
                    for arg in prompt_args
                ],
                fn=prompt_handler,
                meta=scenario_meta if scenario_meta else None,
            )
            self._prompt_manager.add_prompt(prompt)

            # Register RESOURCE - runs evaluate, returns reward
            async def resource_handler() -> str:
                # Get latest session for this scenario
                session_id = scenario_self._scenario_latest.get(scenario_name_ref)
                if not session_id:
                    raise ValueError(
                        f"No active session for scenario '{scenario_name_ref}'. "
                        "Call the prompt first to run setup."
                    )

                gen = scenario_self._scenario_sessions.pop(session_id, None)
                if gen is None:
                    raise ValueError(f"Session '{session_id}' not found or already evaluated.")

                # Get submitted answer (if any)
                answer = scenario_self._scenario_answers.pop(scenario_name_ref, None)

                # Run evaluate phase (code after first yield)
                # Use asend to pass the answer (or None if not submitted)
                try:
                    reward = await gen.asend(answer)
                except StopAsyncIteration:
                    # Generator ended without second yield - assume success
                    reward = 1.0

                logger.debug(
                    "Scenario %s evaluate complete, session=%s, answer=%s, reward=%s",
                    scenario_name_ref,
                    session_id,
                    answer[:50] if answer and len(answer) > 50 else answer,
                    reward,
                )

                # Clean up latest pointer if it matches
                if scenario_self._scenario_latest.get(scenario_name_ref) == session_id:
                    del scenario_self._scenario_latest[scenario_name_ref]

                return json.dumps({"reward": float(reward)})

            # Register as resource with same scenario: URI
            from fastmcp.resources.resource import FunctionResource

            resource = FunctionResource.from_function(
                fn=resource_handler,
                uri=scenario_id,
                name=scenario_name,
                description=f"[Evaluate] {scenario_desc}",
                mime_type="application/json",
                meta=scenario_meta,
            )
            self._resource_manager.add_resource(resource)

            logger.debug(
                "Registered scenario '%s' as prompt and resource: %s",
                scenario_name,
                scenario_id,
            )

            return fn

        return decorator
