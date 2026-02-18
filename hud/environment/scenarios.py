"""Scenario decorator for Environment - defines setup/evaluate phases."""

from __future__ import annotations

import functools
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, get_type_hints

from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict

from hud.tools.types import EvaluationResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from fastmcp.prompts import PromptManager
    from fastmcp.resources import ResourceManager
    from fastmcp.tools import ToolManager

    from hud.eval.task import Task

__all__ = ["ScenarioHandle", "ScenarioMixin", "ScenarioSession"]

P = ParamSpec("P")

logger = logging.getLogger(__name__)


class ScenarioHandle(Generic[P]):
    """Wraps a scenario function, providing a typed ``.task()`` factory.

    Returned by ``@env.scenario``.  Behaves as the original async-generator
    function (``__call__`` delegates), but adds ``.task()`` which creates a
    :class:`~hud.eval.task.Task` whose keyword arguments are type-checked
    against the scenario function's signature via ``ParamSpec``.

    Example::

        @env.scenario(name="fix_bug")
        async def fix_bug(difficulty: int = 1, hint: str | None = None):
            ...

        # IDE autocomplete + Pyright type-checking on scenario kwargs:
        task = fix_bug.task(difficulty=3, hint="look at line 42")
        task.validation = [{"name": "bash", "arguments": {"command": "..."}}]
    """

    def __init__(
        self,
        fn: Any,
        env_name: str,
        scenario_name: str,
    ) -> None:
        self._fn = fn
        self._env_name = env_name
        self._scenario_name = scenario_name
        self._sig = inspect.signature(fn)
        functools.update_wrapper(self, fn)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[Any, None]:
        return self._fn(*args, **kwargs)

    def task(self, *args: P.args, **kwargs: P.kwargs) -> Task:
        """Create a :class:`~hud.eval.task.Task` with typed scenario kwargs.

        Positional and keyword arguments match the scenario function signature.
        The Task's ``env`` defaults to this scenario's environment name;
        override via attribute assignment::

            task = my_scenario.task(difficulty=3)
            task.env = {"name": "custom-image-name"}
            task.validation = [...]
        """
        from hud.eval.task import Task

        bound = self._sig.bind(*args, **kwargs)
        return Task(
            env={"name": self._env_name},
            scenario=f"{self._env_name}:{self._scenario_name}",
            args=dict(bound.arguments),
        )


def _normalize_prompt_yield(value: Any) -> list[str]:
    """Convert various first-yield types to list of strings for FastMCP.

    Accepts:
    - str: Single string prompt
    - TextContent: MCP text content block
    - list[str]: Multiple string prompts
    - list[ContentBlock]: Multiple content blocks (text extracted)

    Returns:
        List of strings that FastMCP will wrap as PromptMessages
    """
    # Simple string - most common case
    if isinstance(value, str):
        return [value]

    # TextContent from mcp.types
    if isinstance(value, TextContent):
        return [value.text]

    # List of items
    if isinstance(value, list):
        results: list[str] = []
        for item in value:
            if isinstance(item, str):
                results.append(item)
            elif isinstance(item, TextContent):
                results.append(item.text)
            elif hasattr(item, "text"):
                # Other content blocks with text attribute
                results.append(str(item.text))
            else:
                # Fall back to string conversion
                results.append(str(item))
        return results

    # ContentBlock with text attribute
    if hasattr(value, "text"):
        return [str(value.text)]

    # Fallback - convert to string
    return [str(value)]


def _normalize_eval_yield(value: Any) -> EvaluationResult:
    """Convert various second-yield types to EvaluationResult.

    Accepts:
    - float/int: Simple reward value (done=True implied)
    - EvaluationResult: Full evaluation result

    Returns:
        EvaluationResult with all fields populated
    """
    # Already an EvaluationResult
    if isinstance(value, EvaluationResult):
        return value

    # Numeric reward - convert to EvaluationResult with done=True
    if isinstance(value, int | float):
        return EvaluationResult.from_float(float(value))

    # Dict-like - try to construct EvaluationResult
    if isinstance(value, dict):
        return EvaluationResult(**value)

    # Fallback - try to convert to float
    try:
        return EvaluationResult.from_float(float(value))
    except (TypeError, ValueError):
        logger.warning("Could not convert yield value %s to EvaluationResult", type(value))
        return EvaluationResult(reward=0.0, done=True, isError=True)


class ScenarioSession(BaseModel):
    """Tracks an active scenario from setup through evaluate.

    Created during run_scenario_setup(), used by submit() and run_scenario_evaluate().
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    local_name: str  # Canonical short name (e.g., "investigate")
    full_name: str  # Full name as called (e.g., "sentry-agent:investigate")
    is_local: bool  # True if running locally (generator exists)
    connection_name: str | None  # Which connection served it (if remote)
    resource_uri: str  # Full URI for reading evaluation result
    generator: Any | None = None  # AsyncGenerator (if local) - Any to avoid validation issues
    answer: str | None = None  # Submitted answer


class ScenarioMixin:
    """Mixin providing @env.scenario decorator for setup/evaluate phases.

    Scenarios are async generators that yield twice:
    - First yield: prompt (setup phase) - str, TextContent, or list
    - Second yield: evaluation (evaluate phase) - float or EvaluationResult

    The scenario can receive the agent's answer via yield:
        answer = yield "Do the task"
        yield 1.0 if "success" in answer else 0.0

    For more detailed evaluation results, yield an EvaluationResult:
        from hud.tools.types import EvaluationResult, SubScore

        answer = yield "Find all items on the page"
        count = await check_items()
        yield EvaluationResult(
            reward=count / 10,
            done=count >= 5,
            content=f"Found {count} items",
            subscores=[
                SubScore(name="detection", weight=0.7, value=count / 10),
                SubScore(name="speed", weight=0.3, value=1.0),
            ],
        )

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

    # These come from Environment/MCPServer (type hints for mixin)
    name: str
    _prompt_manager: PromptManager
    _resource_manager: ResourceManager
    _tool_manager: ToolManager

    # Scenario function registry
    _scenarios: dict[str, Callable[..., AsyncGenerator[Any, Any]]]

    # Single active scenario session - used for BOTH:
    # - Client-side: when we run scenarios (local or remote)
    # - Server-side: when external clients call our scenarios via MCP
    # Only one scenario can be active at a time.
    _active_session: ScenarioSession | None

    def _init_scenarios(self) -> None:
        """Initialize scenario state. Called from Environment.__init__."""
        self._scenarios = {}
        self._active_session = None

        # Register _hud_submit tool (underscore = hidden from agent)
        self._register_hud_submit_tool()

    async def submit(self, scenario: str, answer: str) -> None:
        """Submit the agent's answer for a scenario's evaluate phase.

        Uses _active_session to route to the correct connection (if remote)
        or store locally (if local scenario).

        Args:
            scenario: Name of the scenario (may include env prefix like "env:name")
            answer: The agent's answer/result to submit
        """
        local_name = scenario.split(":")[-1] if ":" in scenario else scenario

        if not self._active_session:
            raise ValueError(
                "No active scenario session. Call run_scenario_setup() before submit()."
            )

        if self._active_session.local_name != local_name:
            raise ValueError(
                f"Scenario mismatch: active session is '{self._active_session.local_name}', "
                f"but submit() called with '{local_name}'"
            )

        self._active_session.answer = answer
        logger.debug("Stored answer in session for scenario '%s'", local_name)

        if not self._active_session.is_local:
            # Remote scenario - send to specific connection
            conn_name = self._active_session.connection_name
            if not conn_name:
                raise ValueError(f"Remote scenario '{local_name}' has no connection")

            conn = self._connections.get(conn_name)  # type: ignore[attr-defined]
            if not conn or not conn.client:
                raise ValueError(f"Connection '{conn_name}' not available")

            await conn.call_tool("_hud_submit", {"scenario": local_name, "answer": answer})
            logger.debug("Sent answer to connection '%s' for scenario '%s'", conn_name, local_name)

    def _register_hud_submit_tool(self) -> None:
        """Register the _hud_submit tool for receiving agent answers.

        Named with underscore prefix to hide from agent tool listings.
        """
        from fastmcp.tools import Tool

        scenario_self = self

        async def _hud_submit(scenario: str, answer: str) -> str:
            """Receive an agent's answer from an external client.

            Called when an external client's Environment.submit() sends an answer
            to us via MCP. Stores in _active_session for resource_handler to use.

            Args:
                scenario: Name of the scenario (may include env prefix like "env:name")
                answer: The agent's answer/result to submit
            """
            local_name = scenario.split(":")[-1] if ":" in scenario else scenario

            if not scenario_self._active_session:
                raise ValueError(f"No active scenario session for '{local_name}'")

            if scenario_self._active_session.local_name != local_name:
                raise ValueError(
                    f"Scenario mismatch: active is '{scenario_self._active_session.local_name}', "
                    f"but received answer for '{local_name}'"
                )

            scenario_self._active_session.answer = answer
            logger.debug(
                "_hud_submit stored answer for scenario '%s': %s...",
                local_name,
                answer[:50] if len(answer) > 50 else answer,
            )
            return f"Answer submitted for scenario '{local_name}'"

        # Register the tool with underscore name
        tool = Tool.from_function(_hud_submit)
        self._tool_manager.add_tool(tool)
        logger.debug("Registered _hud_submit tool")

    async def run_scenario_setup(self, scenario_name: str, args: dict[str, Any]) -> str | None:
        """Run a scenario's setup phase and return the prompt.

        Handles both local scenarios (registered via @env.scenario) and remote
        scenarios (via MCP prompt). Creates _active_session for use by submit/evaluate.

        Args:
            scenario_name: Name of the scenario to run (may include "env:" prefix)
            args: Arguments to pass to the scenario

        Returns:
            The prompt string from the scenario's setup phase, or None if failed
        """
        # Determine if this should be local or remote:
        # - No prefix ("greet") → check local first
        # - Prefix matches our env name ("my-env:greet" when self.name="my-env") → local
        # - Prefix is different ("other-env:greet") → remote only
        local_name: str | None = None
        is_explicitly_remote = False
        if ":" in scenario_name:
            prefix, short_name = scenario_name.rsplit(":", 1)
            # self.name is already normalized (underscores → hyphens) in Environment.__init__
            if prefix == self.name:
                # Prefix matches our env - check local
                local_name = short_name
            else:
                # Different prefix - explicitly remote
                local_name = short_name
                is_explicitly_remote = True
        else:
            # No prefix - check local
            local_name = scenario_name

        # Check if scenario is registered locally (unless explicitly remote)
        if not is_explicitly_remote and local_name in self._scenarios:
            # Local scenario - run setup via generator
            scenario_fn = self._scenarios[local_name]
            gen = scenario_fn(**args)

            # Run setup phase (code before first yield)
            raw_prompt = await gen.__anext__()

            # Normalize the prompt yield to a list of strings
            # Accepts: str, TextContent, list[str], list[ContentBlock]
            prompt_parts = _normalize_prompt_yield(raw_prompt)
            prompt_text = prompt_parts[0] if len(prompt_parts) == 1 else "\n".join(prompt_parts)

            # Create session for local scenario
            self._active_session = ScenarioSession(
                local_name=local_name,
                full_name=scenario_name,
                is_local=True,
                connection_name=None,
                resource_uri=f"{self.name}:{local_name}",
                generator=gen,
            )

            logger.debug(
                "Local scenario setup: %s (session=%s)",
                local_name,
                self._active_session,
            )
            return prompt_text
        else:
            # Remote scenario - call via MCP prompt
            # If scenario_name already contains ":", it's already namespaced - use directly
            # Otherwise, prefix with env name: {env_name}:{scenario_name}
            if ":" in scenario_name:
                prompt_id = scenario_name
            else:
                # Use _source_env_name (from EvalContext) or self.name - both are normalized
                env_name = getattr(self, "_source_env_name", None) or self.name
                prompt_id = f"{env_name}:{scenario_name}"

            # Serialize args for MCP prompt (only supports string values)
            serialized_args: dict[str, str] = {}
            for key, value in args.items():
                serialized_args[key] = value if isinstance(value, str) else json.dumps(value)

            try:
                result = await self.get_prompt(prompt_id, serialized_args)  # type: ignore[attr-defined]
                # Get connection AFTER get_prompt succeeds (routing is now guaranteed built)
                conn_name = self._router.get_prompt_connection(prompt_id)  # type: ignore[attr-defined]
                logger.debug(
                    "Remote scenario: prompt_id=%s, connection=%s",
                    prompt_id,
                    conn_name or "(not found in router)",
                )
            except Exception as e:
                # Fetch available scenarios for error context
                try:
                    prompts = await self.list_prompts()  # type: ignore[attr-defined]
                    scenario_prompts = [p.name for p in prompts if ":" in p.name]
                    available = "\n    ".join(scenario_prompts) if scenario_prompts else "(none)"
                except Exception:
                    available = "(could not fetch)"
                    scenario_prompts = []

                original_error = str(e)
                if prompt_id in scenario_prompts:
                    raise ValueError(
                        f"⚠️ ERROR: Scenario '{prompt_id}' exists but failed to execute.\n\n"
                        f"The scenario was found but encountered an error during setup:\n"
                        f"  {original_error}\n\n"
                        f"This could be caused by:\n"
                        f"  - Missing or invalid scenario arguments\n"
                        f"  - An error in the scenario's setup function\n"
                        f"  - Connection or serialization issues\n\n"
                        f"Check the scenario definition and required arguments."
                    ) from e

                raise ValueError(
                    f"⚠️ ERROR: Scenario not found.\n\n"
                    f"Scenario IDs have the format 'environment_name:scenario_name'.\n"
                    f"If you only specify 'scenario_name', the SDK uses your task's env name "
                    f"as the prefix.\n"
                    f"This won't work if the HUD environment was declared with a different name."
                    f"\n\n"
                    f"  You requested: {scenario_name}\n"
                    f"  SDK looked for: {prompt_id}\n\n"
                    f"Available scenarios:\n    {available}\n\n"
                    f"Fix: Use one of the scenario IDs above in your task JSON."
                ) from e

            # Extract prompt text from response
            prompt_text: str | None = None
            if result.messages:
                first_msg = result.messages[0]
                content = first_msg.content
                if hasattr(content, "text") and isinstance(content.text, str):  # type: ignore[union-attr]
                    prompt_text = content.text  # type: ignore[union-attr]
                elif isinstance(content, str):
                    prompt_text = content

            if not prompt_text:
                raise ValueError(
                    f"Scenario '{scenario_name}' returned an empty response.\n\n"
                    f"The scenario's setup function was called but returned no messages.\n"
                    f"Check that the scenario returns a valid prompt string."
                )

            # Create session for remote scenario - use router's connection info
            self._active_session = ScenarioSession(
                local_name=local_name,
                full_name=scenario_name,
                is_local=False,
                connection_name=conn_name,
                resource_uri=prompt_id,  # Resource has same URI as prompt
                generator=None,
            )

            logger.debug(
                "Remote scenario setup: %s (connection=%s)",
                prompt_id,
                conn_name,
            )
            return prompt_text

    async def run_scenario_evaluate(self, scenario_name: str) -> EvaluationResult | None:
        """Run a scenario's evaluate phase and return the evaluation result.

        Uses _active_session created by run_scenario_setup():
        - Local: use stored generator with submitted answer
        - Remote: read resource from the connection that served setup

        Args:
            scenario_name: Name of the scenario to evaluate

        Returns:
            EvaluationResult with reward, done, content, subscores, etc.
            Returns None if evaluation failed.
        """
        if not self._active_session:
            logger.warning("No active session for scenario '%s'", scenario_name)
            return None

        session = self._active_session
        self._active_session = None  # Clear after use

        if session.is_local:
            # Local scenario - use generator
            if not session.generator:
                logger.warning("Local scenario '%s' has no generator", session.local_name)
                return None

            answer = session.answer
            try:
                raw_result = await session.generator.asend(answer)
                # Normalize to EvaluationResult (handles float, EvaluationResult, dict)
                result = _normalize_eval_yield(raw_result)
                logger.debug(
                    "Local scenario %s evaluate: answer=%s, result=%s",
                    session.local_name,
                    answer[:50] if answer and len(answer) > 50 else answer,
                    result,
                )
                return result
            except StopAsyncIteration:
                # No second yield - default to success
                return EvaluationResult(reward=1.0, done=True)
        else:
            # Remote scenario - read resource via router
            try:
                contents = await self.read_resource(session.resource_uri)  # type: ignore[attr-defined]
                if contents:
                    first = contents[0]
                    if hasattr(first, "text") and isinstance(first.text, str):  # type: ignore[union-attr]
                        data = json.loads(first.text)  # type: ignore[union-attr]
                        # Parse as EvaluationResult (handles both old {"reward": x} and new format)
                        # Default for done is True, so old environments work correctly
                        result = EvaluationResult(**data)
                        logger.debug(
                            "Remote scenario %s evaluate: result=%s",
                            session.local_name,
                            result,
                        )
                        return result
            except Exception as e:
                logger.warning("Failed to get scenario result from %s: %s", session.resource_uri, e)
            return None

    def scenario(
        self,
        name: str | None = None,
        description: str | None = None,
        required_env_vars: list[str] | None = None,
    ) -> Callable[
        [Callable[P, AsyncGenerator[Any, None]]],
        ScenarioHandle[P],
    ]:
        """Decorator to register a scenario with setup and evaluate phases.

        Creates both a prompt and resource with identifier scenario:{name}.
        The scenario function should yield twice:
        - First yield: the prompt string (returned from prompt)
        - Second yield: the reward float (returned from resource)

        Args:
            name: Optional name for the scenario (defaults to function name)
            description: Optional description of what the scenario does
            required_env_vars: Optional list of environment variable names this scenario requires.
                These are used by the HUD platform to check if users have configured the
                necessary API keys/credentials before running this specific scenario.

        Example:
            @env.scenario(required_env_vars=["OPENAI_API_KEY"])
            async def chat(query: str):
                yield f"Answer this question: {query}"
                # ... evaluate
                yield 1.0

            # MCP client usage:
            # 1. get_prompt("{env_name}:chat", {query: "..."}) -> prompt messages
            # 2. agent runs...
            # 3. read_resource("{env_name}:chat") -> {"reward": 0.95}
        """

        def decorator(
            fn: Callable[P, AsyncGenerator[Any, None]],
        ) -> ScenarioHandle[P]:
            scenario_name = name or fn.__name__

            # Validate scenario name - colons are reserved as env:scenario separator
            if ":" in scenario_name:
                raise ValueError(
                    f"Scenario name '{scenario_name}' cannot contain ':' "
                    "(reserved as separator between environment and scenario names)"
                )

            # self.name is already normalized (lowercase, hyphens) by Environment.__init__
            scenario_id = f"{self.name}:{scenario_name}"
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
                        default_val, (str | int | float | bool | list | dict)
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
            scenario_name_ref = scenario_name

            # Resolve parameter type hints for deserialization
            # Use get_type_hints() to handle `from __future__ import annotations`
            # which makes annotations lazy strings (PEP 563)
            # MCP prompts only support string arguments, so we JSON-serialize complex types
            # and use Pydantic TypeAdapter to properly deserialize them
            try:
                param_annotations = get_type_hints(fn)
            except Exception:
                # Fall back to raw annotations if get_type_hints fails
                param_annotations = {
                    p.name: p.annotation
                    for p in sig.parameters.values()
                    if p.annotation is not inspect.Parameter.empty
                }

            async def prompt_handler(**handler_args: Any) -> list[str]:
                from pydantic import TypeAdapter

                # Deserialize JSON-encoded arguments using Pydantic TypeAdapter
                # MCP prompts only support string arguments, so complex types are
                # JSON-serialized on the sending side and deserialized here
                deserialized_args: dict[str, Any] = {}
                for arg_name, arg_value in handler_args.items():
                    annotation = param_annotations.get(arg_name)

                    # Only attempt deserialization on string values
                    if not isinstance(arg_value, str):
                        deserialized_args[arg_name] = arg_value
                        continue

                    # If annotation is explicitly str, keep as string
                    if annotation is str:
                        deserialized_args[arg_name] = arg_value
                        continue

                    # If we have a non-str type annotation, use TypeAdapter
                    if annotation is not None:
                        try:
                            adapter = TypeAdapter(annotation)
                        except Exception:
                            # Unresolvable annotation (e.g. raw string from
                            # PEP 563 fallback) -- treat as untyped
                            adapter = None

                        if adapter is not None:
                            # Try validate_json first (handles Pydantic models,
                            # lists, enums, datetimes from JSON-encoded strings)
                            try:
                                deserialized_args[arg_name] = adapter.validate_json(arg_value)
                                continue
                            except Exception:  # noqa: S110
                                pass

                            # Fall back to validate_python (handles Literal[str]
                            # where validate_json("0") would parse as int 0,
                            # losing the string type)
                            try:
                                deserialized_args[arg_name] = adapter.validate_python(arg_value)
                                continue
                            except Exception:  # noqa: S110
                                pass

                            # TypeAdapter couldn't handle it -- skip generic
                            # heuristics that would lose type information
                            deserialized_args[arg_name] = arg_value
                            continue

                    # No annotation (or unresolvable): try generic JSON decode heuristics
                    stripped = arg_value.strip()
                    if (stripped and stripped[0] in "[{") or stripped in ("true", "false", "null"):
                        try:
                            deserialized_args[arg_name] = json.loads(arg_value)
                            continue
                        except json.JSONDecodeError:
                            pass

                    # Try to decode if it looks like a number
                    if stripped.lstrip("-").replace(".", "", 1).isdigit():
                        try:
                            deserialized_args[arg_name] = json.loads(arg_value)
                            continue
                        except json.JSONDecodeError:
                            pass

                    # Keep as string
                    deserialized_args[arg_name] = arg_value

                # Delegate to run_scenario_setup (consolidates client/server logic)
                prompt_text = await scenario_self.run_scenario_setup(
                    scenario_name_ref, deserialized_args
                )

                if prompt_text is None:
                    raise ValueError(f"Scenario '{scenario_name_ref}' setup returned no prompt")

                # Return just the string - FastMCP wraps it in PromptMessage
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
            if required_env_vars:
                scenario_meta["required_env_vars"] = required_env_vars

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

            # Register RESOURCE - runs evaluate, returns EvaluationResult
            async def resource_handler() -> str:
                # Delegate to run_scenario_evaluate (consolidates client/server logic)
                result = await scenario_self.run_scenario_evaluate(scenario_name_ref)

                if result is None:
                    raise ValueError(f"Scenario '{scenario_name_ref}' evaluation failed")

                # Serialize full EvaluationResult (includes reward, done, content, subscores)
                # Use model_dump to get all fields, excluding None values for cleaner output
                return json.dumps(result.model_dump(exclude_none=True))

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

            return ScenarioHandle(fn=fn, env_name=self.name, scenario_name=scenario_name)

        return decorator
