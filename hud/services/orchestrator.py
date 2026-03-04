"""A2A orchestrator executor backed by a main Chat with sub-chat tools.

The orchestrator builds a local ``hud.Environment`` whose tools are
``AgentTool`` instances — one per ``ChatDefinition``.  An orchestrator
scenario feeds the user's messages to a *main* model that reasons
about which sub-chat tool(s) to call and synthesises a final answer.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution import AgentExecutor
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    Part,
    Role,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from mcp.types import PromptMessage, TextContent

from hud.environment import Environment
from hud.services.chat import Chat
from hud.services.types import ChatDefinition
from hud.tools.agent import AgentTool
from hud.tools.types import ScenarioResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue

LOGGER = logging.getLogger(__name__)


def _status_event(
    *,
    context_id: str,
    task_id: str,
    state: TaskState,
    final: bool,
    text: str | None = None,
) -> TaskStatusUpdateEvent:
    message = None
    if text is not None:
        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[Part(root=TextPart(text=text))],
        )
    return TaskStatusUpdateEvent(
        context_id=context_id,
        task_id=task_id,
        final=final,
        status=TaskStatus(state=state, message=message),
    )


def _build_orchestrator_env(
    definitions: list[ChatDefinition],
    *,
    subagent_max_steps: int = 50,
) -> Environment:
    """Create a local Environment with one AgentTool per definition."""
    env = Environment("orchestrator")

    tool_descriptions: list[str] = []

    for defn in definitions:
        params_override: dict[str, Any] | None = None
        if defn.tool_properties is not None:
            params_override = {
                "type": "object",
                "properties": defn.tool_properties,
                "required": defn.tool_required or [],
            }

        tool = AgentTool(
            defn.task,
            model=defn.model,
            agent_params=defn.agent_params,
            name=defn.name,
            description=defn.description or f"Delegate to {defn.name}",
            parameters=params_override,
            max_steps=subagent_max_steps,
        )
        env.add_tool(tool)
        tool_descriptions.append(f"- {defn.name}: {tool.description}")

    tools_block = "\n".join(tool_descriptions)

    @env.scenario()
    async def orchestrate(messages: list[PromptMessage]) -> AsyncGenerator[Any, Any]:
        system = PromptMessage(
            role="user",  # type: ignore[arg-type]
            content=TextContent(
                type="text",
                text=(
                    "You are an orchestrator agent. You have access to the "
                    "following specialist tools:\n"
                    f"{tools_block}\n\n"
                    "For each user query, decide which tool(s) to call, "
                    "gather results, and synthesise a clear, complete answer. "
                    "Cite data where available.\n\n"
                    "CRITICAL TOOL-CALL RULES:\n"
                    "1. Never call a tool with empty arguments.\n"
                    "2. If a tool has required inputs and any are missing, "
                    "ask the user a concise follow-up question first.\n"
                    "3. Only call tools after you have all required inputs.\n"
                    "4. If the user's request is ambiguous, clarify before "
                    "making any tool call."
                ),
            ),
        )
        answer = yield [system, *messages]

        answer_str = answer if isinstance(answer, str) else str(answer)
        yield ScenarioResult(
            reward=1.0,
            content=answer_str,
            info={"num_messages": len(messages)},
        )

    return env


async def _discover_scenarios(env_name: str) -> list[dict[str, Any]]:
    """Discover scenarios via the HUD platform MCP (no container / no trace).

    Connects to the *platform* MCP server (``api.hud.ai/v3/mcp``) which
    is a read-only metadata API.  Two JSON-RPC calls are made over
    StreamableHTTP:

    1. ``list_environments`` → resolve *env_name* to a registry UUID
    2. ``list_scenarios(environment_id)`` → fetch scenario metadata
    """
    import json

    import httpx

    from hud.settings import settings

    platform_url = f"{settings.hud_api_url.rstrip('/')}/v3/mcp/"
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    async def _call(
        client: httpx.AsyncClient,
        method: str,
        params: dict[str, Any],
        req_id: str,
    ) -> dict[str, Any]:
        resp = await client.post(
            platform_url,
            headers=headers,
            json={"jsonrpc": "2.0", "id": req_id, "method": method, "params": params},
        )
        resp.raise_for_status()

        body = resp.text
        # Server may return SSE-wrapped JSON-RPC — extract the data line
        for line in body.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:])  # type: ignore[no-any-return]
        return json.loads(body)

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Resolve env name → ID
        envs_rpc = await _call(
            client, "tools/call",
            {"name": "list_environments", "arguments": {"page_size": 100}},
            req_id="1",
        )

        env_id: str | None = None
        for block in envs_rpc.get("result", {}).get("content", []):
            if block.get("type") == "text":
                payload = json.loads(block["text"])
                for entry in payload.get("environments", []):
                    if entry.get("name") == env_name:
                        env_id = entry["id"]
                        break

        if not env_id:
            raise ValueError(
                f"Environment {env_name!r} not found on the platform. "
                "Check the name or ensure you have access."
            )

        # 2. List scenarios for that environment
        scen_rpc = await _call(
            client, "tools/call",
            {"name": "list_scenarios", "arguments": {"environment_id": env_id}},
            req_id="2",
        )

        scenarios: list[dict[str, Any]] = []
        for block in scen_rpc.get("result", {}).get("content", []):
            if block.get("type") == "text":
                payload = json.loads(block["text"])
                scenarios = payload.get("scenarios", [])
                break

    return scenarios


def _scenarios_to_definitions(
    scenarios: list[dict[str, Any]],
    *,
    env_name: str,
    model: str,
) -> list[ChatDefinition]:
    """Convert discovered scenario metadata into ChatDefinitions.

    Handles both platform API format (``list_scenarios``) and
    MCP analyze format (``_derive_scenarios``).
    """
    from hud.eval.task import Task

    definitions: list[ChatDefinition] = []
    for s in scenarios:
        # Platform API: "name" is "env:scenario", e.g. "my-env:my_scenario"
        # Analyze format: "id" is the full key, "name" is the short scenario name
        full_name = s.get("name", "") or s.get("id", "")
        if not full_name:
            continue

        # Derive short name and scenario id
        if ":" in full_name:
            _, short_name = full_name.split(":", 1)
            scenario_id = full_name
        else:
            short_name = full_name
            scenario_id = full_name

        # Extract description from metadata or top-level fields
        meta = s.get("metadata", {}) or {}
        desc = (
            meta.get("setup_description", "")
            or meta.get("evaluate_description", "")
            or s.get("setup_description", "")
            or s.get("evaluate_description", "")
        )
        for prefix in ("[Setup] ", "[Evaluate] "):
            if desc.startswith(prefix):
                desc = desc[len(prefix):]

        # Build JSON-Schema properties and required list from scenario arguments.
        raw_args = s.get("arguments") or []
        tool_properties: dict[str, Any] = {}
        tool_required: list[str] = []
        for arg in raw_args:
            if not isinstance(arg, dict) or not arg.get("name"):
                continue
            arg_name = str(arg["name"])
            arg_input_schema = arg.get("inputSchema")
            if isinstance(arg_input_schema, dict) and arg_input_schema:
                # Prefer source-provided schema (often anyOf for optional complex args).
                prop: dict[str, Any] = dict(arg_input_schema)
            else:
                prop = {"type": arg.get("type") or "string"}
            if arg.get("description"):
                prop["description"] = arg["description"]
            tool_properties[arg_name] = prop
            if arg.get("required"):
                tool_required.append(arg_name)

        req_hint = ""
        if tool_required:
            req_hint = " Required inputs: " + ", ".join(tool_required) + "."

        definitions.append(
            ChatDefinition(
                name=short_name,
                task=Task(env={"name": env_name}, scenario=scenario_id),
                model=model,
                description=(desc or f"Run {short_name}") + req_hint,
                tool_properties=tool_properties if tool_properties else None,
                tool_required=tool_required if tool_required else None,
            )
        )
    return definitions


class OrchestratorExecutor(AgentExecutor):
    """A2A executor backed by a main Chat that delegates to sub-chat tools.

    Each ``ChatDefinition`` becomes an ``AgentTool`` available to the
    orchestrator's main model.  A2A sessions are managed internally
    via per-context ``Chat`` instances.
    """

    def __init__(
        self,
        definitions: list[ChatDefinition],
        *,
        main_model: str = "gpt-4o",
        main_max_steps: int = 12,
        subagent_max_steps: int = 50,
        name: str = "hud-chat-orchestrator",
        description: str = "Orchestrator that delegates to sub-chat agents.",
    ) -> None:
        if not definitions:
            raise ValueError("At least one ChatDefinition is required")

        self._definitions = definitions
        self._main_model = main_model
        self._main_max_steps = main_max_steps
        self._subagent_max_steps = subagent_max_steps
        self._name = name
        self._description = description

        self._env = _build_orchestrator_env(
            definitions,
            subagent_max_steps=self._subagent_max_steps,
        )
        self._main_task = self._env("orchestrate")

        self._sessions: dict[str, Chat] = {}

    @classmethod
    async def from_environment(
        cls,
        env_name: str,
        *,
        model: str = "claude-haiku-4-5",
        main_model: str = "gpt-4o",
        main_max_steps: int = 12,
        subagent_max_steps: int = 50,
        scenarios: list[str] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> OrchestratorExecutor:
        """Build an orchestrator by auto-discovering scenarios from a HUD environment.

        Args:
            env_name: Hub environment name (e.g. ``"my-hud-environment"``).
            model: Model for sub-chat agents.
            main_model: Model for the orchestrator's reasoning.
            scenarios: Whitelist of scenario names to include.  When
                ``None``, all discovered scenarios are used.
            name: Display name for the orchestrator.
            description: Description for the A2A agent card.
        """
        discovered = await _discover_scenarios(env_name)

        if scenarios is not None:
            allowed = set(scenarios)
            discovered = [s for s in discovered if s.get("name") in allowed]

        if not discovered:
            raise ValueError(
                f"No scenarios found for environment {env_name!r}"
                + (f" matching {scenarios}" if scenarios else "")
            )

        definitions = _scenarios_to_definitions(
            discovered, env_name=env_name, model=model,
        )

        LOGGER.info(
            "Discovered %d scenario(s) for %s: %s",
            len(definitions),
            env_name,
            [d.name for d in definitions],
        )

        return cls(
            definitions,
            main_model=main_model,
            main_max_steps=main_max_steps,
            subagent_max_steps=subagent_max_steps,
            name=name or f"hud-{env_name}-orchestrator",
            description=description or (
                f"Auto-discovered orchestrator for {env_name} "
                f"with {len(definitions)} sub-agent(s)."
            ),
        )

    def _get_or_create_chat(self, context_id: str) -> Chat:
        """Return existing Chat for context_id or create a new one."""
        if context_id not in self._sessions:
            self._sessions[context_id] = Chat(
                self._main_task,
                model=self._main_model,
                name=self._name,
                description=self._description,
                max_steps=self._main_max_steps,
            )
        return self._sessions[context_id]

    def agent_card(self, url: str = "http://localhost:9999/") -> AgentCard:
        skills = [
            AgentSkill(
                id=defn.name,
                name=defn.display_name or defn.name,
                description=defn.description or f"Delegate to {defn.name}",
                tags=[defn.name],
            )
            for defn in self._definitions
        ]
        return AgentCard(
            name=self._name,
            description=self._description,
            url=url,
            version="1.0",
            capabilities=AgentCapabilities(streaming=True),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=skills,
        )

    def serve(
        self,
        *,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 9999,
        url: str | None = None,
    ) -> None:
        """Serve the orchestrator via the A2A Starlette app."""
        import uvicorn
        from a2a.server.apps import A2AStarletteApplication
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore

        public_url = url or f"http://{host}:{port}/"
        handler = DefaultRequestHandler(
            agent_executor=self,
            task_store=InMemoryTaskStore(),
        )
        app = A2AStarletteApplication(
            agent_card=self.agent_card(public_url),
            http_handler=handler,
        )
        LOGGER.info("Serving A2A orchestrator at %s", public_url)
        uvicorn.run(app.build(), host=host, port=port)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id or str(uuid.uuid4())
        task_id = context.task_id or str(uuid.uuid4())
        message = context.get_user_input()
        message_id = (
            getattr(getattr(context, "message", None), "message_id", "") or ""
        )

        await event_queue.enqueue_event(
            _status_event(
                context_id=context_id,
                task_id=task_id,
                final=False,
                state=TaskState.working,
            )
        )

        try:
            chat = self._get_or_create_chat(context_id)
            result = await chat.send(message)
            content = result.content or ""

            LOGGER.info(
                "a2a_turn_completed context_id=%s task_id=%s "
                "message_id=%s trace_id=%s",
                context_id,
                task_id,
                message_id,
                chat.session_id,
            )

            await event_queue.enqueue_event(
                _status_event(
                    context_id=context_id,
                    task_id=task_id,
                    final=True,
                    state=TaskState.completed,
                    text=content,
                )
            )
        except Exception as e:
            LOGGER.exception("orchestrator execute failed")
            await event_queue.enqueue_event(
                _status_event(
                    context_id=context_id,
                    task_id=task_id,
                    final=True,
                    state=TaskState.failed,
                    text=str(e),
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id or ""
        task_id = context.task_id or ""

        session = self._sessions.pop(context_id, None)
        if session is not None:
            session.clear()

        await event_queue.enqueue_event(
            _status_event(
                context_id=context_id,
                task_id=task_id,
                final=True,
                state=TaskState.canceled,
            )
        )
