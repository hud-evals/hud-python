"""The local authoring agent: ``integration_test``.

The 01-coding-template documents ``hud eval <env> integration_test`` as the
shipping check for a task: pre-stage the golden solution (``Task.validation``),
let the environment's scenario graders run, and require Reward 1.0. This is
the *local* implementation of that agent — no LLM, no platform. It replays
every validation tool call through the task's own MCP capabilities (the bash
capability for the coding template), then ends the trace with an empty answer
so the environment grades the staged workspace.

The Reward-1.0 gate lives in the CLI (``hud.cli.eval``): grading happens after
the agent finishes, so the agent itself cannot see the reward.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from hud.agents.base import Agent
from hud.agents.types import IntegrationTestConfig, ToolStep
from hud.capabilities import MCPClient, SSHClient
from hud.types import MCPToolCall, MCPToolResult
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

DEFAULT_VALIDATION_TIMEOUT_SECONDS = 120.0


class IntegrationTestAgent(Agent):
    """Pre-stage ``Task.validation``, then yield to the scenario graders.

    Stateless per run; one instance can drive concurrent rollouts.
    """

    def __init__(self, config: IntegrationTestConfig) -> None:
        self.config = config

    async def __call__(self, run: Run) -> None:
        validation = list(getattr(run, "validation", None) or [])
        if not validation:
            logger.warning("integration_test: task has no Task.validation — nothing staged")
            return

        connections: dict[str, MCPClient | SSHClient] = {}
        manifest = run.client.manifest
        if manifest is not None:
            for cap in manifest.bindings:
                if cap.protocol not in (MCPClient.protocol, SSHClient.protocol):
                    continue
                opened = await run.client.open(cap.name)
                # open() resolves through the capability registry, so the
                # client type matches the protocol we filtered on above.
                connections[cap.name] = cast("MCPClient | SSHClient", opened)

        if not connections:
            logger.warning(
                "integration_test: no MCP capabilities to stage the golden solution through"
            )

        timeout = self.config.timeout_seconds or DEFAULT_VALIDATION_TIMEOUT_SECONDS
        deadline = asyncio.timeout(timeout)
        try:
            async with deadline:
                for entry in validation:
                    call = self._coerce_call(entry)
                    if call is None:
                        continue
                    result = await self._dispatch(connections, call)
                    run.record(ToolStep(call=call, result=result, started_at=now_iso()))
        except TimeoutError:
            run.trace.status = "error"
            run.trace.stop_reason = "timeout"
            logger.warning("integration_test: validation staging timed out after %gs", timeout)

    @staticmethod
    def _coerce_call(entry: Any) -> MCPToolCall | None:
        if isinstance(entry, MCPToolCall):
            return entry
        if isinstance(entry, dict):
            try:
                return MCPToolCall.model_validate(entry)
            except Exception as exc:  # surface as a warning, not a crash
                logger.warning("integration_test: skipping invalid validation step: %s", exc)
                return None
        logger.warning("integration_test: skipping unsupported validation step %r", entry)
        return None

    async def _dispatch(
        self,
        connections: dict[str, MCPClient | SSHClient],
        call: MCPToolCall,
    ) -> MCPToolResult:
        """Run one validation tool call against whichever capability serves it.

        Most tasks declare a single MCP capability (``bash``); try each
        connected client in turn and stop at the first non-"unknown tool"
        result so a task with several capabilities still routes correctly.
        SSH-published workspaces (protocol ``ssh/2``) run the same golden
        ``bash`` steps via ``bash -lc`` over the SSH connection.
        """
        from mcp.types import TextContent

        raw_args = call.arguments or {}
        if not isinstance(raw_args, dict):
            from mcp.types import TextContent

            return MCPToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="the validation step's arguments arrived as a string "
                        "and were not executed",
                    )
                ],
                isError=True,
            )
        args: dict[str, Any] = raw_args
        last: MCPToolResult | None = None
        for name, client in connections.items():
            try:
                if isinstance(client, SSHClient):
                    result = await self._run_over_ssh(client, call)
                else:
                    result = await client.call_tool(call.name, args)
            except Exception as exc:
                logger.warning(
                    "integration_test: capability %r failed for %r: %s",
                    name,
                    call.name,
                    exc,
                )
                last = MCPToolResult(
                    content=[TextContent(type="text", text=f"tool error: {exc}")],
                    isError=True,
                )
                continue
            text = getattr(result, "content", None)
            unknown = any(
                getattr(item, "type", None) == "text"
                and str(getattr(item, "text", "")).startswith("unknown tool:")
                for item in (text or [])
            )
            if not unknown:
                return result
            last = result

        if last is not None:
            return last
        return MCPToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"unknown tool: {call.name!r} — no connected MCP "
                        "or SSH capability serves it"
                    ),
                )
            ],
            isError=True,
        )

    @staticmethod
    async def _run_over_ssh(client: SSHClient, call: MCPToolCall) -> MCPToolResult:
        from mcp.types import TextContent

        raw_args = call.arguments or {}
        command = raw_args.get("command") if isinstance(raw_args, dict) else None
        if not isinstance(command, str):
            return MCPToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"validation step {call.name!r} has no bash command to run",
                    )
                ],
                isError=True,
            )
        # Single remote command string: asyncssh shlex-splits it, preserving
        # the command's own quoting, and ships `command` to `bash -lc` as one
        # argument (the codebase idiom; avoids multi-arg space-join hazards).
        completed = await client.run(f"bash -lc {command}")
        output_parts = [p for p in (completed.stdout, completed.stderr) if p]
        output = "".join(
            p.decode("utf-8", errors="replace") if isinstance(p, bytes) else str(p)
            for p in output_parts
        )
        return MCPToolResult(
            content=[TextContent(type="text", text=output.strip() or "(no output)")],
            isError=completed.returncode != 0,
        )
