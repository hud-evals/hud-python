"""Codex CLI harness over a workspace SSH capability."""

from __future__ import annotations

import json
import logging
import shlex
from typing import TYPE_CHECKING, Any, cast

import mcp.types as mcp_types

from hud.agents.base import Agent
from hud.agents.cli import WINDOWS_SHELLS, powershell, powershell_quote, run_jsonl
from hud.agents.types import AgentStep, CodexCLIConfig, ToolStep
from hud.settings import settings
from hud.telemetry.context import get_current_trace_id
from hud.types import MCPToolCall, MCPToolResult, Step
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.capabilities import SSHClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)


class CodexEvents:
    """Translate ``codex exec --json`` events into canonical HUD steps."""

    def __init__(self, run: Run, *, model: str, started_at: str) -> None:
        self.run = run
        self.model = model
        self.agent_started_at = started_at
        self.item_started_at: dict[str, str] = {}
        self.saw_completion = False
        self.error: str | None = None

    def consume(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError("Codex stream event must be an object")

        received_at = now_iso()
        match event.get("type"):
            case "thread.started":
                self.run.trace.extra["codex_thread_id"] = event["thread_id"]
            case "turn.started":
                self.agent_started_at = received_at
            case "item.started":
                item = event["item"]
                self.item_started_at[item["id"]] = received_at
            case "item.completed":
                self.record(event["item"], received_at)
            case "turn.completed":
                self.run.trace.extra["usage"] = event["usage"]
                self.saw_completion = True
            case "turn.failed":
                self.error = event["error"]["message"]
            case "error":
                self.error = event["message"]

    def finish(self, *, returncode: int, stderr: str) -> None:
        trace = self.run.trace
        error = self.error
        if returncode != 0:
            trace.extra["returncode"] = returncode
            error = error or stderr.strip() or f"codex CLI exited with return code {returncode}"
        elif error is None and not self.saw_completion:
            error = "codex CLI exited without a turn.completed event"

        if error is not None and stderr and self.error is None:
            trace.extra["stderr"] = stderr
        if error is not None:
            raise RuntimeError(error)

    def record(self, item: dict[str, Any], received_at: str) -> None:
        item_id = item["id"]
        started_at = self.item_started_at.pop(item_id, self.agent_started_at)
        match item["type"]:
            case "agent_message":
                text = item["text"]
                self.run.trace.content = text
                self.run.record(
                    AgentStep(
                        content=text,
                        model=self.model,
                        raw=item,
                        started_at=started_at,
                        ended_at=received_at,
                    )
                )
            case "reasoning":
                self.run.record(
                    AgentStep(
                        reasoning=item["text"],
                        model=self.model,
                        raw=item,
                        started_at=started_at,
                        ended_at=received_at,
                    )
                )
            case "command_execution" | "file_change" | "mcp_tool_call" | "web_search":
                self.record_tool(item, started_at, received_at)
            case _:
                self.run.record(
                    Step(
                        source="agent",
                        extra={"codex_item": item},
                        started_at=started_at,
                        ended_at=received_at,
                    )
                )
        self.agent_started_at = received_at

    def record_tool(self, item: dict[str, Any], started_at: str, ended_at: str) -> None:
        call_id = item["id"]
        match item["type"]:
            case "command_execution":
                call = MCPToolCall(
                    id=call_id,
                    name="shell",
                    arguments={"command": item["command"]},
                )
                result = MCPToolResult(
                    call_id=call_id,
                    content=[mcp_types.TextContent(type="text", text=item["aggregated_output"])],
                    isError=item["status"] != "completed" or item["exit_code"] not in (None, 0),
                )
            case "file_change":
                changes = item["changes"]
                call = MCPToolCall(
                    id=call_id,
                    name="apply_patch",
                    arguments={"changes": changes},
                )
                result = MCPToolResult(
                    call_id=call_id,
                    content=[
                        mcp_types.TextContent(
                            type="text",
                            text="\n".join(
                                f"{change['kind']}: {change['path']}" for change in changes
                            ),
                        )
                    ],
                    isError=item["status"] != "completed",
                )
            case "mcp_tool_call":
                call = MCPToolCall(
                    id=call_id,
                    name=item["tool"],
                    provider_name=f"{item['server']}.{item['tool']}",
                    arguments=item["arguments"],
                )
                raw_result = item.get("result") or {}
                error = item.get("error")
                result = MCPToolResult.model_validate(
                    {
                        "call_id": call_id,
                        "content": raw_result.get("content")
                        or ([{"type": "text", "text": error["message"]}] if error else []),
                        "structuredContent": raw_result.get("structured_content"),
                        "_meta": raw_result.get("_meta"),
                        "isError": item["status"] == "failed",
                    }
                )
            case "web_search":
                call = MCPToolCall(
                    id=call_id,
                    name="web_search",
                    arguments={"query": item["query"], "action": item["action"]},
                )
                result = MCPToolResult(
                    call_id=call_id,
                    content=[
                        mcp_types.TextContent(
                            type="text",
                            text=json.dumps(item["action"], separators=(",", ":")),
                        )
                    ],
                    isError=False,
                )
            case _:
                raise ValueError(f"unsupported Codex tool item {item['type']!r}")

        self.run.record(
            ToolStep(
                call=call,
                result=result,
                extra={"codex_item": item},
                started_at=started_at,
                ended_at=ended_at,
            )
        )


def codex_command(config: CodexCLIConfig, shell: str) -> str:
    env: dict[str, str] = {}
    args = [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--sandbox",
        config.sandbox,
        "--model",
        config.model,
    ]

    use_hud_gateway = config.use_hud_gateway
    if use_hud_gateway is None:
        use_hud_gateway = settings.api_key is not None
    if use_hud_gateway:
        if not settings.api_key:
            raise ValueError("HUD_API_KEY is required for HUD gateway routing")
        env["HUD_API_KEY"] = settings.api_key
        overrides = {
            "model_provider": "hud",
            "model_providers.hud.name": "HUD",
            "model_providers.hud.base_url": settings.hud_gateway_url,
            "model_providers.hud.env_key": "HUD_API_KEY",
            "model_providers.hud.wire_api": "responses",
        }
        for key, value in overrides.items():
            args.extend(["-c", f"{key}={json.dumps(value)}"])
        if trace_id := get_current_trace_id():
            args.extend(
                [
                    "-c",
                    f'model_providers.hud.http_headers={{"Trace-Id"={json.dumps(trace_id)}}}',
                ]
            )
    elif settings.openai_api_key:
        env["CODEX_API_KEY"] = settings.openai_api_key

    args.append("-")
    if shell in WINDOWS_SHELLS:
        script = ";".join(
            [
                *(f"$env:{key}={powershell_quote(value)}" for key, value in env.items()),
                f"& codex {' '.join(powershell_quote(arg) for arg in args[1:])}",
                "exit $LASTEXITCODE",
            ]
        )
        return powershell(script)

    command = " ".join(shlex.quote(arg) for arg in args)
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    invocation = f"{env_prefix} {command}" if env_prefix else command
    return f'export PATH="$HOME/.local/bin:$PATH"; {invocation}'


async def run_codex(
    config: CodexCLIConfig,
    run: Run,
    *,
    ssh: SSHClient,
    shell: str,
    prompt: str,
) -> None:
    command = codex_command(config, shell)
    logger.info("SSH exec codex CLI (%d chars)", len(command))
    events = CodexEvents(run, model=config.model, started_at=now_iso())
    returncode, stderr = await run_jsonl(ssh, command, events.consume, input_text=prompt)
    logger.info("exit=%s stderr=%d", returncode, len(stderr))
    events.finish(returncode=returncode, stderr=stderr)


class CodexCLIAgent(Agent):
    """Runs ``codex exec`` over SSH inside the environment workspace."""

    config: CodexCLIConfig

    def __init__(self, config: CodexCLIConfig | None = None) -> None:
        self.config = config or CodexCLIConfig()

    async def __call__(self, run: Run) -> None:
        ssh = cast("SSHClient", await run.client.open("ssh"))
        await run_codex(
            self.config,
            run,
            ssh=ssh,
            shell=ssh.capability.params.get("shell", "bash"),
            prompt=run.prompt_text,
        )


__all__ = ["CodexCLIAgent"]
