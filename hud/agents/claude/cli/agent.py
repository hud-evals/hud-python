"""Claude CLI harness over a workspace SSH capability."""

from __future__ import annotations

import json
import logging
import shlex
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, cast

import asyncssh
import mcp.types as mcp_types
from anthropic.types.beta import BetaMessage

from hud.agents.base import Agent
from hud.agents.claude.agent import ClaudeAgent
from hud.agents.cli import WINDOWS_SHELLS, powershell, powershell_quote, run_jsonl
from hud.agents.types import ClaudeCLIConfig, ToolStep
from hud.settings import settings
from hud.telemetry.context import get_current_trace_id
from hud.types import MCPToolCall, MCPToolResult
from hud.utils.time import now_iso

from . import computer_mcp

if TYPE_CHECKING:
    from hud.capabilities import SSHClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

PROMPT_PATH = ".hud_prompt.txt"
MCP_CONFIG_PATH = ".hud_mcp_config.json"
RUN_SCRIPT_PATH = ".hud_run.bat"


class ClaudeEvents:
    """Translate Claude CLI stream messages into canonical HUD steps."""

    def __init__(self, run: Run, *, started_at: str) -> None:
        self.run = run
        self.agent_started_at = started_at
        self.pending_calls: dict[str, tuple[MCPToolCall, str]] = {}
        self.saw_result = False
        self.error: str | None = None

    def consume(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        message = json.loads(line)
        if not isinstance(message, dict):
            raise ValueError("Claude stream event must be an object")
        received_at = now_iso()
        match message.get("type"):
            case "system" if message.get("subtype") == "init":
                self.agent_started_at = received_at
            case "assistant":
                step = ClaudeAgent.message_to_agent_step(
                    BetaMessage.model_validate(message["message"])
                )
                step.started_at = self.agent_started_at
                step.ended_at = received_at
                if step.content:
                    self.run.trace.content = step.content
                self.run.record(step)
                for call in step.tool_calls:
                    self.pending_calls[call.id] = (call, received_at)
            case "user":
                saw_result = False
                for block in message["message"]["content"]:
                    if block["type"] != "tool_result":
                        continue
                    call_id = block["tool_use_id"]
                    try:
                        call, started_at = self.pending_calls.pop(call_id)
                    except KeyError:
                        raise ValueError(
                            f"Claude returned a result for unknown tool call {call_id!r}"
                        ) from None

                    raw_result = block.get("content")
                    raw_items = raw_result if isinstance(raw_result, list) else [raw_result]
                    content: list[mcp_types.ContentBlock] = []
                    for item in raw_items:
                        if isinstance(item, str):
                            content.append(mcp_types.TextContent(type="text", text=item))
                        elif item["type"] == "text":
                            content.append(mcp_types.TextContent(type="text", text=item["text"]))
                        elif item["type"] == "image":
                            source = item["source"]
                            content.append(
                                mcp_types.ImageContent(
                                    type="image",
                                    data=source["data"],
                                    mimeType=source["media_type"],
                                )
                            )
                        else:
                            raise ValueError(f"unsupported Claude tool result block: {item!r}")

                    self.run.record(
                        ToolStep(
                            call=call,
                            result=MCPToolResult(
                                call_id=call_id,
                                content=content,
                                isError=block.get("is_error") is True,
                            ),
                            started_at=started_at,
                            ended_at=received_at,
                        )
                    )
                    saw_result = True
                if saw_result:
                    self.agent_started_at = received_at
            case "result":
                self.saw_result = True
                trace = self.run.trace
                result = message.get("result")
                if isinstance(result, str):
                    trace.content = result
                if message.get("is_error") is True:
                    self.error = trace.content or "claude CLI reported an error"
                for key in (
                    "subtype",
                    "session_id",
                    "duration_ms",
                    "duration_api_ms",
                    "stop_reason",
                    "num_turns",
                    "total_cost_usd",
                ):
                    if (value := message.get(key)) is not None:
                        trace.extra[key] = value

    def finish(self, *, returncode: int, stderr: str) -> None:
        trace = self.run.trace
        error = self.error
        if returncode != 0:
            trace.extra["returncode"] = returncode
            error = stderr.strip() or f"claude CLI exited with return code {returncode}"
        elif not self.saw_result:
            error = "claude CLI exited without a result event"
        elif self.pending_calls:
            missing = ", ".join(sorted(self.pending_calls))
            error = f"claude CLI exited without results for tool calls: {missing}"

        if error is not None and stderr:
            trace.extra["stderr"] = stderr
        if error is not None:
            raise RuntimeError(error)


def claude_command(
    config: ClaudeCLIConfig,
    shell: str,
    prompt: str,
    mcp_config_path: str | None = None,
) -> str:
    env: dict[str, str] = {}
    use_hud_gateway = config.use_hud_gateway
    if use_hud_gateway is None:
        use_hud_gateway = settings.api_key is not None

    if use_hud_gateway:
        if not settings.api_key:
            raise ValueError("HUD_API_KEY is required for HUD gateway routing")
        env["ANTHROPIC_BASE_URL"] = settings.hud_gateway_url
        env["ANTHROPIC_API_KEY"] = settings.api_key
        if trace_id := get_current_trace_id():
            env["ANTHROPIC_CUSTOM_HEADERS"] = f"Trace-Id: {trace_id}"
    elif settings.anthropic_api_key:
        env["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

    env["ANTHROPIC_MODEL"] = config.model
    env["ANTHROPIC_SMALL_FAST_MODEL"] = config.model

    # A custom base URL must own every model tier; otherwise background calls
    # can escape to Anthropic instead of using the configured gateway.
    if "ANTHROPIC_BASE_URL" in env:
        for name in (
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
        ):
            env[name] = config.model

    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["IS_SANDBOX"] = "1"

    args: list[str] = [
        "claude",
        "--verbose",
        "--output-format=stream-json",
        "--print",
        f"--permission-mode={config.permission_mode}",
    ]
    if config.max_steps > 0:
        args.append(f"--max-turns={config.max_steps}")
    if config.system_prompt:
        args.extend(["--system-prompt", config.system_prompt])
    for tool in config.allowed_tools:
        args.extend(["--allowedTools", tool])
    if mcp_config_path:
        args.extend(["--mcp-config", mcp_config_path])

    if shell in WINDOWS_SHELLS:
        script = ";".join(
            [
                *(f"$env:{key}={powershell_quote(value)}" for key, value in env.items()),
                f"Get-Content -Raw -Encoding UTF8 {powershell_quote(PROMPT_PATH)}"
                f" | & claude {' '.join(powershell_quote(arg) for arg in args[1:])}",
                "exit $LASTEXITCODE",
            ]
        )
        return powershell(script)

    args.extend(["--", prompt])
    command = " ".join(shlex.quote(arg) for arg in args)
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    return f'export PATH="$HOME/.local/bin:$PATH"; {env_prefix} {command}'


async def run_claude(
    config: ClaudeCLIConfig,
    run: Run,
    *,
    ssh: SSHClient,
    shell: str,
    mcp_servers: dict[str, dict[str, Any]],
    prompt: str,
) -> None:
    files: dict[str, str] = {}
    mcp_config_path = MCP_CONFIG_PATH if mcp_servers else None
    if mcp_servers:
        files[MCP_CONFIG_PATH] = json.dumps({"mcpServers": mcp_servers}, indent=2)
    if shell in WINDOWS_SHELLS:
        files[PROMPT_PATH] = prompt

    command = claude_command(config, shell, prompt, mcp_config_path)
    if shell in WINDOWS_SHELLS:
        files[RUN_SCRIPT_PATH] = f"@echo off\r\n{command}\r\n"
        command = f"cmd /c {RUN_SCRIPT_PATH}"

    try:
        for path, content in files.items():
            await ssh.write_text(path, content)
        logger.info("SSH exec claude CLI (%d chars)", len(command))
        events = ClaudeEvents(run, started_at=now_iso())
        returncode, stderr = await run_jsonl(ssh, command, events.consume)
        logger.info("exit=%s stderr=%d", returncode, len(stderr))
        events.finish(returncode=returncode, stderr=stderr)
    finally:
        if files:
            if shell in WINDOWS_SHELLS:
                cleanup = f"cmd /c del /f /q {' '.join(files)} 2>nul"
            else:
                cleanup = "rm -f -- " + " ".join(shlex.quote(path) for path in files)
            try:
                await ssh.run(cleanup, check=False)
            except (OSError, asyncssh.Error):
                logger.warning("Failed to remove Claude CLI runtime files")


class ClaudeCLIAgent(Agent):
    """Runs ``claude`` CLI over SSH inside the environment workspace."""

    config: ClaudeCLIConfig

    def __init__(self, config: ClaudeCLIConfig | None = None) -> None:
        self.config = config or ClaudeCLIConfig()

    async def __call__(self, run: Run) -> None:
        mcp_servers: dict[str, dict[str, Any]] = {}
        ssh = cast("SSHClient", await run.client.open("ssh"))
        manifest = run.client.manifest
        assert manifest is not None
        bindings = manifest.bindings
        shell = ssh.capability.params.get("shell", "bash")

        rfb_bindings = [cap for cap in bindings if cap.protocol.split("/", 1)[0] == "rfb"]
        async with AsyncExitStack() as resources:
            for cap in bindings:
                family = cap.protocol.split("/", 1)[0]
                if family == "mcp":
                    token = cap.params.get("auth_token")
                    transport = "http" if cap.params["transport"] == "streamable-http" else "sse"
                    server_config: dict[str, Any] = {"type": transport, "url": cap.url}
                    if token:
                        server_config["headers"] = {"Authorization": f"Bearer {token}"}
                    if cap.name in mcp_servers:
                        raise RuntimeError(f"duplicate MCP server name {cap.name!r}")
                    mcp_servers[cap.name] = server_config
                elif family == "rfb":
                    server_name = (
                        "computer-use" if len(rfb_bindings) == 1 else f"computer-use-{cap.name}"
                    )
                    if server_name in mcp_servers:
                        raise RuntimeError(f"duplicate MCP server name {server_name!r}")
                    routed = run.client.binding(cap.name)
                    mcp_servers[server_name] = await resources.enter_async_context(
                        computer_mcp.bridge_computer_mcp(
                            ssh,
                            routed,
                            self.config.screenshot_encoding,
                            shell=shell,
                        )
                    )

            await run_claude(
                self.config,
                run,
                ssh=ssh,
                shell=shell,
                mcp_servers=mcp_servers,
                prompt=run.prompt_text,
            )


__all__ = ["ClaudeCLIAgent"]
