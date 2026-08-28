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
from hud.agents.cli import (
    WINDOWS_SHELLS,
    powershell,
    powershell_quote,
    resolve_executable,
    run_jsonl,
)
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

INPUT_PATH = ".hud_input.jsonl"
MCP_CONFIG_PATH = ".hud_mcp_config.json"
RUN_SCRIPT_PATH = ".hud_run.bat"

_MANAGED_CLAUDE_PATHS = {
    "linux-x64": "/media/hud/bin/claude/linux-x64/claude",
    "linux-x64-musl": "/media/hud/bin/claude/linux-x64-musl/claude",
}


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


class ClaudeCLIAgent(Agent):
    """Runs ``claude`` CLI over SSH inside the env workspace.

    Stateless w.r.t. the env: driven by ``await agent(run)``. SSH is opened
    live off the run. Environment MCP bindings are used directly; computer MCP
    servers are bridged over the run's SSH connection.
    """

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
        executable = await resolve_executable(
            ssh,
            "claude",
            _MANAGED_CLAUDE_PATHS,
            run.runtime_config,
        )

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

            await self._exec(
                run,
                ssh=ssh,
                shell=shell,
                mcp_servers=mcp_servers,
                prompt=run.prompt_text,
                executable=executable,
            )

    async def _exec(
        self,
        run: Run,
        *,
        ssh: SSHClient,
        shell: str,
        mcp_servers: dict[str, dict[str, Any]],
        prompt: str,
        executable: str = "claude",
    ) -> None:
        mcp_config_path = await self._write_mcp_config(ssh, mcp_servers)
        input_text = (
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    },
                }
            )
            + "\n"
        )
        files = [mcp_config_path] if mcp_config_path else []
        if shell in WINDOWS_SHELLS:
            await ssh.write_text(INPUT_PATH, input_text)
            files.append(INPUT_PATH)

        command = self._build_cli_command(
            shell=shell,
            mcp_config_path=mcp_config_path,
            executable=executable,
        )
        if shell in WINDOWS_SHELLS:
            await ssh.write_text(RUN_SCRIPT_PATH, f"@echo off\r\n{command}\r\n")
            files.append(RUN_SCRIPT_PATH)
            command = f"cmd /c {RUN_SCRIPT_PATH}"

        try:
            logger.info("SSH exec claude CLI (%d chars)", len(command))
            events = ClaudeEvents(run, started_at=now_iso())
            returncode, stderr = await run_jsonl(
                ssh,
                command,
                events.consume,
                input_text=None if shell in WINDOWS_SHELLS else input_text,
            )
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

    def _build_env_vars(self) -> dict[str, str]:
        env: dict[str, str] = {}
        use_hud_gateway = self.config.use_hud_gateway
        if use_hud_gateway is None:
            use_hud_gateway = settings.api_key is not None

        if use_hud_gateway:
            if not settings.api_key:
                raise ValueError("HUD_API_KEY is required for HUD gateway routing")
            env["ANTHROPIC_BASE_URL"] = settings.hud_gateway_url
            env["ANTHROPIC_API_KEY"] = settings.api_key
            env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
            env["DISABLE_AUTO_COMPACT"] = "1"
            if trace_id := get_current_trace_id():
                env["ANTHROPIC_CUSTOM_HEADERS"] = f"Trace-Id: {trace_id}"
        elif settings.anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

        env["ANTHROPIC_MODEL"] = self.config.model
        env["ANTHROPIC_SMALL_FAST_MODEL"] = self.config.model

        # A custom base URL must own every model tier; otherwise background calls
        # can escape to Anthropic instead of using the configured gateway.
        if "ANTHROPIC_BASE_URL" in env:
            for name in (
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "CLAUDE_CODE_SUBAGENT_MODEL",
            ):
                env[name] = self.config.model

        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["DISABLE_AUTOUPDATER"] = "1"
        env["IS_SANDBOX"] = "1"
        return env

    async def _write_mcp_config(
        self,
        ssh: SSHClient,
        mcp_servers: dict[str, dict[str, Any]],
    ) -> str | None:
        """Write MCP config into the workspace and return its path."""
        if not mcp_servers:
            return None
        await ssh.write_text(
            MCP_CONFIG_PATH,
            json.dumps({"mcpServers": mcp_servers}, indent=2),
        )
        return MCP_CONFIG_PATH

    def _build_cli_command(
        self,
        *,
        shell: str,
        mcp_config_path: str | None = None,
        executable: str = "claude",
    ) -> str:
        env = self._build_env_vars()
        args: list[str] = [
            executable,
            "--verbose",
            "--input-format=stream-json",
            "--output-format=stream-json",
            "--print",
            f"--permission-mode={self.config.permission_mode}",
        ]
        if self.config.max_steps > 0:
            args.append(f"--max-turns={self.config.max_steps}")
        if self.config.system_prompt:
            args.extend(["--system-prompt", self.config.system_prompt])
        for tool in self.config.allowed_tools:
            args.extend(["--allowedTools", tool])
        if mcp_config_path:
            args.extend(["--mcp-config", mcp_config_path])

        if shell in WINDOWS_SHELLS:
            script = ";".join(
                [
                    *(f"$env:{key}={powershell_quote(value)}" for key, value in env.items()),
                    f"Get-Content -Raw -Encoding UTF8 {powershell_quote(INPUT_PATH)}"
                    f" | & {powershell_quote(executable)} "
                    f"{' '.join(powershell_quote(arg) for arg in args[1:])}",
                    "exit $LASTEXITCODE",
                ]
            )
            return powershell(script)

        command = " ".join(shlex.quote(arg) for arg in args)
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        return f'export PATH="$HOME/.local/bin:$PATH"; {env_prefix} {command}'


__all__ = ["ClaudeCLIAgent"]
