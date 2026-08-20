"""Claude CLI harness over a workspace SSH capability."""

from __future__ import annotations

import asyncio
import base64
import contextlib
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

_WINDOWS_SHELLS = ("cmd", "powershell")
_PROMPT_PATH = ".hud_prompt.txt"
_MCP_CONFIG_PATH = ".hud_mcp_config.json"
_RUN_SCRIPT_PATH = ".hud_run.bat"
_PROCESS_CLOSE_TIMEOUT_S = 5.0


class _ClaudeStreamParser:
    """Translate Claude CLI stream messages into canonical HUD steps."""

    def __init__(self, run: Run, *, started_at: str) -> None:
        self._run = run
        self._agent_started_at = started_at
        self._pending_calls: dict[str, tuple[MCPToolCall, str]] = {}
        self._last_agent_content = ""
        self._saw_result = False
        self._result_error: str | None = None

    def feed_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        message = json.loads(line)
        if not isinstance(message, dict):
            raise ValueError("Claude stream event must be an object")
        received_at = now_iso()
        match message.get("type"):
            case "system" if message.get("subtype") == "init":
                self._agent_started_at = received_at
            case "assistant":
                self._record_assistant(message, received_at)
            case "user":
                self._record_tool_results(message, received_at)
            case "result":
                self._record_result(message)

    def finish(self, *, returncode: int, stderr: str) -> None:
        trace = self._run.trace
        error = self._result_error
        if returncode != 0:
            trace.extra["returncode"] = returncode
            error = stderr.strip() or f"claude CLI exited with return code {returncode}"
        elif not self._saw_result:
            error = "claude CLI exited without a result event"
        elif self._pending_calls:
            missing = ", ".join(sorted(self._pending_calls))
            error = f"claude CLI exited without results for tool calls: {missing}"

        if not trace.content and self._last_agent_content:
            trace.content = self._last_agent_content
        if error is not None and stderr:
            trace.extra["stderr"] = stderr
        if error is not None:
            raise RuntimeError(error)

    def _record_assistant(self, event: dict[str, Any], received_at: str) -> None:
        message = BetaMessage.model_validate(event["message"])
        step = ClaudeAgent._message_to_agent_step(message)
        step.started_at = self._agent_started_at
        step.ended_at = received_at
        if step.content:
            self._last_agent_content = step.content
        self._run.record(step)
        for call in step.tool_calls:
            self._pending_calls[call.id] = (call, received_at)

    def _record_tool_results(self, event: dict[str, Any], received_at: str) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, list):
            return

        saw_result = False
        for raw_block in content:
            if not isinstance(raw_block, dict) or raw_block.get("type") != "tool_result":
                continue
            call_id = raw_block.get("tool_use_id")
            if not isinstance(call_id, str):
                raise ValueError("Claude tool result is missing tool_use_id")
            try:
                call, started_at = self._pending_calls.pop(call_id)
            except KeyError:
                raise ValueError(
                    f"Claude returned a result for unknown tool call {call_id!r}"
                ) from None

            raw_result = raw_block.get("content")
            raw_items = raw_result if isinstance(raw_result, list) else [raw_result]
            result_content: list[mcp_types.ContentBlock] = []
            for item in raw_items:
                if isinstance(item, str):
                    result_content.append(mcp_types.TextContent(type="text", text=item))
                elif isinstance(item, dict) and item.get("type") == "text":
                    result_content.append(mcp_types.TextContent(type="text", text=item["text"]))
                elif isinstance(item, dict) and item.get("type") == "image":
                    source = item["source"]
                    result_content.append(
                        mcp_types.ImageContent(
                            type="image",
                            data=source["data"],
                            mimeType=source["media_type"],
                        )
                    )
                elif item is not None:
                    raise ValueError(f"unsupported Claude tool result block: {item!r}")

            saw_result = True
            self._run.record(
                ToolStep(
                    call=call,
                    result=MCPToolResult(
                        call_id=call_id,
                        content=result_content,
                        isError=raw_block.get("is_error") is True,
                    ),
                    started_at=started_at,
                    ended_at=received_at,
                )
            )
        if saw_result:
            self._agent_started_at = received_at

    def _record_result(self, event: dict[str, Any]) -> None:
        self._saw_result = True
        trace = self._run.trace
        result = event.get("result")
        trace.content = result if isinstance(result, str) else self._last_agent_content
        if event.get("is_error") is True:
            self._result_error = trace.content or "claude CLI reported an error"
        for key in (
            "subtype",
            "session_id",
            "duration_ms",
            "duration_api_ms",
            "stop_reason",
            "num_turns",
            "total_cost_usd",
        ):
            value = event.get(key)
            if value is not None:
                trace.extra[key] = value


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _powershell(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"


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

            await self._run_cli(
                run,
                ssh=ssh,
                shell=shell,
                mcp_servers=mcp_servers,
                prompt=run.prompt_text,
            )

    async def _run_cli(
        self,
        run: Run,
        *,
        ssh: SSHClient,
        shell: str,
        mcp_servers: dict[str, dict[str, Any]],
        prompt: str,
    ) -> None:
        files: dict[str, str] = {}
        mcp_config_path = _MCP_CONFIG_PATH if mcp_servers else None
        if mcp_servers:
            files[_MCP_CONFIG_PATH] = json.dumps({"mcpServers": mcp_servers}, indent=2)
        if shell in _WINDOWS_SHELLS:
            files[_PROMPT_PATH] = prompt

        command = self._build_command(
            shell=shell,
            prompt=prompt,
            mcp_config_path=mcp_config_path,
        )
        if shell in _WINDOWS_SHELLS:
            files[_RUN_SCRIPT_PATH] = f"@echo off\r\n{command}\r\n"
            command = f"cmd /c {_RUN_SCRIPT_PATH}"

        try:
            for path, content in files.items():
                await ssh.write_text(path, content)
            logger.info("SSH exec claude CLI (%d chars)", len(command))
            await self._stream_cli(run, ssh, command)
        finally:
            if files:
                if shell in _WINDOWS_SHELLS:
                    cleanup = f"cmd /c del /f /q {' '.join(files)} 2>nul"
                else:
                    cleanup = "rm -f -- " + " ".join(shlex.quote(path) for path in files)
                try:
                    await ssh.run(cleanup, check=False)
                except (OSError, asyncssh.Error):
                    logger.warning("Failed to remove Claude CLI runtime files")

    async def _stream_cli(self, run: Run, ssh: SSHClient, command: str) -> None:
        parser = _ClaudeStreamParser(run, started_at=now_iso())
        process = await ssh.create_process(command)
        stderr_task = asyncio.create_task(process.stderr.read())
        try:
            while line := await process.stdout.readline():
                parser.feed_line(line.decode(errors="replace"))
            await process.wait_closed()
            stderr_output = await stderr_task
        except BaseException:
            process.close()
            if not stderr_task.done():
                stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            with contextlib.suppress(OSError, TimeoutError, asyncssh.Error):
                async with asyncio.timeout(_PROCESS_CLOSE_TIMEOUT_S):
                    await process.wait_closed()
            raise

        stderr = stderr_output.decode(errors="replace")
        returncode = process.returncode
        if returncode is None:
            raise RuntimeError("claude CLI process closed without an exit status")
        logger.info("exit=%s stderr=%d", returncode, len(stderr))
        parser.finish(returncode=returncode, stderr=stderr)

    def _build_command(
        self,
        *,
        shell: str,
        prompt: str,
        mcp_config_path: str | None = None,
    ) -> str:
        env: dict[str, str] = {}
        use_hud_gateway = self.config.use_hud_gateway
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

        env["ANTHROPIC_MODEL"] = self.config.model
        env["ANTHROPIC_SMALL_FAST_MODEL"] = self.config.model

        # When using a custom base URL, alias all model tiers to the same model
        # so the CLI doesn't try to reach Anthropic for background requests.
        if "ANTHROPIC_BASE_URL" in env:
            for name in (
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "CLAUDE_CODE_SUBAGENT_MODEL",
            ):
                env[name] = self.config.model

        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["IS_SANDBOX"] = "1"

        base_args: list[str] = [
            "claude",
            "--verbose",
            "--output-format=stream-json",
            "--print",
            f"--permission-mode={self.config.permission_mode}",
        ]
        if self.config.max_steps > 0:
            base_args.append(f"--max-turns={self.config.max_steps}")
        if self.config.system_prompt:
            base_args.extend(["--system-prompt", self.config.system_prompt])
        for tool in self.config.allowed_tools:
            base_args.extend(["--allowedTools", tool])
        if mcp_config_path:
            base_args.extend(["--mcp-config", mcp_config_path])

        if shell in _WINDOWS_SHELLS:
            script = ";".join(
                [
                    *(f"$env:{key}={_powershell_quote(value)}" for key, value in env.items()),
                    f"Get-Content -Raw -Encoding UTF8 {_powershell_quote(_PROMPT_PATH)}"
                    f" | & claude {' '.join(_powershell_quote(arg) for arg in base_args[1:])}",
                    "exit $LASTEXITCODE",
                ]
            )
            return _powershell(script)

        cli_parts = [shlex.quote(a) for a in base_args]
        cli_parts.extend(["--", shlex.quote(prompt)])
        cli_cmd = " ".join(cli_parts)
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        return f'export PATH="$HOME/.local/bin:$PATH"; {env_prefix} {cli_cmd}'


__all__ = ["ClaudeCLIAgent"]
