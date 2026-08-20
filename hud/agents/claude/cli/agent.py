"""ClaudeCLIAgent — runs ``claude`` CLI over SSH inside the env workspace.

SSH-execs the ``claude`` CLI on the remote workspace so all built-in tools
(Bash, Read, Write, Edit, Glob, Grep) operate on the env's filesystem.
MCP capabilities from the manifest are written as MCP server config so the
CLI can call env-hosted MCP tools too.

Inspired by harbor-framework/harbor's ClaudeCode agent.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import shlex
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import asyncssh
import mcp.types as mcp_types
from anthropic.types.beta import BetaMessage

from hud.agents.base import Agent
from hud.agents.claude.agent import ClaudeAgent
from hud.agents.types import ClaudeCLIConfig, ToolStep
from hud.settings import settings
from hud.telemetry.context import get_current_trace_id
from hud.types import MCPToolCall, MCPToolResult, Step
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.capabilities import SSHClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

WINDOWS_SHELLS = ("cmd", "powershell")
_PROMPT_PATH = ".hud_prompt.txt"
_MCP_CONFIG_PATH = ".hud_mcp_config.json"
_RUN_SCRIPT_PATH = ".hud_run.bat"
_PROCESS_CLOSE_TIMEOUT_S = 5.0


@dataclass(slots=True)
class RemoteInvocation:
    """How to run an assembled CLI command on the remote workspace shell.

    ``command`` is what gets exec'd over SSH. When ``script_name`` is set, that
    file must be written (with ``script_body``) before exec'ing ``command``.
    """

    command: str
    script_name: str | None = None
    script_body: str | None = None


@dataclass(slots=True)
class _PendingToolCall:
    call: MCPToolCall
    started_at: str


class _ClaudeStreamParser:
    """Translate Claude CLI stream messages into canonical HUD steps."""

    def __init__(self, run: Run, *, started_at: str) -> None:
        self._run = run
        self._agent_started_at = started_at
        self._pending_calls: dict[str, _PendingToolCall] = {}
        self._last_agent_content = ""
        self._message_count = 0
        self._saw_result = False
        self._error_recorded = False

    @property
    def message_count(self) -> int:
        return self._message_count

    def feed_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Ignoring non-JSON Claude stream output")
            return
        if not isinstance(raw, dict):
            logger.warning("Ignoring non-object Claude stream message")
            return

        message = cast("dict[str, Any]", raw)
        self._message_count += 1
        received_at = now_iso()
        match message.get("type"):
            case "system" if message.get("subtype") == "init":
                self._agent_started_at = received_at
            case "assistant":
                self._record_assistant(message, received_at)
            case "user":
                self._record_tool_results(message, received_at)
            case "result":
                self._record_result(message, received_at)

    def finish(self, *, returncode: int, stderr: str) -> None:
        trace = self._run.trace
        if returncode != 0:
            trace.extra["returncode"] = returncode
        if stderr and (returncode != 0 or trace.status == "error"):
            trace.extra["stderr"] = stderr
        if not trace.content and self._last_agent_content:
            trace.content = self._last_agent_content

        if returncode != 0:
            trace.status = "error"
            self._record_error(stderr.strip() or f"claude CLI exited with return code {returncode}")
        elif not self._saw_result:
            trace.status = "error"
            self._record_error("claude CLI exited without a result event")
        elif self._pending_calls:
            trace.status = "error"
            missing = ", ".join(sorted(self._pending_calls))
            self._record_error(f"claude CLI exited without results for tool calls: {missing}")

    def _record_assistant(self, event: dict[str, Any], received_at: str) -> None:
        raw_message = event.get("message")
        if not isinstance(raw_message, dict):
            raise ValueError("Claude assistant event is missing its message payload")
        message = BetaMessage.model_validate(raw_message)
        step = ClaudeAgent._message_to_agent_step(message)
        step.started_at = self._agent_started_at
        step.ended_at = received_at
        step.extra = _event_metadata(event, raw_message)
        if step.content:
            self._last_agent_content = step.content
        self._run.record(step)
        for call in step.tool_calls:
            self._pending_calls[call.id] = _PendingToolCall(call=call, started_at=received_at)

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
            block = cast("dict[str, Any]", raw_block)
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str):
                continue
            pending = self._pending_calls.pop(call_id, None)
            if pending is None:
                logger.warning("Claude returned a result for unknown tool call %s", call_id)
                continue
            saw_result = True
            self._run.record(
                ToolStep(
                    call=pending.call,
                    result=MCPToolResult(
                        call_id=call_id,
                        content=_tool_result_content(block.get("content")),
                        isError=block.get("is_error") is True,
                    ),
                    started_at=pending.started_at,
                    ended_at=received_at,
                    extra=_event_metadata(event, message),
                )
            )
        if saw_result:
            self._agent_started_at = received_at

    def _record_result(self, event: dict[str, Any], received_at: str) -> None:
        self._saw_result = True
        trace = self._run.trace
        result = event.get("result")
        trace.content = result if isinstance(result, str) else self._last_agent_content
        is_error = event.get("is_error") is True
        trace.status = "error" if is_error else "completed"
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
        if is_error:
            self._record_error(trace.content or "claude CLI reported an error", received_at)

    def _record_error(self, error: str, at: str | None = None) -> None:
        if self._error_recorded:
            return
        timestamp = at or now_iso()
        self._run.record(
            Step(source="system", error=error, started_at=timestamp, ended_at=timestamp)
        )
        self._error_recorded = True


def _tool_result_content(value: Any) -> list[mcp_types.ContentBlock]:
    values = value if isinstance(value, list) else [value]
    content: list[mcp_types.ContentBlock] = []
    for item in values:
        if isinstance(item, str):
            content.append(mcp_types.TextContent(type="text", text=item))
        elif (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ):
            content.append(mcp_types.TextContent(type="text", text=item["text"]))
        elif isinstance(item, dict) and item.get("type") == "image":
            source = item.get("source")
            if (
                isinstance(source, dict)
                and source.get("type") == "base64"
                and isinstance(source.get("data"), str)
                and isinstance(source.get("media_type"), str)
            ):
                content.append(
                    mcp_types.ImageContent(
                        type="image",
                        data=source["data"],
                        mimeType=source["media_type"],
                    )
                )
                continue
            content.append(
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                )
            )
        elif item is not None:
            content.append(
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                )
            )
    return content


def _event_metadata(event: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("session_id", "uuid", "parent_tool_use_id"):
        value = event.get(key)
        if value is not None:
            metadata[key] = value
    message_id = message.get("id")
    if message_id is not None:
        metadata["message_id"] = message_id
    return metadata


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _powershell(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"


def build_remote_invocation(shell: str, run_cmd: str) -> RemoteInvocation:
    """Build the remote exec command for ``run_cmd`` under the given login shell.

    Windows shells can't take the assembled command inline — ``cmd.exe`` mangles
    the quotes — so it is written to a batch file and invoked through ``cmd /c``.
    A bare ``.hud_run.bat`` is rejected as an unknown command, and silently fails
    to run under a PowerShell default shell, so ``cmd /c`` is required for both.
    POSIX shells take the command inline.
    """
    if shell in WINDOWS_SHELLS:
        return RemoteInvocation(
            command=f"cmd /c {_RUN_SCRIPT_PATH}",
            script_name=_RUN_SCRIPT_PATH,
            script_body=f"@echo off\r\n{run_cmd}\r\n",
        )
    return RemoteInvocation(command=run_cmd)


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
        manifest = run.client.manifest
        bindings = manifest.bindings if manifest is not None else []
        families = {c.protocol.split("/", 1)[0] for c in bindings}

        if "ssh" not in families:
            raise RuntimeError("ClaudeCLIAgent requires an SSH capability")
        ssh = cast("SSHClient", await run.client.open("ssh"))
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
                    from hud.agents.claude.cli.computer_mcp import bridge_computer_mcp

                    server_name = (
                        "computer-use" if len(rfb_bindings) == 1 else f"computer-use-{cap.name}"
                    )
                    if server_name in mcp_servers:
                        raise RuntimeError(f"duplicate MCP server name {server_name!r}")
                    routed = run.client.binding(cap.name)
                    mcp_servers[server_name] = await resources.enter_async_context(
                        bridge_computer_mcp(
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
                max_steps=self.config.max_steps,
                system_prompt=self.config.system_prompt,
            )

    async def _exec(
        self,
        run: Run,
        *,
        ssh: SSHClient,
        shell: str,
        mcp_servers: dict[str, dict[str, Any]],
        prompt: str,
        max_steps: int = -1,
        system_prompt: str | None = None,
    ) -> None:
        runtime_files: list[str] = []
        try:
            mcp_config_path = await self._write_mcp_config(ssh, mcp_servers)
            if mcp_config_path is not None:
                runtime_files.append(mcp_config_path)
            if shell in WINDOWS_SHELLS:
                await ssh.write_text(_PROMPT_PATH, prompt)
                runtime_files.append(_PROMPT_PATH)

            run_cmd = self._build_cli_command(
                shell=shell,
                prompt=prompt,
                max_steps=max_steps,
                system_prompt=system_prompt,
                mcp_config_path=mcp_config_path,
            )
            invocation = build_remote_invocation(shell, run_cmd)
            if invocation.script_name is not None:
                assert invocation.script_body is not None
                await ssh.write_text(invocation.script_name, invocation.script_body)
                runtime_files.append(invocation.script_name)

            logger.info("SSH exec claude CLI (%d chars)", len(invocation.command))
            await self._stream_cli(run, ssh, invocation.command)
        finally:
            await self._remove_runtime_files(ssh, shell, runtime_files)

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
            try:
                process.terminate()
            except (OSError, asyncssh.Error):
                process.close()
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
            try:
                async with asyncio.timeout(_PROCESS_CLOSE_TIMEOUT_S):
                    await process.wait_closed()
            except (OSError, TimeoutError, asyncssh.Error):
                process.close()
                with contextlib.suppress(OSError, TimeoutError, asyncssh.Error):
                    async with asyncio.timeout(_PROCESS_CLOSE_TIMEOUT_S):
                        await process.wait_closed()
            raise

        stderr = stderr_output.decode(errors="replace")
        returncode = process.returncode
        if returncode is None:
            raise RuntimeError("claude CLI process closed without an exit status")
        logger.info(
            "exit=%s events=%d stderr=%d",
            returncode,
            parser.message_count,
            len(stderr),
        )
        parser.finish(returncode=returncode, stderr=stderr)

    async def _remove_runtime_files(
        self,
        ssh: SSHClient,
        shell: str,
        paths: list[str],
    ) -> None:
        if not paths:
            return
        if shell in WINDOWS_SHELLS:
            command = f"cmd /c del /f /q {' '.join(paths)} 2>nul"
        else:
            command = "rm -f -- " + " ".join(shlex.quote(path) for path in paths)
        try:
            await ssh.run(command, check=False)
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
            if trace_id := get_current_trace_id():
                env["ANTHROPIC_CUSTOM_HEADERS"] = f"Trace-Id: {trace_id}"
        elif settings.anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

        env["ANTHROPIC_MODEL"] = self.config.model
        env["ANTHROPIC_SMALL_FAST_MODEL"] = self.config.model

        # When using a custom base URL, alias all model tiers to the same model
        # so the CLI doesn't try to reach Anthropic for background requests.
        if "ANTHROPIC_BASE_URL" in env:
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = self.config.model
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = self.config.model
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = self.config.model
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = self.config.model

        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
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
        mcp_json = json.dumps({"mcpServers": mcp_servers}, indent=2)
        path = _MCP_CONFIG_PATH
        await ssh.write_text(path, mcp_json)
        logger.info("Wrote MCP config")
        return path

    def _build_cli_command(
        self,
        *,
        shell: str,
        prompt: str,
        max_steps: int,
        system_prompt: str | None,
        mcp_config_path: str | None = None,
    ) -> str:
        env_vars = self._build_env_vars()
        is_win = shell in WINDOWS_SHELLS

        base_args: list[str] = [
            "claude",
            "--verbose",
            "--output-format=stream-json",
            "--print",
            f"--permission-mode={self.config.permission_mode}",
        ]
        if max_steps > 0:
            base_args.append(f"--max-turns={max_steps}")
        if system_prompt:
            base_args.extend(["--system-prompt", system_prompt])
        for tool in self.config.allowed_tools:
            base_args.extend(["--allowedTools", tool])
        if mcp_config_path:
            base_args.extend(["--mcp-config", mcp_config_path])

        if is_win:
            script = ";".join(
                [
                    *(f"$env:{key}={_powershell_quote(value)}" for key, value in env_vars.items()),
                    f"Get-Content -Raw -Encoding UTF8 {_powershell_quote(_PROMPT_PATH)}"
                    f" | & claude {' '.join(_powershell_quote(arg) for arg in base_args[1:])}",
                    "exit $LASTEXITCODE",
                ]
            )
            return _powershell(script)

        # POSIX path: shell-quote everything and embed prompt inline.
        cli_parts = [shlex.quote(a) for a in base_args]
        cli_parts.extend(["--", shlex.quote(prompt)])
        cli_cmd = " ".join(cli_parts)
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items())
        return f'export PATH="$HOME/.local/bin:$PATH"; {env_prefix} {cli_cmd}'


__all__ = ["ClaudeCLIAgent", "ClaudeCLIConfig", "RemoteInvocation", "build_remote_invocation"]
