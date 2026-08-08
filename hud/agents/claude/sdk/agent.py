"""ClaudeSDKAgent — runs ``claude`` CLI over SSH inside the env workspace.

SSH-execs the ``claude`` CLI on the remote workspace so all built-in tools
(Bash, Read, Write, Edit, Glob, Grep) operate on the env's filesystem.
MCP capabilities from the manifest are written as MCP server config so the
CLI can call env-hosted MCP tools too.

Inspired by harbor-framework/harbor's ClaudeCode agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import asyncssh
import mcp.types as mcp_types

from hud.agents.base import Agent
from hud.agents.types import AgentStep, ClaudeSDKConfig, ToolStep, Usage
from hud.settings import settings
from hud.telemetry.context import get_current_trace_id
from hud.types import MCPToolCall, MCPToolResult, Step
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.capabilities import RFBClient, SSHClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

WINDOWS_SHELLS = ("cmd", "powershell")
#: Bare ``claude`` install bootstrap for POSIX shells (no-op when already present).
_POSIX_INSTALL_CHECK = (
    "command -v claude >/dev/null 2>&1 || "
    "{ curl -fsSL https://claude.ai/install.sh | bash -s -- 2>/dev/null; "
    'export PATH="$HOME/.local/bin:$PATH"; }'
)
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

    def __init__(self, run: Run, *, model: str, started_at: str) -> None:
        self._run = run
        self._model = model
        self._agent_started_at = started_at
        self._pending_calls: dict[str, _PendingToolCall] = {}
        self._messages: list[dict[str, Any]] = []
        self._last_agent_content = ""
        self._saw_result = False
        self._error_recorded = False

    @property
    def message_count(self) -> int:
        return len(self._messages)

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
        self._messages.append(message)
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

    def finish(self, *, exit_status: int, stderr: str) -> None:
        trace = self._run.trace
        trace.extra["messages"] = self._messages
        trace.extra["exit_status"] = exit_status
        if stderr:
            trace.extra["stderr"] = stderr
        if not trace.content and self._last_agent_content:
            trace.content = self._last_agent_content

        if exit_status != 0:
            trace.status = "error"
            self._record_error(stderr or f"claude CLI exited with status {exit_status}")
        elif not self._saw_result:
            trace.status = "error"
            self._record_error("claude CLI exited without a result message")
        elif self._pending_calls:
            trace.status = "error"
            missing = ", ".join(sorted(self._pending_calls))
            self._record_error(f"claude CLI exited without results for tool calls: {missing}")

    def _record_assistant(self, event: dict[str, Any], received_at: str) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[MCPToolCall] = []
        content = message.get("content")
        if isinstance(content, list):
            for raw_block in content:
                if not isinstance(raw_block, dict):
                    continue
                block = cast("dict[str, Any]", raw_block)
                match block.get("type"):
                    case "text":
                        if isinstance(block.get("text"), str):
                            text_parts.append(block["text"])
                    case "thinking":
                        if isinstance(block.get("thinking"), str):
                            thinking_parts.append(block["thinking"])
                    case "tool_use":
                        call = _tool_call(block)
                        if call is not None:
                            tool_calls.append(call)

        text = "".join(text_parts)
        if text:
            self._last_agent_content = text
        model = message.get("model")
        stop_reason = message.get("stop_reason")
        step = AgentStep(
            content=text,
            reasoning="\n".join(thinking_parts) if thinking_parts else None,
            tool_calls=tool_calls,
            done=not tool_calls,
            finish_reason=stop_reason if isinstance(stop_reason, str) else None,
            model=model if isinstance(model, str) else self._model,
            usage=_usage(message.get("usage")),
            started_at=self._agent_started_at,
            ended_at=received_at,
            extra=_event_metadata(event, message),
        )
        self._run.record(step)
        for call in tool_calls:
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


def _tool_call(block: dict[str, Any]) -> MCPToolCall | None:
    call_id = block.get("id")
    name = block.get("name")
    if not isinstance(call_id, str) or not isinstance(name, str):
        logger.warning("Ignoring malformed Claude tool call")
        return None
    raw_arguments = block.get("input")
    if isinstance(raw_arguments, dict):
        arguments: dict[str, Any] | str = cast("dict[str, Any]", raw_arguments)
    elif isinstance(raw_arguments, str):
        arguments = raw_arguments
    else:
        arguments = json.dumps(raw_arguments, ensure_ascii=False)
    return MCPToolCall(id=call_id, name=name, arguments=arguments)


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
        elif item is not None:
            content.append(
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                )
            )
    return content


def _usage(value: Any) -> Usage | None:
    if not isinstance(value, dict):
        return None
    usage = cast("dict[str, Any]", value)
    normalized = Usage(
        prompt_tokens=_integer(usage.get("input_tokens")),
        completion_tokens=_integer(usage.get("output_tokens")),
        cached_tokens=_integer(usage.get("cache_read_input_tokens")),
    )
    return normalized if any(v is not None for v in normalized.model_dump().values()) else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def build_remote_invocation(shell: str, run_cmd: str) -> RemoteInvocation:
    """Build the remote exec command for ``run_cmd`` under the given login shell.

    Windows shells can't take the assembled command inline — ``cmd.exe`` mangles
    the quotes — so it is written to a batch file and invoked through ``cmd /c``.
    A bare ``.hud_run.bat`` is rejected as an unknown command, and silently fails
    to run under a PowerShell default shell, so ``cmd /c`` is required for both.
    POSIX shells take the command inline, prefixed with a one-shot install check.
    """
    if shell in WINDOWS_SHELLS:
        return RemoteInvocation(
            command="cmd /c .hud_run.bat",
            script_name=".hud_run.bat",
            script_body=f"@echo off\r\n{run_cmd}\r\n",
        )
    return RemoteInvocation(command=f"{_POSIX_INSTALL_CHECK} && {run_cmd}")


class ClaudeSDKAgent(Agent):
    """Runs ``claude`` CLI over SSH inside the env workspace.

    Stateless w.r.t. the env: driven by ``await agent(run)``. SSH and RFB are
    opened live off the run (we
    drive them); MCP servers are read as raw bindings and written into the CLI's
    MCP config (the CLI connects to them itself).
    """

    config: ClaudeSDKConfig

    def __init__(self, config: ClaudeSDKConfig | None = None) -> None:
        self.config = config or ClaudeSDKConfig()
        self._ssh: SSHClient | None = None
        self._mcp_servers: dict[str, dict[str, Any]] = {}
        self._shell = "bash"

    async def __call__(self, run: Run) -> None:
        self._mcp_servers = {}
        manifest = run.client.manifest
        bindings = manifest.bindings if manifest is not None else []
        families = {c.protocol.split("/", 1)[0] for c in bindings}

        if "ssh" not in families:
            raise RuntimeError("ClaudeSDKAgent requires an SSH capability")
        self._ssh = cast("SSHClient", await run.client.open("ssh"))
        self._shell = self._ssh.capability.params.get("shell", "bash")

        for cap in bindings:
            family = cap.protocol.split("/", 1)[0]
            if family == "mcp":
                token = cap.params.get("auth_token")
                transport = "http" if cap.params["transport"] == "streamable-http" else "sse"
                server_config: dict[str, Any] = {"type": transport, "url": cap.url}
                if token:
                    server_config["headers"] = {"Authorization": f"Bearer {token}"}
                self._mcp_servers[cap.name] = server_config
            elif family == "rfb":
                from hud.agents.claude.sdk.computer_mcp import serve_computer_mcp

                rfb = cast("RFBClient", await run.client.open("rfb"))
                port = await serve_computer_mcp(rfb, self.config.screenshot_encoding)
                self._mcp_servers["computer-use"] = {
                    "type": "http",
                    "url": f"http://127.0.0.1:{port}/mcp",
                }

        await self._exec(
            run,
            prompt=run.prompt_text,
            max_steps=self.config.max_steps,
            system_prompt=self.config.system_prompt,
        )

    async def _exec(
        self,
        run: Run,
        *,
        prompt: str,
        max_steps: int = -1,
        system_prompt: str | None = None,
    ) -> None:
        assert self._ssh is not None

        mcp_config_path = await self._write_mcp_config()

        await self._ssh.write_text(".hud_prompt.txt", prompt)

        run_cmd = self._build_cli_command(
            prompt=prompt,
            max_steps=max_steps,
            system_prompt=system_prompt,
            mcp_config_path=mcp_config_path,
        )

        invocation = build_remote_invocation(self._shell, run_cmd)
        if invocation.script_name is not None:
            assert invocation.script_body is not None
            # cmd.exe mangles inline quotes, so the command rides a batch file.
            await self._ssh.write_text(invocation.script_name, invocation.script_body)

        full_cmd = invocation.command
        logger.info("SSH exec claude CLI (%d chars)", len(full_cmd))
        logger.info("Full command: %s", full_cmd)

        parser = _ClaudeStreamParser(run, model=self.config.model, started_at=now_iso())
        process = await self._ssh.create_process(full_cmd)
        stderr_task = asyncio.create_task(process.stderr.read())
        try:
            while line := await process.stdout.readline():
                parser.feed_line(line if isinstance(line, str) else line.decode(errors="replace"))
            await process.wait_closed()
            stderr_output = await stderr_task
        except BaseException:
            process.close()
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
            with contextlib.suppress(OSError, TimeoutError, asyncssh.Error):
                async with asyncio.timeout(_PROCESS_CLOSE_TIMEOUT_S):
                    await process.wait_closed()
            raise

        stderr = (
            stderr_output
            if isinstance(stderr_output, str)
            else stderr_output.decode(errors="replace")
        )
        exit_status = process.exit_status
        if exit_status is None:
            raise RuntimeError("claude CLI process closed without an exit status")
        logger.info(
            "exit=%s messages=%d stderr=%d",
            exit_status,
            parser.message_count,
            len(stderr),
        )
        parser.finish(exit_status=exit_status, stderr=stderr)

    def _build_env_vars(self) -> dict[str, str]:
        env: dict[str, str] = {}

        if settings.api_key:
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

    async def _write_mcp_config(self) -> str | None:
        """Write MCP config into the workspace and return its path."""
        if not self._mcp_servers or self._ssh is None:
            return None
        mcp_json = json.dumps({"mcpServers": self._mcp_servers}, indent=2)
        path = ".hud_mcp_config.json"
        await self._ssh.write_text(path, mcp_json)
        logger.info("Wrote MCP config")
        return path

    def _build_cli_command(
        self,
        *,
        prompt: str,
        max_steps: int,
        system_prompt: str | None,
        mcp_config_path: str | None = None,
    ) -> str:
        env_vars = self._build_env_vars()
        is_win = self._shell in WINDOWS_SHELLS
        self._win_redirect = False

        # Raw args list (no shell quoting) — used directly for Windows Python launcher.
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
            # On Windows, two problems combine:
            #  1. claude is installed as claude.cmd (Node.js wrapper) — Python's
            #     subprocess.run can't execute .cmd files via CreateProcess directly.
            #  2. Embedding the prompt inline in the bat file breaks — cmd.exe parses
            #     line-by-line, so newlines inside quoted strings split the command.
            # Solution: use `cmd /c claude [args]` (no inline prompt) and feed the
            # prompt via stdin from .hud_prompt.txt. claude --print reads stdin as
            # the initial message when no -- argument is provided.
            set_parts = [f"set {k}={v}" for k, v in env_vars.items()]
            cmd_args = ["cmd", "/c", "claude"] + base_args[1:]  # noqa: RUF005
            py_args_repr = "[" + ",".join(f"'{a}'" for a in cmd_args) + "]"
            python_launcher = (
                'python -c "'
                "import subprocess,sys;"
                f"r=subprocess.run({py_args_repr},stdin=open('.hud_prompt.txt','rb'));"
                'sys.exit(r.returncode)"'
            )
            return " && ".join([*set_parts, python_launcher])

        # POSIX path: shell-quote everything and embed prompt inline.
        cli_parts = [shlex.quote(a) for a in base_args]
        cli_parts.extend(["--", shlex.quote(prompt)])
        cli_cmd = " ".join(cli_parts)
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items())
        return f'export PATH="$HOME/.local/bin:$PATH"; {env_prefix} {cli_cmd}'


__all__ = ["ClaudeSDKAgent", "ClaudeSDKConfig", "RemoteInvocation", "build_remote_invocation"]
