"""ClaudeSDKAgent — runs ``claude`` CLI over SSH inside the env workspace.

SSH-execs the ``claude`` CLI on the remote workspace so all built-in tools
(Bash, Read, Write, Edit, Glob, Grep) operate on the env's filesystem.
MCP capabilities from the manifest are written as MCP server config so the
CLI can call env-hosted MCP tools too.

Inspired by harbor-framework/harbor's ClaudeCode agent.
"""

from __future__ import annotations

import json
import logging
import shlex
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from hud.agents.base import Agent
from hud.agents.tool_agent import to_prompt_messages
from hud.agents.types import ClaudeSDKConfig
from hud.settings import settings

if TYPE_CHECKING:
    from hud.capabilities import RFBClient, SSHClient
    from hud.client import Run
    from hud.types import Trace

logger = logging.getLogger(__name__)


def _prompt_text(prompt: str | list[Any] | None) -> str:
    """Flatten a run prompt (text or chat-style message list) into CLI prompt text."""
    parts: list[str] = []
    for message in to_prompt_messages(prompt):
        text = getattr(message.content, "text", "")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n\n".join(parts)


@dataclass
class _SDKRun:
    """Per-rollout state: the live SSH client, remote shell, and MCP server config.

    Held in a local (never on ``self``) so one agent instance can drive many
    concurrent rollouts without races.
    """

    ssh: SSHClient
    shell: str
    mcp_servers: dict[str, dict[str, Any]]


class ClaudeSDKAgent(Agent):
    """Runs ``claude`` CLI over SSH inside the env workspace.

    Stateless w.r.t. the env: driven by ``await agent(run)``. SSH and RFB are
    opened live off the run (we
    drive them); MCP servers are read as raw bindings and written into the CLI's
    MCP config (the CLI connects to them itself).
    """

    def __init__(self, config: ClaudeSDKConfig | None = None) -> None:
        self.config = config or ClaudeSDKConfig()
        self.model = self.config.model

    async def __call__(
        self,
        run: Run,
        *,
        max_steps: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        manifest = run.client.manifest
        bindings = manifest.bindings if manifest is not None else []
        families = {c.protocol.split("/", 1)[0] for c in bindings}

        if "ssh" not in families:
            raise RuntimeError("ClaudeSDKAgent requires an SSH capability")
        ssh = cast("SSHClient", await run.client.open("ssh"))
        shell = ssh.capability.params.get("shell", "bash")
        mcp_servers: dict[str, dict[str, Any]] = {}

        async with AsyncExitStack() as stack:
            for cap in bindings:
                family = cap.protocol.split("/", 1)[0]
                if family == "mcp":
                    token = cap.params.get("auth_token")
                    transport = "http" if cap.url.startswith("http") else "sse"
                    server_config: dict[str, Any] = {"type": transport, "url": cap.url}
                    if token:
                        server_config["headers"] = {"Authorization": f"Bearer {token}"}
                    mcp_servers[cap.name] = server_config
                elif family == "rfb":
                    from hud.agents.claude.sdk.computer_mcp import computer_mcp_server

                    rfb = cast("RFBClient", await run.client.open("rfb"))
                    # Scope the local computer-use MCP server to this rollout: the
                    # exit stack tears it down when __call__ returns.
                    port = await stack.enter_async_context(computer_mcp_server(rfb))
                    mcp_servers["computer-use"] = {
                        "type": "http",
                        "url": f"http://127.0.0.1:{port}/mcp",
                    }

            sdk_run = _SDKRun(ssh=ssh, shell=shell, mcp_servers=mcp_servers)
            await self._exec(
                run.trace,
                sdk_run,
                prompt=_prompt_text(run.prompt),
                max_steps=max_steps if max_steps is not None else self.config.max_turns or -1,
                system_prompt=system_prompt,
            )

    async def _exec(
        self,
        trace: Trace,
        sdk_run: _SDKRun,
        *,
        prompt: str,
        max_steps: int = -1,
        system_prompt: str | None = None,
    ) -> None:
        ssh = sdk_run.ssh
        mcp_config_path = await self._write_mcp_config(sdk_run)

        # Write prompt to file via SFTP — avoids all shell quoting issues.
        async with ssh.conn.start_sftp_client() as sftp, sftp.open(".hud_prompt.txt", "wb") as f:
            await f.write(prompt.encode("utf-8"))

        run_cmd = self._build_cli_command(
            sdk_run.shell,
            prompt=prompt,
            max_steps=max_steps,
            system_prompt=system_prompt,
            mcp_config_path=mcp_config_path,
        )

        if sdk_run.shell in ("cmd", "powershell"):
            # Write command to bat file — cmd.exe mangles inline quotes.
            bat_content = f"@echo off\r\n{run_cmd}\r\n"
            async with ssh.conn.start_sftp_client() as sftp, sftp.open(".hud_run.bat", "wb") as f:
                await f.write(bat_content.encode("utf-8"))
            full_cmd = ".hud_run.bat"
        else:
            parts: list[str] = [
                'command -v claude >/dev/null 2>&1 || '
                '{ curl -fsSL https://claude.ai/install.sh | bash -s -- 2>/dev/null; '
                'export PATH="$HOME/.local/bin:$PATH"; }',
                run_cmd,
            ]
            full_cmd = " && ".join(parts)

        # Do not log full_cmd: it embeds ANTHROPIC_API_KEY / HUD_API_KEY in the env prefix.
        logger.info("SSH exec claude CLI (%d chars)", len(full_cmd))

        completed = await ssh.conn.run(full_cmd, check=False)
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""

        logger.info("exit=%s stdout=%d stderr=%d", completed.exit_status, len(stdout), len(stderr))

        if completed.exit_status != 0 and not stdout.strip():
            trace.finish(
                stderr or f"claude CLI exited with status {completed.exit_status}",
                isError=True,
                exit_status=completed.exit_status,
                stderr=stderr,
            )
            return

        self._parse_stream_json(trace, stdout, stderr)

    def _build_env_vars(self) -> dict[str, str]:
        env: dict[str, str] = {}

        if settings.api_key:
            env["ANTHROPIC_BASE_URL"] = settings.hud_gateway_url
            env["ANTHROPIC_API_KEY"] = settings.api_key
        elif settings.anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

        env["ANTHROPIC_MODEL"] = self.model
        env["ANTHROPIC_SMALL_FAST_MODEL"] = self.model

        # When using a custom base URL, alias all model tiers to the same model
        # so the CLI doesn't try to reach Anthropic for background requests.
        if "ANTHROPIC_BASE_URL" in env:
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = self.model
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = self.model
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = self.model
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = self.model

        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["IS_SANDBOX"] = "1"

        return env

    async def _write_mcp_config(self, sdk_run: _SDKRun) -> str | None:
        """Write MCP config via SFTP and return the file path, or None."""
        if not sdk_run.mcp_servers:
            return None
        mcp_json = json.dumps({"mcpServers": sdk_run.mcp_servers}, indent=2)
        # Write into the workspace root (SFTP is chrooted there).
        sftp_path = ".hud_mcp_config.json"
        async with sdk_run.ssh.conn.start_sftp_client() as sftp, sftp.open(sftp_path, "wb") as f:
            await f.write(mcp_json.encode("utf-8"))
        # Return the absolute path the CLI will see (cwd = workspace root).
        logger.info("Wrote MCP config via SFTP")
        return sftp_path

    def _build_cli_command(
        self,
        shell: str,
        *,
        prompt: str,
        max_steps: int,
        system_prompt: str | None,
        mcp_config_path: str | None = None,
    ) -> str:
        env_vars = self._build_env_vars()
        is_win = shell in ("cmd", "powershell")

        def q(s: str) -> str:
            if is_win:
                escaped = s.replace('"', '""')
                return f'"{escaped}"'
            return shlex.quote(s)

        cli_parts = [
            "claude",
            "--verbose",
            "--output-format=stream-json",
            "--print",
            f"--permission-mode={self.config.permission_mode}",
        ]
        if max_steps > 0:
            cli_parts.append(f"--max-turns={max_steps}")
        effective_system = system_prompt or self.config.system_prompt
        if effective_system:
            cli_parts.extend(["--system-prompt", q(effective_system)])
        for tool in self.config.allowed_tools:
            cli_parts.extend(["--allowedTools", tool])
        if mcp_config_path:
            cli_parts.extend(["--mcp-config", mcp_config_path])

        cli_parts.extend(["--", q(prompt)])

        cli_cmd = " ".join(cli_parts)

        if is_win:
            set_parts = [f"set {k}={v}" for k, v in env_vars.items()]
            return " && ".join([*set_parts, cli_cmd])

        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items())
        return f'export PATH="$HOME/.local/bin:$PATH"; {env_prefix} {cli_cmd}'

    def _parse_stream_json(self, trace: Trace, stdout: str, stderr: str) -> None:
        messages: list[dict[str, Any]] = []
        content_parts: list[str] = []
        is_error = False
        info: dict[str, Any] = {}

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages.append(msg)
            msg_type = msg.get("type")

            if msg_type == "assistant" and isinstance(msg.get("message"), dict):
                message = cast("dict[str, Any]", msg["message"])
                for raw_block in cast("list[Any]", message.get("content", [])):
                    if not isinstance(raw_block, dict):
                        continue
                    block = cast("dict[str, Any]", raw_block)
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if isinstance(text, str) and text:
                            content_parts.append(text)

            elif msg_type == "result":
                is_error = msg.get("is_error", False)
                result_text = msg.get("result")
                if result_text:
                    content_parts.append(result_text)
                info["session_id"] = msg.get("session_id")
                info["num_turns"] = msg.get("num_turns")
                info["duration_ms"] = msg.get("duration_ms")
                info["stop_reason"] = msg.get("stop_reason")
                cost = msg.get("total_cost_usd")
                if cost is not None:
                    info["total_cost_usd"] = cost

        if stderr:
            info["stderr"] = stderr

        trace.finish(
            "\n".join(content_parts),
            isError=is_error,
            messages=messages,
            **info,
        )


__all__ = ["ClaudeSDKAgent", "ClaudeSDKConfig"]
