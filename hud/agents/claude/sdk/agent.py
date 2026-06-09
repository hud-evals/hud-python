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
from typing import TYPE_CHECKING, Any, cast

from hud.agents.base import Agent
from hud.agents.types import ClaudeSDKConfig
from hud.settings import settings

if TYPE_CHECKING:
    from hud.capabilities import RFBClient, SSHClient
    from hud.client import Run
    from hud.types import Trace

logger = logging.getLogger(__name__)


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
                transport = "http" if cap.url.startswith("http") else "sse"
                server_config: dict[str, Any] = {"type": transport, "url": cap.url}
                if token:
                    server_config["headers"] = {"Authorization": f"Bearer {token}"}
                self._mcp_servers[cap.name] = server_config
            elif family == "rfb":
                from hud.agents.claude.sdk.computer_mcp import serve_computer_mcp

                rfb = cast("RFBClient", await run.client.open("rfb"))
                port = await serve_computer_mcp(rfb)
                self._mcp_servers["computer-use"] = {
                    "type": "http",
                    "url": f"http://127.0.0.1:{port}/mcp",
                }

        await self._exec(
            run.trace,
            prompt=run.prompt or "",
            max_steps=self.config.max_steps,
            system_prompt=self.config.system_prompt,
        )

    async def _exec(
        self,
        trace: Trace,
        *,
        prompt: str,
        max_steps: int = -1,
        system_prompt: str | None = None,
    ) -> None:
        assert self._ssh is not None

        mcp_config_path = await self._write_mcp_config()

        # Write prompt to file via SFTP — avoids all shell quoting issues.
        async with (
            self._ssh.conn.start_sftp_client() as sftp,
            sftp.open(".hud_prompt.txt", "wb") as f,
        ):
            await f.write(prompt.encode("utf-8"))

        run_cmd = self._build_cli_command(
            prompt=prompt,
            max_steps=max_steps,
            system_prompt=system_prompt,
            mcp_config_path=mcp_config_path,
        )

        if self._shell in ("cmd", "powershell"):
            # Write command to bat file — cmd.exe mangles inline quotes.
            bat_content = f"@echo off\r\n{run_cmd}\r\n"
            async with (
                self._ssh.conn.start_sftp_client() as sftp,
                sftp.open(".hud_run.bat", "wb") as f,
            ):
                await f.write(bat_content.encode("utf-8"))
            full_cmd = ".hud_run.bat"
        else:
            parts: list[str] = [
                "command -v claude >/dev/null 2>&1 || "
                "{ curl -fsSL https://claude.ai/install.sh | bash -s -- 2>/dev/null; "
                'export PATH="$HOME/.local/bin:$PATH"; }',
                run_cmd,
            ]
            full_cmd = " && ".join(parts)

        logger.info("SSH exec claude CLI (%d chars)", len(full_cmd))
        logger.info("Full command: %s", full_cmd)

        completed = await self._ssh.conn.run(full_cmd, check=False)
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""

        logger.info("exit=%s stdout=%d stderr=%d", completed.exit_status, len(stdout), len(stderr))

        if completed.exit_status != 0 and not stdout.strip():
            trace.done = True
            trace.content = stderr or f"claude CLI exited with status {completed.exit_status}"
            trace.isError = True
            trace.info.update({"exit_status": completed.exit_status, "stderr": stderr})
            return

        self._parse_stream_json(trace, stdout, stderr)

    def _build_env_vars(self) -> dict[str, str]:
        env: dict[str, str] = {}

        if settings.api_key:
            env["ANTHROPIC_BASE_URL"] = settings.hud_gateway_url
            env["ANTHROPIC_API_KEY"] = settings.api_key
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
        """Write MCP config via SFTP and return the file path, or None."""
        if not self._mcp_servers or self._ssh is None:
            return None
        mcp_json = json.dumps({"mcpServers": self._mcp_servers}, indent=2)
        # Write into the workspace root (SFTP is chrooted there).
        sftp_path = ".hud_mcp_config.json"
        async with self._ssh.conn.start_sftp_client() as sftp, sftp.open(sftp_path, "wb") as f:
            await f.write(mcp_json.encode("utf-8"))
        # Return the absolute path the CLI will see (cwd = workspace root).
        logger.info("Wrote MCP config via SFTP")
        return sftp_path

    def _build_cli_command(
        self,
        *,
        prompt: str,
        max_steps: int,
        system_prompt: str | None,
        mcp_config_path: str | None = None,
    ) -> str:
        env_vars = self._build_env_vars()
        is_win = self._shell in ("cmd", "powershell")
        self._win_redirect = False

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
        if system_prompt:
            cli_parts.extend(["--system-prompt", q(system_prompt)])
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
                for block in msg["message"].get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
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

        trace.done = True
        trace.content = "\n".join(content_parts)
        trace.isError = is_error
        trace.messages = messages
        trace.info.update(info)


__all__ = ["ClaudeSDKAgent", "ClaudeSDKConfig"]
