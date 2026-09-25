"""ClaudeCLIAgent — runs ``claude`` CLI over SSH inside the env workspace.

SSH-execs the ``claude`` CLI on the remote workspace so all built-in tools
(Bash, Read, Write, Edit, Glob, Grep) operate on the env's filesystem.
MCP capabilities from the manifest are written as MCP server config so the
CLI can call env-hosted MCP tools too.
"""

from __future__ import annotations

import json
import logging
import shlex
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, cast

import asyncssh

from hud.agents.base import Agent
from hud.agents.cli import (
    WINDOWS_SHELLS,
    powershell,
    powershell_quote,
    resolve_executable,
    run_jsonl,
)
from hud.agents.types import ClaudeCLIConfig
from hud.settings import settings
from hud.telemetry.context import get_trace_headers
from hud.utils.gateway import routes_to_gateway
from hud.utils.time import now_iso

from . import computer_mcp
from .events import ClaudeEvents

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


class ClaudeCLIAgent(Agent[ClaudeCLIConfig]):
    """Runs ``claude`` CLI over SSH inside the env workspace.

    Stateless w.r.t. the env: driven by ``await agent(run)``. SSH is opened
    live off the run. Environment MCP bindings are used directly; computer MCP
    servers are bridged over the run's SSH connection.
    """

    config_cls = ClaudeCLIConfig

    async def __call__(self, run: Run) -> None:
        ssh = cast("SSHClient", await run.client.open("ssh"))
        manifest = run.client.manifest
        assert manifest is not None
        shell = ssh.capability.params.get("shell", "bash")
        windows = shell in WINDOWS_SHELLS
        executable = await resolve_executable(
            ssh, "claude", _MANAGED_CLAUDE_PATHS, run.runtime_config
        )
        rfb_count = sum(cap.protocol.partition("/")[0] == "rfb" for cap in manifest.bindings)

        async with AsyncExitStack() as resources:
            mcp_servers: dict[str, dict[str, Any]] = {}
            for cap in manifest.bindings:
                family = cap.protocol.partition("/")[0]
                if family == "mcp":
                    name = cap.name
                    transport = "http" if cap.params["transport"] == "streamable-http" else "sse"
                    server: dict[str, Any] = {"type": transport, "url": cap.url}
                    if token := cap.params.get("auth_token"):
                        server["headers"] = {"Authorization": f"Bearer {token}"}
                elif family == "rfb":
                    name = "computer-use" if rfb_count == 1 else f"computer-use-{cap.name}"
                    server = await resources.enter_async_context(
                        computer_mcp.bridge_computer_mcp(
                            ssh,
                            run.client.binding(cap.name),
                            self.config.screenshot_encoding,
                            shell=shell,
                        )
                    )
                else:
                    continue
                if name in mcp_servers:
                    raise RuntimeError(f"duplicate MCP server name {name!r}")
                mcp_servers[name] = server

            env: dict[str, str] = {
                "ANTHROPIC_MODEL": self.config.model,
                "ANTHROPIC_SMALL_FAST_MODEL": self.config.model,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                "IS_SANDBOX": "1",
            }
            if routes_to_gateway("anthropic", gateway=self.config.gateway):
                if not settings.api_key:
                    raise ValueError("HUD_API_KEY is required for HUD gateway routing")
                env["ANTHROPIC_BASE_URL"] = settings.hud_gateway_url
                env["ANTHROPIC_API_KEY"] = settings.api_key
                env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
                env["DISABLE_AUTO_COMPACT"] = "1"
                # Alias every model tier so background requests never bypass the gateway.
                env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = self.config.model
                env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = self.config.model
                env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = self.config.model
                env["CLAUDE_CODE_SUBAGENT_MODEL"] = self.config.model
                if trace_headers := get_trace_headers():
                    env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(
                        f"{name}: {value}" for name, value in trace_headers.items()
                    )
            elif settings.anthropic_api_key:
                env["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

            args = [
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
            if mcp_servers:
                args.extend(["--mcp-config", MCP_CONFIG_PATH])

            if windows:
                script = ";".join(
                    [
                        *(f"$env:{key}={powershell_quote(value)}" for key, value in env.items()),
                        f"Get-Content -Raw -Encoding UTF8 {powershell_quote(INPUT_PATH)}"
                        f" | & {powershell_quote(executable)} "
                        f"{' '.join(powershell_quote(arg) for arg in args)}",
                        "exit $LASTEXITCODE",
                    ]
                )
                command = powershell(script)
            else:
                env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
                cli = " ".join(shlex.quote(arg) for arg in [executable, *args])
                command = f'export PATH="$HOME/.local/bin:$PATH"; {env_prefix} {cli}'

            input_text = (
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": run.prompt_text}],
                        },
                    }
                )
                + "\n"
            )
            files: list[str] = []
            try:
                if mcp_servers:
                    await ssh.write_text(
                        MCP_CONFIG_PATH, json.dumps({"mcpServers": mcp_servers}, indent=2)
                    )
                    files.append(MCP_CONFIG_PATH)
                if windows:
                    await ssh.write_text(INPUT_PATH, input_text)
                    await ssh.write_text(RUN_SCRIPT_PATH, f"@echo off\r\n{command}\r\n")
                    files += [INPUT_PATH, RUN_SCRIPT_PATH]
                    command = f"cmd /c {RUN_SCRIPT_PATH}"

                logger.info("SSH exec claude CLI (%d chars)", len(command))
                events = ClaudeEvents(run, started_at=now_iso())
                returncode, stderr = await run_jsonl(
                    ssh,
                    command,
                    events.consume,
                    input_text=None if windows else input_text,
                )
                logger.info("exit=%s stderr=%d", returncode, len(stderr))
                events.finish(returncode=returncode, stderr=stderr)
            finally:
                if files:
                    if windows:
                        cleanup = f"cmd /c del /f /q {' '.join(files)} 2>nul"
                    else:
                        cleanup = "rm -f -- " + " ".join(shlex.quote(path) for path in files)
                    try:
                        await ssh.run(cleanup, check=False)
                    except (OSError, asyncssh.Error):
                        logger.warning("Failed to remove Claude CLI runtime files")


__all__ = ["ClaudeCLIAgent"]
