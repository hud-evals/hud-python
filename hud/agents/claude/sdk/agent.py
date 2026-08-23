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
from hud.telemetry.context import get_current_trace_id
from hud.utils.time import now_iso

from . import computer_mcp
from .events import ClaudeEvents

if TYPE_CHECKING:
    from hud.capabilities import SSHClient
    from hud.eval.run import InferenceConnection, Run

logger = logging.getLogger(__name__)

INPUT_PATH = ".hud_input.jsonl"
MCP_CONFIG_PATH = ".hud_mcp_config.json"
RUN_SCRIPT_PATH = ".hud_run.bat"

_MANAGED_CLAUDE_PATHS = {
    "linux-x64": "/usr/local/lib/agents/claude/linux-x64/claude",
    "linux-x64-musl": "/usr/local/lib/agents/claude/linux-x64-musl/claude",
}


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
                inference=run.inference,
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
        inference: InferenceConnection | None = None,
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
            inference=inference,
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

    def _build_env_vars(self, inference: InferenceConnection | None = None) -> dict[str, str]:
        env: dict[str, str] = {}
        use_hud_gateway = self.config.use_hud_gateway
        if use_hud_gateway is None:
            use_hud_gateway = inference is not None or settings.api_key is not None

        if use_hud_gateway:
            if inference is not None:
                base_url = inference.base_url
                api_key = inference.credential
            elif settings.api_key:
                base_url = settings.hud_gateway_url
                api_key = settings.api_key
            else:
                raise ValueError("HUD_API_KEY is required for HUD gateway routing")
            env["ANTHROPIC_BASE_URL"] = base_url
            env["ANTHROPIC_API_KEY"] = api_key
            env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
            env["DISABLE_AUTO_COMPACT"] = "1"
            if inference is None and (trace_id := get_current_trace_id()):
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
        mcp_json = json.dumps({"mcpServers": mcp_servers}, indent=2)
        path = MCP_CONFIG_PATH
        await ssh.write_text(path, mcp_json)
        logger.info("Wrote MCP config")
        return path

    def _build_cli_command(
        self,
        *,
        shell: str,
        mcp_config_path: str | None = None,
        executable: str = "claude",
        inference: InferenceConnection | None = None,
    ) -> str:
        env_vars = self._build_env_vars(inference)
        is_win = shell in WINDOWS_SHELLS
        base_args: list[str] = [
            executable,
            "--verbose",
            "--input-format=stream-json",
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

        if is_win:
            script = ";".join(
                [
                    *(f"$env:{key}={powershell_quote(value)}" for key, value in env_vars.items()),
                    f"Get-Content -Raw -Encoding UTF8 {powershell_quote(INPUT_PATH)}"
                    f" | & {powershell_quote(executable)} "
                    f"{' '.join(powershell_quote(arg) for arg in base_args[1:])}",
                    "exit $LASTEXITCODE",
                ]
            )
            return powershell(script)

        cli_parts = [shlex.quote(a) for a in base_args]
        cli_cmd = " ".join(cli_parts)
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items())
        return f'export PATH="$HOME/.local/bin:$PATH"; {env_prefix} {cli_cmd}'


__all__ = ["ClaudeCLIAgent"]
