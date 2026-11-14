"""ClaudeCLIAgent — runs ``claude`` CLI over SSH inside the env workspace."""

from __future__ import annotations

import json
import logging
import shlex
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, cast

import asyncssh

from hud.agents.cli import (
    WINDOWS_SHELLS,
    CLIAgent,
    model_endpoint,
    powershell,
    powershell_quote,
    resolve_executable,
    run_jsonl,
)
from hud.agents.types import ClaudeCLIConfig
from hud.utils.time import now_iso

from . import computer_mcp
from .events import ClaudeEvents

if TYPE_CHECKING:
    from hud.capabilities.ssh import SSHClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

INPUT_PATH = ".hud_input.jsonl"
MCP_CONFIG_PATH = ".hud_mcp_config.json"
RUN_SCRIPT_PATH = ".hud_run.bat"

_MANAGED_CLAUDE_PATHS = {
    "linux-x64": "/usr/local/lib/agents/claude/linux-x64/claude",
    "linux-x64-musl": "/usr/local/lib/agents/claude/linux-x64-musl/claude",
}


class ClaudeCLIAgent(CLIAgent[ClaudeCLIConfig]):
    """Runs ``claude`` CLI over SSH inside the env workspace.

    The env's MCP bindings become the CLI's MCP servers; RFB computers are bridged in over SSH.
    """

    config_cls = ClaudeCLIConfig
    provider = "anthropic"

    async def __call__(self, run: Run) -> None:
        endpoint = model_endpoint(run, self.provider, gateway=self.config.gateway)
        ssh = cast("SSHClient", await run.client.open("ssh"))
        manifest = run.client.manifest
        assert manifest is not None
        shell = ssh.capability.params.get("shell", "bash")
        windows = shell in WINDOWS_SHELLS
        connection = endpoint.connection if endpoint is not None else None
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
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                "IS_SANDBOX": "1",
            }
            if endpoint is not None:
                env["ANTHROPIC_BASE_URL"] = endpoint.url
                env["ANTHROPIC_API_KEY"] = endpoint.key
                if endpoint.headers:
                    env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(
                        f"{name}: {value}" for name, value in endpoint.headers.items()
                    )
                if endpoint.gateway:
                    env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
            # PowerShell's environment ignores case.
            taken = {key.casefold() for key in env}
            if endpoint is not None:
                taken.add("claude_config_dir")
            if clash := sorted(key for key in self.config.env if key.casefold() in taken):
                raise ValueError(f"the agent sets {', '.join(clash)} itself")
            env.update(self.config.env)

            args = [
                "--verbose",
                "--input-format=stream-json",
                "--output-format=stream-json",
                "--print",
                f"--permission-mode={self.config.permission_mode}",
            ]
            if self.config.max_steps > 0:
                args.append(f"--max-turns={self.config.max_steps}")
            if self.config.reasoning_effort is not None:
                args.extend(["--effort", self.config.reasoning_effort])
            if self.config.system_prompt:
                args.extend(["--system-prompt", self.config.system_prompt])
            for tool in self.config.allowed_tools:
                args.extend(["--allowedTools", tool])
            for tool in self.config.disallowed_tools:
                args.extend(["--disallowedTools", tool])
            if mcp_servers:
                args.extend(["--mcp-config", MCP_CONFIG_PATH])

            # Injected credentials get a throwaway CLAUDE_CONFIG_DIR, so the host user's login,
            # settings, memory and MCP servers stay out of the run; otherwise they apply.
            if windows:
                invocation = (
                    f"Get-Content -Raw -Encoding UTF8 {powershell_quote(INPUT_PATH)}"
                    f" | & {powershell_quote(executable)} "
                    f"{' '.join(powershell_quote(arg) for arg in args)}"
                )
                statements = [
                    *(f"$env:{key}={powershell_quote(value)}" for key, value in env.items()),
                    f"{invocation}; $hudExitCode=$LASTEXITCODE",
                ]
                if endpoint is not None:
                    statements = [
                        "$claudeConfig=Join-Path ([System.IO.Path]::GetTempPath()) "
                        "('hud-claude-' + [System.Guid]::NewGuid())",
                        "New-Item -ItemType Directory -Force -Path $claudeConfig | Out-Null",
                        "$env:CLAUDE_CONFIG_DIR=$claudeConfig",
                        *statements[:-1],
                        f"try {{ {statements[-1]} }} "
                        "finally { Remove-Item -Recurse -Force $claudeConfig }",
                    ]
                command = powershell(";".join([*statements, "exit $hudExitCode"]))
            else:
                env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
                cli = " ".join(shlex.quote(arg) for arg in [executable, *args])
                statements: list[str] = []
                if endpoint is not None:
                    statements.append(
                        'claude_config=$(mktemp -d "${TMPDIR:-/tmp}/hud-claude.XXXXXX") || exit 1'
                    )
                    # Under exec no trap survives; the sandbox holding a guarded connection is
                    # discarded after the rollout.
                    if connection is None:
                        statements.append("trap 'rm -rf -- \"$claude_config\"' EXIT")
                    statements.append('export CLAUDE_CONFIG_DIR="$claude_config"')
                statements.append('export PATH="$HOME/.local/bin:$PATH"')
                # A guarded connection is bound to the launched process, so exec the CLI itself.
                exec_ = "exec env " if connection is not None else ""
                statements.append(f"{exec_}{env_prefix} {cli}")
                command = "; ".join(statements)

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
                    connections=(connection,) if connection is not None else (),
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
