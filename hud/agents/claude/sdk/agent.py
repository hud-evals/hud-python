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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from hud.agents.base import Agent
from hud.agents.types import AgentStep, ClaudeSDKConfig, Usage
from hud.settings import settings
from hud.types import Step

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


@dataclass(slots=True)
class RemoteInvocation:
    """How to run an assembled CLI command on the remote workspace shell.

    ``command`` is what gets exec'd over SSH. When ``script_name`` is set, that
    file must be written (with ``script_body``) before exec'ing ``command``.
    """

    command: str
    script_name: str | None = None
    script_body: str | None = None


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

        completed = await self._ssh.conn.run(full_cmd, check=False)
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""

        logger.info("exit=%s stdout=%d stderr=%d", completed.exit_status, len(stdout), len(stderr))

        if completed.exit_status != 0 and not stdout.strip():
            error = stderr or f"claude CLI exited with status {completed.exit_status}"
            run.trace.status = "error"
            run.trace.extra.update({"exit_status": completed.exit_status, "stderr": stderr})
            run.record(Step(source="system", error=error))
            return

        self._parse_stream_json(run, stdout, stderr)

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

    def _parse_stream_json(self, run: Run, stdout: str, stderr: str) -> None:
        messages: list[dict[str, Any]] = []
        content_parts: list[str] = []
        is_error = False
        info: dict[str, Any] = {}
        cost_usd: float | None = None
        num_turns: int | None = None

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
                for raw_block in msg["message"].get("content", []):
                    if not isinstance(raw_block, dict):
                        continue
                    block = cast("dict[str, Any]", raw_block)
                    if block.get("type") == "text" and block.get("text"):
                        content_parts.append(str(block["text"]))

            elif msg_type == "result":
                is_error = msg.get("is_error", False)
                result_text = msg.get("result")
                if result_text:
                    content_parts.append(result_text)
                info["session_id"] = msg.get("session_id")
                info["duration_ms"] = msg.get("duration_ms")
                info["stop_reason"] = msg.get("stop_reason")
                num_turns = msg.get("num_turns")
                cost_usd = msg.get("total_cost_usd")

        content = "\n".join(content_parts)
        trace = run.trace
        trace.status = "error" if is_error else "completed"
        trace.content = content
        # Raw CLI stream kept locally; a claude-native serializer can take over
        # per-turn fidelity later (the CLI session is its own span vocabulary).
        trace.extra["messages"] = messages
        if stderr:
            trace.extra["stderr"] = stderr

        # The CLI run collapses to one coarse agent step with aggregate usage.
        run.record(
            AgentStep(
                content=content,
                done=True,
                model=self.config.model,
                usage=Usage(cost_usd=cost_usd, llm_call_count=num_turns),
                error=content if is_error else None,
                extra={k: v for k, v in info.items() if v is not None},
            ),
        )


__all__ = ["ClaudeSDKAgent", "ClaudeSDKConfig", "RemoteInvocation", "build_remote_invocation"]
