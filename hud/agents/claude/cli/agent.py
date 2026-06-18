"""ClaudeCLIAgent — runs the claude CLI over SSH inside the env workspace.

SSH-execs the ``claude`` CLI on the remote workspace so all built-in tools
(Bash, Read, Write, Edit, Glob, Grep) operate on the env's filesystem.
MCP capabilities from the manifest are written as MCP server config so the
CLI can call env-hosted MCP tools too.

Inspired by harbor-framework/harbor's claude CLI adapter.
"""

from __future__ import annotations

import json
import logging
import shlex
from typing import TYPE_CHECKING, Any, cast

from hud.agents.cli import CLIAgent
from hud.agents.cli.agent import (
    RemoteInvocation,
)
from hud.agents.cli.agent import (
    build_remote_invocation as build_cli_remote_invocation,
)
from hud.agents.types import AgentStep, ClaudeCLIConfig, Usage
from hud.settings import settings
from hud.types import Step

if TYPE_CHECKING:
    from hud.capabilities import Capability, RFBClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

WINDOWS_SHELLS = ("cmd", "powershell")
#: Bare ``claude`` install bootstrap for POSIX shells (no-op when already present).
_POSIX_INSTALL_CHECK = (
    "command -v claude >/dev/null 2>&1 || "
    "{ curl -fsSL https://claude.ai/install.sh | bash -s -- 2>/dev/null; "
    'export PATH="$HOME/.local/bin:$PATH"; }'
)


def build_remote_invocation(shell: str, run_cmd: str) -> RemoteInvocation:
    """Build the remote exec command for ``run_cmd`` under the given login shell.

    Windows shells can't take the assembled command inline — ``cmd.exe`` mangles
    the quotes — so it is written to a batch file and invoked through ``cmd /c``.
    A bare ``.hud_run.bat`` is rejected as an unknown command, and silently fails
    to run under a PowerShell default shell, so ``cmd /c`` is required for both.
    POSIX shells take the command inline, prefixed with a one-shot install check.
    """
    invocation = build_cli_remote_invocation(shell, run_cmd)
    if invocation.script_name is not None:
        return invocation
    return RemoteInvocation(command=f"{_POSIX_INSTALL_CHECK} && {run_cmd}")


class ClaudeCLIAgent(CLIAgent):
    """Runs ``claude`` CLI over SSH inside the env workspace.

    Stateless w.r.t. the env: driven by ``await agent(run)``. SSH and RFB are
    opened live off the run (we
    drive them); MCP servers are read as raw bindings and written into the CLI's
    MCP config (the CLI connects to them itself).
    """

    def __init__(self, config: ClaudeCLIConfig | None = None) -> None:
        super().__init__(config or ClaudeCLIConfig())

    async def _collect_mcp_servers(self, run: Run, bindings: list[Capability]) -> None:
        await super()._collect_mcp_servers(run, bindings)
        for cap in bindings:
            if cap.protocol.split("/", 1)[0] == "rfb":
                from hud.agents.claude.cli.computer_mcp import serve_computer_mcp

                rfb = cast("RFBClient", await run.client.open("rfb"))
                port = await serve_computer_mcp(rfb)
                self._mcp_servers["computer-use"] = {
                    "type": "http",
                    "url": f"http://127.0.0.1:{port}/mcp",
                }

    def _build_remote_invocation(self, run_cmd: str) -> RemoteInvocation:
        return build_remote_invocation(self._shell, run_cmd)

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

    def _build_cli_command(
        self,
        *,
        prompt: str,
        mcp_config_path: str | None = None,
    ) -> str:
        config = cast("ClaudeCLIConfig", self.config)
        env_vars = self._build_env_vars()
        is_win = self._shell in WINDOWS_SHELLS
        self._win_redirect = False

        # Raw args list (no shell quoting) — used directly for Windows Python launcher.
        base_args: list[str] = [
            "claude",
            "--verbose",
            "--output-format=stream-json",
            "--print",
            f"--permission-mode={config.permission_mode}",
        ]
        if config.max_steps > 0:
            base_args.append(f"--max-turns={config.max_steps}")
        if config.system_prompt:
            base_args.extend(["--system-prompt", config.system_prompt])
        for tool in config.allowed_tools:
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

    async def _record_cli_result(
        self,
        run: Run,
        *,
        stdout: str,
        stderr: str,
        exit_status: int,
    ) -> None:
        logger.info("exit=%s stdout=%d stderr=%d", exit_status, len(stdout), len(stderr))

        if exit_status != 0 and not stdout.strip():
            error = stderr or f"claude CLI exited with status {exit_status}"
            run.trace.status = "error"
            run.trace.extra.update({"exit_status": exit_status, "stderr": stderr})
            run.record(Step(source="system", error=error))
            return

        self._parse_stream_json(run, stdout, stderr)

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


__all__ = ["ClaudeCLIAgent", "ClaudeCLIConfig", "RemoteInvocation", "build_remote_invocation"]
