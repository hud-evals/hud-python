"""CLI agents that run non-interactive coding-agent CLIs over SSH."""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from asyncssh.sftp import SFTPError

from hud.agents.base import Agent
from hud.agents.types import (
    AgentStep,
    AiderConfig,
    CLIConfig,
    CodexConfig,
    GrokBuildConfig,
    MiniSweAgentConfig,
    OpenCodeConfig,
    Terminus2Config,
)
from hud.settings import settings
from hud.types import Step

if TYPE_CHECKING:
    from hud.capabilities import Capability, SSHClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

WINDOWS_SHELLS = ("cmd", "powershell")
PROMPT_PATH = ".hud_prompt.txt"
MCP_CONFIG_PATH = ".hud_mcp_config.json"
TERMINUS2_SCRIPT_PATH = ".hud_terminus2.py"
TERMINUS2_RESULT_PATH = ".hud_terminus2_logs/trajectory.json"
TERMINUS2_SCRIPT = r"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.paths import TrialPaths


class LocalEnvironment(BaseEnvironment):
    def __init__(self, workdir: Path, logs_dir: Path) -> None:
        self.workdir = workdir
        self.trial_paths = TrialPaths(trial_dir=logs_dir)
        self.trial_paths.mkdir()
        self.default_user = None
        self.session_id = "hud"
        self.logger = logging.getLogger(__name__)

    def type(self) -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    def _validate_definition(self) -> None:
        return None

    async def start(self, force_build: bool) -> None:
        return None

    async def stop(self, delete: bool) -> None:
        return None

    async def prepare_logs_for_host(self) -> None:
        return None

    async def upload_file(self, source_path, target_path) -> None:
        shutil.copy(source_path, target_path)

    async def upload_dir(self, source_dir, target_dir) -> None:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    async def download_file(self, source_path, target_path) -> None:
        shutil.copy(source_path, target_path)

    async def download_dir(self, source_dir, target_dir) -> None:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = user
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd or str(self.workdir),
                env={**os.environ, **(env or {})},
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr="Command timed out", return_code=124)
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )


async def main() -> None:
    workdir = Path(os.environ.get("AGENT_WORKDIR") or os.getcwd())
    logs_dir = Path(".hud_terminus2_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    instruction = Path(".hud_prompt.txt").read_text()
    subprocess.run(
        ["tmux", "kill-session", "-t", "terminus-2"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    api_base = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )
    agent = Terminus2(
        logs_dir=logs_dir,
        model_name=os.environ.get("HUD_TERMINUS2_MODEL", "openai/gpt-5.4"),
        api_base=api_base,
        max_turns=int(os.environ.get("HUD_TERMINUS2_MAX_TURNS", "10")),
    )
    env = LocalEnvironment(workdir=workdir, logs_dir=logs_dir)
    await agent.setup(env)
    await agent.run(instruction, env, AgentContext())


asyncio.run(main())
"""


@dataclass(slots=True)
class RemoteInvocation:
    command: str
    script_name: str | None = None
    script_body: str | None = None


def build_remote_invocation(shell: str, run_cmd: str) -> RemoteInvocation:
    if shell in WINDOWS_SHELLS:
        return RemoteInvocation(
            command="cmd /c .hud_run.bat",
            script_name=".hud_run.bat",
            script_body=f"@echo off\r\n{run_cmd}\r\n",
        )
    return RemoteInvocation(command=run_cmd)


class CLIAgent(Agent):
    """Runs a configured non-interactive CLI inside the env workspace."""

    config: CLIConfig

    def __init__(self, config: CLIConfig | None = None) -> None:
        self.config = config or CLIConfig()
        self._ssh: SSHClient | None = None
        self._shell = "bash"
        self._mcp_servers: dict[str, dict[str, Any]] = {}

    async def __call__(self, run: Run) -> None:
        self._mcp_servers = {}
        manifest = run.client.manifest
        bindings = manifest.bindings if manifest is not None else []

        await self._open_ssh(run, bindings)
        await self._collect_mcp_servers(run, bindings)
        await self._exec(run, prompt=run.prompt_text)

    async def _open_ssh(self, run: Run, bindings: list[Capability]) -> None:
        if "ssh" not in {c.protocol.split("/", 1)[0] for c in bindings}:
            raise RuntimeError(f"{self.config.model_name} requires an SSH capability")

        self._ssh = cast("SSHClient", await run.client.open("ssh"))
        self._shell = self._ssh.capability.params.get("shell", "bash")

    async def _collect_mcp_servers(self, run: Run, bindings: list[Capability]) -> None:
        _ = run
        for cap in bindings:
            if cap.protocol.split("/", 1)[0] != "mcp":
                continue
            token = cap.params.get("auth_token")
            transport = "http" if cap.url.startswith("http") else "sse"
            server_config: dict[str, Any] = {"type": transport, "url": cap.url}
            if token:
                server_config["headers"] = {"Authorization": f"Bearer {token}"}
            self._mcp_servers[cap.name] = server_config

    async def _exec(self, run: Run, *, prompt: str) -> None:
        if not self.config.command:
            raise ValueError("CLIConfig.command is required")
        assert self._ssh is not None

        await self._write_file(PROMPT_PATH, prompt.encode("utf-8"))
        mcp_config_path = await self._write_mcp_config()
        run_cmd = self._build_cli_command(prompt=prompt, mcp_config_path=mcp_config_path)
        invocation = self._build_remote_invocation(run_cmd)

        if invocation.script_name is not None:
            assert invocation.script_body is not None
            await self._write_file(invocation.script_name, invocation.script_body.encode("utf-8"))

        logger.info("SSH exec %s CLI (%d chars)", self.config.model_name, len(invocation.command))
        completed = await self._ssh.conn.run(invocation.command, check=False)
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        await self._record_cli_result(
            run,
            stdout=stdout,
            stderr=stderr,
            exit_status=completed.exit_status if completed.exit_status is not None else 1,
        )

    def _build_remote_invocation(self, run_cmd: str) -> RemoteInvocation:
        return build_remote_invocation(self._shell, run_cmd)

    async def _record_cli_result(
        self,
        run: Run,
        *,
        stdout: str,
        stderr: str,
        exit_status: int,
    ) -> None:
        result_content = await self._read_result_file()
        content = (result_content if result_content is not None else stdout).strip()

        trace = run.trace
        trace.status = "error" if exit_status != 0 else "completed"
        trace.content = content
        trace.extra.update(
            {
                "agent": self.config.model_name,
                "command": self.config.command,
                "args": self.config.args,
                "exit_status": exit_status,
            }
        )
        if stderr:
            trace.extra["stderr"] = stderr

        error = None
        if exit_status != 0:
            error = (
                stderr or content or f"{self.config.model_name} exited with status {exit_status}"
            )
            run.record(Step(source="system", error=error))

        run.record(
            AgentStep(
                content=content,
                done=True,
                model=self.config.model,
                error=error,
                extra={"agent": self.config.model_name, "exit_status": exit_status},
            )
        )

    async def _write_file(self, path: str, data: bytes) -> None:
        assert self._ssh is not None
        async with self._ssh.conn.start_sftp_client() as sftp, sftp.open(path, "wb") as f:
            await f.write(data)

    async def _read_result_file(self) -> str | None:
        if self.config.result_file is None:
            return None
        assert self._ssh is not None
        try:
            async with (
                self._ssh.conn.start_sftp_client() as sftp,
                sftp.open(self.config.result_file, "rb") as f,
            ):
                data = cast("bytes | str", await f.read())
        except (OSError, SFTPError):
            return None
        if isinstance(data, str):
            return data
        return data.decode("utf-8", errors="replace")

    async def _write_mcp_config(self) -> str | None:
        if not self.config.mcp_config or not self._mcp_servers:
            return None
        await self._write_file(
            MCP_CONFIG_PATH,
            json.dumps({"mcpServers": self._mcp_servers}, indent=2).encode("utf-8"),
        )
        return MCP_CONFIG_PATH

    def _build_env_vars(self) -> dict[str, str]:
        env: dict[str, str] = {}

        if self.config.use_hud_gateway and settings.api_key:
            env["OPENAI_API_KEY"] = settings.api_key
            env["OPENAI_API_BASE"] = settings.hud_gateway_url
            env["OPENAI_BASE_URL"] = settings.hud_gateway_url
        if settings.openai_api_key:
            env.setdefault("OPENAI_API_KEY", settings.openai_api_key)
            env.setdefault("CODEX_API_KEY", settings.openai_api_key)
        if settings.anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        if settings.gemini_api_key:
            env["GEMINI_API_KEY"] = settings.gemini_api_key
        if settings.xai_api_key:
            env["XAI_API_KEY"] = settings.xai_api_key

        env.update(self.config.extra_env)
        return env

    def _build_cli_command(self, *, prompt: str, mcp_config_path: str | None) -> str:
        args = [self.config.command, *self.config.args]
        expanded = [
            arg.format(
                prompt=prompt,
                prompt_file=PROMPT_PATH,
                model=self.config.model,
                max_steps=self.config.max_steps,
                mcp_config=mcp_config_path or "",
            )
            for arg in args
        ]

        cmd = " ".join(shlex.quote(arg) for arg in expanded)
        if self.config.stdin:
            cmd = f"{cmd} < {shlex.quote(PROMPT_PATH)}"

        env_vars = self._build_env_vars()
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_vars.items())
        command = f"{env_prefix} {cmd}" if env_prefix else cmd
        if self.config.install_check:
            return f"{self.config.install_check} && {command}"
        return command


class OpenCodeAgent(CLIAgent):
    def __init__(self, config: OpenCodeConfig | None = None) -> None:
        super().__init__(config or OpenCodeConfig())


class AiderAgent(CLIAgent):
    def __init__(self, config: AiderConfig | None = None) -> None:
        super().__init__(config or AiderConfig())


class CodexAgent(CLIAgent):
    def __init__(self, config: CodexConfig | None = None) -> None:
        super().__init__(config or CodexConfig())


class GrokBuildAgent(CLIAgent):
    def __init__(self, config: GrokBuildConfig | None = None) -> None:
        super().__init__(config or GrokBuildConfig())

    def _build_env_vars(self) -> dict[str, str]:
        env = super()._build_env_vars()
        if env.get("XAI_API_KEY"):
            env.setdefault("HOME", "/tmp/hud-grok-home")  # noqa: S108
        return env


class MiniSweAgent(CLIAgent):
    def __init__(self, config: MiniSweAgentConfig | None = None) -> None:
        super().__init__(config or MiniSweAgentConfig())


class Terminus2Agent(CLIAgent):
    def __init__(self, config: Terminus2Config | None = None) -> None:
        super().__init__(config or Terminus2Config())

    async def _exec(self, run: Run, *, prompt: str) -> None:
        await self._write_file(TERMINUS2_SCRIPT_PATH, TERMINUS2_SCRIPT.encode("utf-8"))
        await super()._exec(run, prompt=prompt)

    def _build_env_vars(self) -> dict[str, str]:
        env = super()._build_env_vars()
        config = cast("Terminus2Config", self.config)
        env["HUD_TERMINUS2_MODEL"] = config.model
        env["HUD_TERMINUS2_MAX_TURNS"] = str(config.max_steps)
        return env


__all__ = [
    "TERMINUS2_RESULT_PATH",
    "AiderAgent",
    "AiderConfig",
    "CLIAgent",
    "CLIConfig",
    "CodexAgent",
    "CodexConfig",
    "GrokBuildAgent",
    "GrokBuildConfig",
    "MiniSweAgent",
    "MiniSweAgentConfig",
    "OpenCodeAgent",
    "OpenCodeConfig",
    "RemoteInvocation",
    "Terminus2Agent",
    "Terminus2Config",
    "build_remote_invocation",
]
