"""Process boundary for JSONL CLI agents."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import shlex
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import asyncssh

from hud.agents.base import Agent, ConfigT_co
from hud.settings import settings
from hud.telemetry.context import get_trace_headers
from hud.utils.gateway import provider_api_key, routes_to_gateway

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from hud.capabilities import Connection, SSHClient
    from hud.eval.run import Run
    from hud.runtime import RuntimeConfig

WINDOWS_SHELLS = ("cmd", "powershell")
PROCESS_CLOSE_TIMEOUT_S = 5.0
_DECLARED_OS = {"darwin": "darwin", "linux": "linux", "macos": "darwin", "windows": "windows"}
_PROVIDER_URLS = {
    "anthropic": ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    "openai": ("OPENAI_BASE_URL", "https://api.openai.com/v1"),
}


class CLIAgent(Agent[ConfigT_co]):
    """An agent that runs a coding CLI in the environment's workspace, where the CLI
    calls its model itself; ``rollout`` relays those calls for it in a sandbox."""

    #: The provider whose API the CLI speaks.
    provider: ClassVar[str]


class ModelEndpoint(NamedTuple):
    url: str
    key: str
    headers: dict[str, str]
    gateway: bool
    connection: Connection | None


def model_endpoint(run: Run, provider: str, *, gateway: bool) -> ModelEndpoint | None:
    """Where the CLI in *run* calls its model.

    Through the run's ``inference`` connection when it has one. Otherwise the CLI
    holds a key, which only a workspace declared unsandboxed (isolation ``none``)
    is given.
    """
    if (connection := run.connections.get("inference")) is not None:
        return ModelEndpoint(connection.client_url, "hud-process-bound", {}, True, connection)
    if routes_to_gateway(provider, gateway=gateway):
        if not settings.api_key:
            raise ValueError("HUD_API_KEY is required for HUD gateway routing")
        endpoint = ModelEndpoint(
            settings.hud_gateway_url, settings.api_key, get_trace_headers(), True, None
        )
    elif key := provider_api_key(provider):
        variable, default = _PROVIDER_URLS[provider]
        endpoint = ModelEndpoint(os.environ.get(variable) or default, key, {}, False, None)
    else:
        return None
    shell = run.binding("ssh")
    if shell.params.get("isolation") != "none":
        raise RuntimeError(
            f"the {shell.name!r} workspace is sandboxed, so it isn't given the agent's model "
            "key: pass the rollout an 'inference' connection, or give the workspace its own "
            "network (network=False or allowed_hosts) on a host with a process guard so the "
            "rollout relays HUD_API_KEY"
        )
    return endpoint


async def resolve_executable(
    ssh: SSHClient,
    command: str,
    managed_paths: dict[str, str],
    runtime_config: RuntimeConfig | None,
) -> str:
    """Resolve a CLI against the live SSH target and its declared runtime config."""
    if ssh.capability.params.get("shell", "bash") in WINDOWS_SHELLS:
        result = await ssh.run(
            powershell("[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture"),
            check=True,
            encoding=None,
        )
        os_name = "windows"
        platform = f"windows-{_normalize_arch(_output_text(result.stdout).strip().lower())}"
    else:
        result = await ssh.run(
            "uname -s; uname -m; "
            "if ls /lib/ld-musl-*.so.1 >/dev/null 2>&1; then echo musl; else echo gnu; fi",
            check=True,
            encoding=None,
        )
        system, machine, libc = (
            line.strip().lower() for line in _output_text(result.stdout).splitlines()
        )
        if system not in ("darwin", "linux"):
            raise RuntimeError(f"unsupported SSH runtime operating system {system!r}")
        os_name = system
        platform = f"{os_name}-{_normalize_arch(machine)}"
        if os_name == "linux" and libc == "musl":
            platform += "-musl"

    declared = runtime_config.resources.os if runtime_config and runtime_config.resources else None
    if declared is not None and _DECLARED_OS.get(declared.lower(), os_name) != os_name:
        raise RuntimeError(
            f"runtime_config.resources.os requested {declared!r}, "
            f"but the SSH runtime reports {os_name!r}"
        )

    managed = managed_paths.get(platform)
    if managed is not None:
        result = await ssh.run(f"test -x {shlex.quote(managed)}", check=False, encoding=None)
        if result.returncode == 0:
            return managed

    lookup = f"where.exe {command}" if os_name == "windows" else f"command -v -- {command}"
    result = await ssh.run(lookup, check=False, encoding=None)
    if result.returncode != 0:
        raise RuntimeError(
            f"{command} is unavailable for runtime platform {platform}; "
            "install it in the environment or provide a managed runtime bundle"
        )
    return _output_text(result.stdout).splitlines()[0].strip()


def _normalize_arch(value: str) -> str:
    normalized = {
        "amd64": "x64",
        "x86_64": "x64",
        "x64": "x64",  # Windows RuntimeInformation.OSArchitecture
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(value)
    if normalized is None:
        raise RuntimeError(f"unsupported SSH runtime architecture {value!r}")
    return normalized


def _output_text(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def powershell(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"


async def run_jsonl(
    ssh: SSHClient,
    command: str,
    consume: Callable[[str], None],
    *,
    input_text: str | None = None,
    connections: Sequence[Connection] = (),
) -> tuple[int, str]:
    """Stream one remote JSONL process and own its cancellation cleanup."""
    process = await ssh.create_process(command, connections=connections)
    stderr_task = asyncio.create_task(process.stderr.read())
    try:
        if input_text is not None:
            process.stdin.write(input_text.encode())
            await process.stdin.drain()
            process.stdin.write_eof()
        while line := await process.stdout.readline():
            consume(line.decode(errors="replace"))
        await process.wait_closed()
        stderr = (await stderr_task).decode(errors="replace")
    except BaseException:
        process.close()
        stderr_task.cancel()
        await asyncio.gather(stderr_task, return_exceptions=True)
        with contextlib.suppress(OSError, TimeoutError, asyncssh.Error):
            async with asyncio.timeout(PROCESS_CLOSE_TIMEOUT_S):
                await process.wait_closed()
        raise

    if process.returncode is None:
        raise RuntimeError("CLI process closed without an exit status")
    return process.returncode, stderr
