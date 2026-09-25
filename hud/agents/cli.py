"""Process boundary for JSONL CLI agents."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import shlex
from typing import TYPE_CHECKING

import asyncssh

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from hud.capabilities import Connection, SSHClient
    from hud.eval.runtime import RuntimeConfig

WINDOWS_SHELLS = ("cmd", "powershell")
#: What a CLI is given as its credential on a process-bound connection; the relay
#: replaces it with the scoped credential, so it never needs to be secret.
PROCESS_BOUND_CREDENTIAL = "hud-process-bound"
PROCESS_CLOSE_TIMEOUT_S = 5.0
_DECLARED_OS = {"darwin": "darwin", "linux": "linux", "macos": "darwin", "windows": "windows"}


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
