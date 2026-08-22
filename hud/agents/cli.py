"""Process boundary for JSONL CLI agents."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import shlex
from typing import TYPE_CHECKING

import asyncssh

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.capabilities import SSHClient
    from hud.environment.platform_inference import InferenceBinding
    from hud.eval.runtime import RuntimeConfig

WINDOWS_SHELLS = ("cmd", "powershell")
PROCESS_CLOSE_TIMEOUT_S = 5.0


def require_platform_isolation(ssh: SSHClient, binding: InferenceBinding | None) -> None:
    """Refuse a platform credential binding when the remote shell is not isolated."""
    if binding is not None and ssh.capability.params.get("isolation") != "bwrap":
        raise RuntimeError(
            "platform inference requires a bwrap-isolated workspace; refusing to expose "
            "the workspace-local binding to an unisolated SSH session"
        )


async def resolve_executable(
    ssh: SSHClient,
    command: str,
    managed_paths: dict[str, str],
    runtime_config: RuntimeConfig | None,
) -> str:
    """Resolve a CLI against the live SSH target and its declared runtime config."""
    platform = await _runtime_platform(ssh)
    _validate_runtime_os(runtime_config, platform.partition("-")[0])

    managed = managed_paths.get(platform)
    if managed is not None:
        result = await ssh.run(
            f"test -x {shlex.quote(managed)}",
            check=False,
            encoding=None,
        )
        if result.returncode == 0:
            return managed

    if platform.startswith("windows-"):
        result = await ssh.run(f"where.exe {command}", check=False, encoding=None)
    else:
        result = await ssh.run(f"command -v -- {command}", check=False, encoding=None)
    if result.returncode == 0:
        stdout = _output_text(result.stdout)
        if path := stdout.splitlines()[0].strip():
            return path

    raise RuntimeError(
        f"{command} is unavailable for runtime platform {platform}; "
        "install it in the environment or provide a managed runtime bundle"
    )


async def _runtime_platform(ssh: SSHClient) -> str:
    shell = ssh.capability.params.get("shell", "bash")
    if shell in WINDOWS_SHELLS:
        result = await ssh.run(
            powershell("[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture"),
            check=True,
            encoding=None,
        )
        arch = _output_text(result.stdout).strip().lower()
        return f"windows-{_normalize_arch(arch)}"

    result = await ssh.run(
        "uname -s; uname -m; "
        "if ls /lib/ld-musl-*.so.1 >/dev/null 2>&1; then echo musl; else echo gnu; fi",
        check=True,
        encoding=None,
    )
    lines = _output_text(result.stdout).splitlines()
    if len(lines) != 3:
        raise RuntimeError("SSH runtime platform probe returned an invalid response")
    system, machine, libc = (line.strip().lower() for line in lines)
    os_name = {"darwin": "darwin", "linux": "linux"}.get(system)
    if os_name is None:
        raise RuntimeError(f"unsupported SSH runtime operating system {system!r}")
    platform = f"{os_name}-{_normalize_arch(machine)}"
    return f"{platform}-musl" if os_name == "linux" and libc == "musl" else platform


def _normalize_arch(value: str) -> str:
    normalized = {
        "amd64": "x64",
        "x86_64": "x64",
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


def _validate_runtime_os(runtime_config: RuntimeConfig | None, actual: str) -> None:
    if runtime_config is None or runtime_config.resources is None:
        return
    declared = runtime_config.resources.os
    if declared is None:
        return
    normalized = {
        "darwin": "darwin",
        "linux": "linux",
        "macos": "darwin",
        "windows": "windows",
    }.get(declared.lower())
    if normalized is not None and normalized != actual:
        raise RuntimeError(
            f"runtime_config.resources.os requested {declared!r}, "
            f"but the SSH runtime reports {actual!r}"
        )


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
) -> tuple[int, str]:
    """Stream one remote JSONL process and own its cancellation cleanup."""
    process = await ssh.create_process(command)
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
        if not stderr_task.done():
            stderr_task.cancel()
        await asyncio.gather(stderr_task, return_exceptions=True)
        with contextlib.suppress(OSError, TimeoutError, asyncssh.Error):
            async with asyncio.timeout(PROCESS_CLOSE_TIMEOUT_S):
                await process.wait_closed()
        raise

    if process.returncode is None:
        raise RuntimeError("CLI process closed without an exit status")
    return process.returncode, stderr
