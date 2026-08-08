"""macOS Seatbelt (sandbox-exec) probing and SBPL profile generation.

Generates deny-default SBPL profiles for HUD Workspace isolation on Darwin:
scoped file reads/writes and optional localhost proxy ports. No global
``(allow file-read*)``.

Static policy text lives in sibling ``.sbpl`` files next to this module.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MACOS_PATH_TO_SEATBELT_EXECUTABLE: str = "/usr/bin/sandbox-exec"

_POLICY_DIR = Path(__file__).resolve().parent

_TMPDIR_WRITE_PATHS = (
    "/tmp",  # noqa: S108 — macOS sandbox path literal, not a temp-file API
    "/private/tmp",
    "/var/folders",
    "/private/var/folders",
)


@dataclass(frozen=True, slots=True)
class Seatbelt:
    path: str


@dataclass(frozen=True, slots=True)
class SeatbeltPolicyInputs:
    writable_roots: tuple[Path, ...]
    readable_roots: tuple[Path, ...]
    allow_tmpdir_write: bool = True
    proxy_loopback_ports: tuple[int, ...] = ()
    #: ``(host, port)`` loopback endpoints beyond ``127.0.0.1`` (e.g. peer
    #: ``127.0.0.2:5432``). Numeric ``127.0.0.0/8`` peers need an explicit
    #: ``remote ip`` rule; the ``localhost:<port>`` filter only matches
    #: ``127.0.0.1`` / ``::1`` / the ``localhost`` hostname, not other 127.x.
    proxy_loopback_endpoints: tuple[tuple[str, int], ...] = ()
    allow_all_network: bool = False


_seatbelt_usable: Seatbelt | Literal[False] | None = None


@cache
def _load_policy(name: str) -> str:
    """Load a sibling ``.sbpl`` policy file (cached)."""
    path = _POLICY_DIR / name
    return path.read_text(encoding="utf-8").rstrip()


def policy_params(inputs: SeatbeltPolicyInputs) -> dict[str, str]:
    """Map writable/readable roots to ``-D`` parameter names."""
    params: dict[str, str] = {}
    for index, root in enumerate(inputs.writable_roots):
        params[f"WRITABLE_ROOT_{index}"] = str(root.resolve())
    for index, root in enumerate(inputs.readable_roots):
        params[f"READABLE_ROOT_{index}"] = str(root.resolve())
    return params


def generate_seatbelt_profile(inputs: SeatbeltPolicyInputs) -> str:
    """Build an inline SBPL profile for ``sandbox-exec -p``."""
    parts = [_load_policy("seatbelt_base_policy.sbpl")]

    for index in range(len(inputs.writable_roots)):
        param = f"WRITABLE_ROOT_{index}"
        parts.append(
            f'(allow file-read* file-write* file-write-create (subpath (param "{param}")))'
        )

    for index in range(len(inputs.readable_roots)):
        param = f"READABLE_ROOT_{index}"
        parts.append(f'(allow file-read* (subpath (param "{param}")))')

    if inputs.allow_tmpdir_write:
        # SBPL requires double-quoted string literals; single quotes (from !r)
        # are rejected by sandbox-exec on macOS 15 (Darwin 25).
        parts.extend(
            f'(allow file-read* file-write* file-write-create (subpath "{path}"))'
            for path in _TMPDIR_WRITE_PATHS
        )

    if inputs.allow_all_network:
        parts.append("(allow network*)")
    elif inputs.proxy_loopback_ports:
        # Restricted network: per-port / peer endpoints only (no localhost:*).
        parts.append(_load_policy("seatbelt_network_policy.sbpl"))
        for port in inputs.proxy_loopback_ports:
            parts.append(f'(allow network-outbound (remote ip "127.0.0.1:{port}"))')
            parts.append(f'(allow network-outbound (remote ip "[::1]:{port}"))')
            # Clients that dial the hostname ``localhost`` need this form;
            # Seatbelt treats it separately from numeric 127.0.0.1 / ::1.
            parts.append(f'(allow network-outbound (remote ip "localhost:{port}"))')
        for host, port in inputs.proxy_loopback_endpoints:
            parts.append(f'(allow network-outbound (remote ip "{host}:{port}"))')

    return "\n".join(parts)


def usable_seatbelt() -> Seatbelt | None:
    """Return a working sandbox-exec path on Darwin, probing once per process."""
    global _seatbelt_usable
    if sys.platform != "darwin":
        return None
    if isinstance(_seatbelt_usable, Seatbelt):
        return _seatbelt_usable
    if _seatbelt_usable is False:
        return None

    probe = subprocess.run(
        [
            MACOS_PATH_TO_SEATBELT_EXECUTABLE,
            "-p",
            (
                "(version 1)(deny default)(allow process-exec)(allow process-fork)"
                '(allow file-read*)(allow file-write-data (literal "/dev/null"))'
            ),
            "--",
            "/usr/bin/true",
        ],
        capture_output=True,
        timeout=15,
        check=False,
    )
    if probe.returncode == 0:
        _seatbelt_usable = Seatbelt(MACOS_PATH_TO_SEATBELT_EXECUTABLE)
        return _seatbelt_usable

    _seatbelt_usable = False
    return None


def seatbelt_argv(
    command: Sequence[str],
    *,
    profile: str,
    params: Mapping[str, str] | None = None,
) -> list[str]:
    """Build a ``sandbox-exec`` argv wrapping *command*."""
    argv: list[str] = [MACOS_PATH_TO_SEATBELT_EXECUTABLE, "-p", profile]
    if params:
        for key, value in params.items():
            argv.append(f"-D{key}={value}")
    argv.append("--")
    argv.extend(command)
    return argv
