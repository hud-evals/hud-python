"""macOS Seatbelt (sandbox-exec) probing and SBPL profile generation.

Generates deny-default SBPL profiles for HUD Workspace isolation on Darwin:
scoped file reads/writes and optional localhost proxy ports. No global
``(allow file-read*)``.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MACOS_PATH_TO_SEATBELT_EXECUTABLE: str = "/usr/bin/sandbox-exec"

# Deny-default shell/process base. System paths are granted explicitly below;
# do not add a global ``(allow file-read*)``.
_BASE_POLICY = """\
(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow file-write-data
 (require-all
  (path "/dev/null")
  (vnode-type CHARACTER-DEVICE)))
(allow sysctl-read
 (sysctl-name "hw.activecpu")
 (sysctl-name "hw.byteorder")
 (sysctl-name "hw.cacheconfig")
 (sysctl-name "hw.cpufamily")
 (sysctl-name "hw.cputype")
 (sysctl-name "hw.logicalcpu_max")
 (sysctl-name "hw.machine")
 (sysctl-name "hw.memsize")
 (sysctl-name "hw.ncpu")
 (sysctl-name "hw.pagesize")
 (sysctl-name "hw.physicalcpu_max")
 (sysctl-name "kern.argmax")
 (sysctl-name "kern.hostname")
 (sysctl-name "kern.osproductversion")
 (sysctl-name "kern.osrelease")
 (sysctl-name "kern.ostype")
 (sysctl-name "kern.osversion")
 (sysctl-name "kern.version")
 (sysctl-name "vm.loadavg")
 (sysctl-name-prefix "hw.optional.arm.")
 (sysctl-name-prefix "hw.perflevel")
 (sysctl-name-prefix "kern.proc.pid.")
 (sysctl-name-prefix "net.routetable."))
(allow sysctl-write (sysctl-name "kern.grade_cputype"))
(allow mach-lookup
 (global-name "com.apple.system.opendirectoryd.libinfo"))
(allow ipc-posix-sem)
(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-read* file-write*
 (require-all
  (regex #"^/dev/ttys[0-9]+")
  (extension "com.apple.sandbox.pty")))
(allow file-ioctl (regex #"^/dev/ttys[0-9]+"))
(allow file-read* (subpath "/usr"))
(allow file-read* (subpath "/bin"))
(allow file-read* (subpath "/sbin"))
(allow file-read* (subpath "/System"))
(allow file-read* (subpath "/Library/Frameworks"))
(allow file-read* (subpath "/dev"))
(allow file-read* (literal "/private/var/db/timezone"))
(allow file-read* (subpath "/etc"))
(allow file-read* (subpath "/private/etc"))
(allow file-read* (subpath "/private/var/select"))
(allow file-read* (subpath "/var/select"))
(allow file-read-data (literal "/"))
"""

_TMPDIR_WRITE_PATHS = (
    "/tmp",
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
    #: ``127.0.0.2:5432``). Seatbelt's ``localhost:*`` filter matches only
    #: ``127.0.0.1`` / ``::1``, not other 127.0.0.0/8 addresses.
    proxy_loopback_endpoints: tuple[tuple[str, int], ...] = ()
    allow_all_network: bool = False


_seatbelt_usable: Seatbelt | Literal[False] | None = None


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
    parts = [_BASE_POLICY]

    for index in range(len(inputs.writable_roots)):
        param = f"WRITABLE_ROOT_{index}"
        parts.append(
            f'(allow file-read* file-write* file-write-create (subpath (param "{param}")))'
        )

    for index in range(len(inputs.readable_roots)):
        param = f"READABLE_ROOT_{index}"
        parts.append(f'(allow file-read* (subpath (param "{param}")))')

    if inputs.allow_tmpdir_write:
        for path in _TMPDIR_WRITE_PATHS:
            # SBPL requires double-quoted string literals; single quotes (from !r)
            # are rejected by sandbox-exec on macOS 15 (Darwin 25).
            parts.append(
                f'(allow file-read* file-write* file-write-create (subpath "{path}"))'
            )

    if inputs.allow_all_network:
        parts.append("(allow network*)")
    elif inputs.proxy_loopback_ports:
        # Loopback outbound for proxy/peer ports. ``localhost:*`` covers
        # 127.0.0.1/::1; other 127.0.0.0/8 hosts need explicit ``remote ip``
        # rules (Seatbelt has no CIDR form).
        parts.append('(allow network-outbound (remote ip "localhost:*"))')
        for port in inputs.proxy_loopback_ports:
            parts.append(f'(allow network-outbound (remote ip "127.0.0.1:{port}"))')
            parts.append(f'(allow network-outbound (remote ip "[::1]:{port}"))')
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
