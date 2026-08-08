"""Linux bubblewrap probing for Workspace isolation.

Detects a working ``bwrap`` launch mode (direct vs staged ``unshare``) once per
process. Argv construction and sandbox lifecycle stay on :class:`Workspace`;
this module is the substrate probe, parallel to :mod:`hud.environment.seatbelt`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

LOGGER = logging.getLogger("hud.environment.bwrap")


@dataclass(slots=True, frozen=True)
class Bubblewrap:
    path: str
    pid_unshare: str | None = None


# Set once the first probe runs (avoid per-instance work).
_bwrap_usable: Bubblewrap | Literal[False] | None = None


def usable_bwrap() -> Bubblewrap | None:
    """A working bubblewrap launch mode for this substrate, if one exists."""
    global _bwrap_usable
    if isinstance(_bwrap_usable, Bubblewrap):
        return _bwrap_usable
    if _bwrap_usable is False:
        return None

    path = shutil.which("bwrap")
    if path is None:
        return None
    probe_binary = shutil.which("true")
    if probe_binary is None:
        return None

    direct = Bubblewrap(path)
    launches = [
        (
            direct,
            [
                path,
                "--unshare-user",
                "--unshare-pid",
                "--ro-bind",
                "/",
                "/",
                "--proc",
                "/proc",
                "--",
                probe_binary,
            ],
        )
    ]
    if unshare := shutil.which("unshare"):
        staged = Bubblewrap(path, pid_unshare=unshare)
        launches.append(
            (
                staged,
                [
                    unshare,
                    "--kill-child=KILL",
                    "--pid",
                    "--mount-proc",
                    path,
                    "--unshare-user",
                    "--ro-bind",
                    "/",
                    "/",
                    "--",
                    probe_binary,
                ],
            )
        )

    failure = "unknown error"
    for launch, argv in launches:
        try:
            probe = subprocess.run(
                argv,
                capture_output=True,
                timeout=15,
                check=False,
            )
            if probe.returncode == 0:
                _bwrap_usable = launch
                return launch
            failure = probe.stderr.decode("utf-8", "replace").strip()[:120]
        except (OSError, subprocess.SubprocessError):
            continue

    _bwrap_usable = False
    LOGGER.warning(
        "bwrap is installed but cannot create an isolated process namespace (%s); "
        "sessions will run WITHOUT isolation.",
        failure,
    )
    return None
