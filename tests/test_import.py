"""Importing the package: without the CLI, and where unix sockets do not exist.

Each check runs in a fresh interpreter. Where unix sockets do not exist is settled at
import time, by the platform the standard library and the dependencies see at their
first import.

Both routes out of a workspace are served on unix sockets, through a
``socketserver.ThreadingUnixStreamServer`` subclass, and a session runs on a pty.
Neither exists on Windows, so ``import hud`` must work there for the commands that
never go near a workspace. The interpreter imports the package once as the
platform it really is, then again with only the package's own modules dropped,
the platform reported as Windows, and the unix pieces taken away.
"""

from __future__ import annotations

import subprocess
import sys

IMPORT_AS_WINDOWS = """
import socketserver
import sys

import hud

for name in [n for n in sys.modules if n == "hud" or n.startswith("hud.")]:
    del sys.modules[name]

sys.platform = "win32"
sys.modules["pty"] = None  # a Windows interpreter has no pty module to import
socketserver.__dict__.pop("ThreadingUnixStreamServer", None)  # nor this server

import hud

print("IMPORTED")
"""


def test_the_package_imports_where_unix_sockets_do_not_exist() -> None:
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_AS_WINDOWS], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "IMPORTED" in result.stdout


def test_the_sdk_imports_without_the_cli() -> None:
    """``import hud`` loads none of the CLI, so the SDK can ship apart from it."""
    loaded = (
        "import sys, hud; "
        "print([m for m in ('typer', 'hud.cli', 'hud.utils.hud_console') if m in sys.modules])"
    )

    result = subprocess.run(
        [sys.executable, "-c", loaded], capture_output=True, text=True, timeout=120
    )

    assert (result.returncode, result.stdout.strip()) == (0, "[]"), result.stderr
