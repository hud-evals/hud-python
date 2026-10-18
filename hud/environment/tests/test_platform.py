"""Import HUD without initializing Unix-only workspace machinery.

Preload dependencies on the real platform, then reimport HUD in a fresh
interpreter with Windows' missing Unix facilities and unavailable libc.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_IMPORT_AS_WINDOWS = """
import ctypes
import socketserver
import sys

import hud

for name in [n for n in sys.modules if n == "hud" or n.startswith("hud.")]:
    del sys.modules[name]

sys.platform = "win32"
sys.modules["pty"] = None  # a Windows interpreter has no pty module to import
socketserver.__dict__.pop("ThreadingUnixStreamServer", None)  # nor this server

if sys.argv[1] == "fcntl":
    sys.modules["fcntl"] = None
else:
    def unavailable_libc(*args, **kwargs):
        raise OSError("libc is unavailable on this platform")

    ctypes.CDLL = unavailable_libc

import hud
from hud.environment.process_guard import process_connections_supported

assert process_connections_supported() is False
print("IMPORTED")
"""


@pytest.mark.parametrize("unavailable", ["fcntl", "libc"])
def test_the_package_imports_without_unix_dependencies(unavailable: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_AS_WINDOWS, unavailable],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "IMPORTED" in result.stdout
