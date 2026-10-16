"""Regression test for importing the package where unix sockets do not exist.

Both routes out of a workspace are served on unix sockets and a session runs on a pty,
and each reached for its platform at module scope: a
``socketserver.ThreadingUnixStreamServer`` subclass in :mod:`hud.environment.egress`,
and a top-level ``import pty`` in :mod:`hud.environment.namespace`. Neither exists on
Windows, so ``import hud`` raised there before any of it was used, taking down the
commands that never go near a workspace with it. Both now sit behind the platform
guard the rest of the package already applies.

The breakage is at import time, so the check runs in a fresh interpreter. That
interpreter imports the package once as the platform it really is, because the
standard library and the dependencies settle their own platform at their first import
and will not be told otherwise afterwards, and then imports it again with only the
package's own modules dropped, the platform reported as Windows, and the unix pieces
taken away.
"""

from __future__ import annotations

import subprocess
import sys

_IMPORT_AS_WINDOWS = """
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
        [sys.executable, "-c", _IMPORT_AS_WINDOWS],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "IMPORTED" in result.stdout
