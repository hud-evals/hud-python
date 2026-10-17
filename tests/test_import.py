"""Importing the package: where unix sockets do not exist, and what warnings it leaves visible.

Both are settled at import time, by the platform the standard library and the
dependencies see at their first import and by the order warning filters are
installed in, so each check runs in a fresh interpreter.

Both routes out of a workspace are served on unix sockets, through a
``socketserver.ThreadingUnixStreamServer`` subclass, and a session runs on a pty.
Neither exists on Windows, so ``import hud`` must work there for the commands that
never go near a workspace. The interpreter imports the package once as the
platform it really is, then again with only the package's own modules dropped,
the platform reported as Windows, and the unix pieces taken away.

``fastmcp`` imports ``authlib.jose``, which emits an ``AuthlibDeprecationWarning``
on every CLI launch that the user can never act on. ``import hud`` hides that one
warning and no other, even with every warning shown; the warning is emitted the
way ``authlib.jose`` emits it, so the check holds whichever authlib is installed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

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

SHOW_WARNING = """
import sys
import warnings

shown = []
warnings.simplefilter("always")
warnings.showwarning = lambda message, *args, **kwargs: shown.append(str(message))

import hud
from authlib.deprecate import AuthlibDeprecationWarning, deprecate

category, message = sys.argv[1:]
if category == "jose":
    deprecate(message, version="2.0.0")
else:
    warnings.warn({"authlib": AuthlibDeprecationWarning, "deprecation": DeprecationWarning,
                   "user": UserWarning}[category](message))
print(shown)
"""


def run(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=120
    )


def test_the_package_imports_where_unix_sockets_do_not_exist() -> None:
    result = run(IMPORT_AS_WINDOWS)

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "IMPORTED" in result.stdout


@pytest.mark.parametrize(
    ("category", "message", "shown"),
    [
        ("jose", "authlib.jose module is deprecated, please use joserfc instead.", False),
        ("authlib", "authlib.oauth2 client is deprecated", True),
        ("deprecation", "an unrelated deprecation", True),
        ("user", "an unrelated user warning", True),
    ],
)
def test_importing_hud_hides_only_the_authlib_jose_deprecation(
    category: str, message: str, shown: bool
) -> None:
    result = run(SHOW_WARNING, category, message)

    assert result.returncode == 0, result.stderr
    assert (message in result.stdout) is shown
