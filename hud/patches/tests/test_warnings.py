"""Regression tests for the ``authlib.jose`` deprecation-warning suppression.

``authlib.deprecate`` runs ``warnings.simplefilter("always", AuthlibDeprecationWarning)``
at import time, prepending an "always" filter to ``warnings.filters``. If
``suppress_known_import_warnings`` installs its "ignore" filter before authlib's
module is imported, authlib's filter lands ahead of ours and wins (the machinery
applies the first matching filter), so ``authlib.jose module is deprecated`` leaks
on every CLI launch. The fix imports ``authlib.deprecate`` first so our filter stays
ahead, and scopes the filter narrowly so nothing else is silenced.

Filters are process-global and the breakage is purely about import order, so the
checks run in a fresh subprocess. The suppression module is loaded in isolation to
keep the interpreter free of unrelated filters that a full ``import hud`` registers.

The warning is emitted directly via ``authlib.deprecate.deprecate`` (what
``authlib.jose`` does internally) rather than by importing ``authlib.jose``, so the
test behaves identically regardless of the resolved authlib version -- the jose
import only emits the warning from ``authlib>=1.7`` -- and therefore regardless of
platform. ``fastmcp`` pins only ``authlib>=1.6.5``, so that version (and thus whether
the warning appears at all) floats per machine; the suppression itself is pure stdlib
``warnings`` and is OS-independent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Loads the suppression helper in a pristine interpreter without importing the rest
# of ``hud`` (which would register unrelated filters). A custom showwarning() runs
# only for warnings that pass the active filters, so the captured list reflects
# exactly what survives the filter -- deterministic across Python versions.
_SETUP = """
import importlib.util
import sys
import warnings

assert "authlib.deprecate" not in sys.modules, "test requires a pristine interpreter"

spec = importlib.util.spec_from_file_location("_hud_patch_warnings", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

shown = []
warnings.showwarning = lambda message, *args, **kwargs: shown.append(str(message))
"""

_SILENCES_JOSE = (
    _SETUP
    + """
module.suppress_known_import_warnings()

from authlib.deprecate import deprecate

# Exactly what authlib.jose raises at import time via authlib.deprecate.deprecate().
deprecate("authlib.jose module is deprecated, please use joserfc instead.", version="2.0.0")

leaked = [m for m in shown if "authlib.jose module is deprecated" in m]
assert not leaked, f"authlib.jose deprecation warning was not suppressed: {leaked}"
print("SUPPRESSED")
"""
)

_LEAVES_OTHERS = (
    _SETUP
    + """
warnings.simplefilter("always")  # baseline: nothing is hidden unless our filter says so
module.suppress_known_import_warnings()

from authlib.deprecate import AuthlibDeprecationWarning

warnings.warn(AuthlibDeprecationWarning("authlib.jose module is deprecated"))
warnings.warn(AuthlibDeprecationWarning("authlib.oauth2 client is deprecated"))
warnings.warn(DeprecationWarning("an unrelated deprecation"))
warnings.warn(UserWarning("an unrelated user warning"))

assert not any("authlib.jose module is deprecated" in m for m in shown), shown
assert any("authlib.oauth2 client is deprecated" in m for m in shown), shown
assert any("an unrelated deprecation" in m for m in shown), shown
assert any("an unrelated user warning" in m for m in shown), shown
print("SPECIFIC")
"""
)


def _run_pristine(code: str) -> subprocess.CompletedProcess[str]:
    warnings_module = Path(__file__).resolve().parents[1] / "warnings.py"
    return subprocess.run(
        [sys.executable, "-c", code, str(warnings_module)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_suppress_known_import_warnings_silences_authlib_jose_deprecation() -> None:
    result = _run_pristine(_SILENCES_JOSE)

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "SUPPRESSED" in result.stdout


def test_suppress_known_import_warnings_leaves_other_warnings_untouched() -> None:
    result = _run_pristine(_LEAVES_OTHERS)

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "SPECIFIC" in result.stdout
