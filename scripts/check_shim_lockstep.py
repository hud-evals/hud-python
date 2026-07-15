"""Fail the release when the hud-python shim falls out of lockstep with hud.

Four values carry the release: the version in pyproject.toml, the version in
shim/pyproject.toml, the floor/cap of the shim's 'hud' dependency, and
__version__ in hud/version.py. Publishing them out of sync half-breaks a
release — PyPI rejects the duplicate shim upload only after 'hud' has already
gone out. Run by release.yml before anything is built or published.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main() -> None:
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    shim = tomllib.loads((ROOT / "shim" / "pyproject.toml").read_text())["project"]
    version = root["version"]

    problems: list[str] = []

    if shim["version"] != version:
        problems.append(f"shim/pyproject.toml version {shim['version']} != {version}")

    code = re.search(r'__version__ = "([^"]+)"', (ROOT / "hud" / "version.py").read_text())
    if code is None or code.group(1) != version:
        found = code.group(1) if code else "<missing>"
        problems.append(f"hud/version.py __version__ {found} != {version}")

    # The shim must pull the matching hud release, capped at the next minor so
    # a stale shim can never drag in a hud it was not published against.
    major, minor = version.split(".")[:2]
    expected_dep = f"hud>={version},<{major}.{int(minor) + 1}"
    deps = [d.replace(" ", "") for d in shim["dependencies"]]
    if deps != [expected_dep]:
        problems.append(f"shim dependencies {deps} != ['{expected_dep}']")

    if shim["requires-python"] != root["requires-python"]:
        problems.append(
            f"shim requires-python {shim['requires-python']!r} != {root['requires-python']!r}"
        )

    if problems:
        sys.exit("shim lockstep check failed:\n  " + "\n  ".join(problems))
    print(f"lockstep OK: hud=={version}, shim depends on '{expected_dep}'")


if __name__ == "__main__":
    main()
