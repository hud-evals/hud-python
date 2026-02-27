from __future__ import annotations

from hud.cli import version


def test_version_does_not_crash():
    version()
