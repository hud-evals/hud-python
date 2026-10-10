"""Fixtures for scenarios that generate projects and install this checkout's SDK into them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.conftest import REAL_ENVIRONMENT

from .projects import ROOT, Uv

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="session")
def uv() -> Uv:
    return Uv(REAL_ENVIRONMENT)


@pytest.fixture(scope="session")
def sdk_wheel(uv: Uv, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """This checkout's SDK as a wheel, built once per test worker."""
    out = tmp_path_factory.mktemp("sdk-wheel")
    uv("build", "--wheel", "--no-create-gitignore", "--out-dir", str(out), str(ROOT), cwd=out)
    (wheel,) = out.glob("hud-*.whl")
    return wheel
