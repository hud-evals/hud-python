"""Sanity checks for the public API contract tests themselves."""

from __future__ import annotations

from importlib import import_module

import pytest

import hud
from hud.tests.public_api.test_v5_surface_imports import (
    DEEP_MODULES,
    DEEP_SURFACE,
    DOCS_EXAMPLES_DEEP_SURFACE,
    DOCS_EXAMPLES_LAZY_PUBLIC_EXPORTS,
    DOCS_EXAMPLES_PUBLIC_SURFACE,
    ENVIRONMENT_DEEP_SURFACE,
    ENVIRONMENT_LAZY_PUBLIC_EXPORTS,
    ENVIRONMENT_PUBLIC_SURFACE,
    LAZY_PUBLIC_EXPORTS,
    PUBLIC_SURFACE,
    TOP_LEVEL_DOCS_EXAMPLES_SURFACE,
    TOP_LEVEL_ENVIRONMENT_SURFACE,
    TOP_LEVEL_EXPORTS,
)


def test_contract_tables_are_not_empty() -> None:
    assert TOP_LEVEL_EXPORTS
    assert PUBLIC_SURFACE
    assert DEEP_SURFACE
    assert DEEP_MODULES
    assert LAZY_PUBLIC_EXPORTS
    assert TOP_LEVEL_DOCS_EXAMPLES_SURFACE
    assert TOP_LEVEL_ENVIRONMENT_SURFACE
    assert DOCS_EXAMPLES_PUBLIC_SURFACE
    assert ENVIRONMENT_PUBLIC_SURFACE
    assert DOCS_EXAMPLES_DEEP_SURFACE
    assert ENVIRONMENT_DEEP_SURFACE
    assert DOCS_EXAMPLES_LAZY_PUBLIC_EXPORTS
    assert ENVIRONMENT_LAZY_PUBLIC_EXPORTS


def test_top_level_evidence_sources_cover_exact_surface() -> None:
    assert set(TOP_LEVEL_EXPORTS) == (
        set(TOP_LEVEL_DOCS_EXAMPLES_SURFACE) | set(TOP_LEVEL_ENVIRONMENT_SURFACE)
    )


def test_package_version_is_exposed_for_install_checks() -> None:
    assert isinstance(hud.__version__, str)
    assert hud.__version__


@pytest.mark.parametrize(("module_name", "symbols"), sorted(LAZY_PUBLIC_EXPORTS.items()))
def test_lazy_public_exports_resolve(module_name: str, symbols: tuple[str, ...]) -> None:
    module = import_module(module_name)
    missing = [symbol for symbol in symbols if not hasattr(module, symbol)]

    assert not missing, f"{module_name} missing lazy public exports: {missing}"
