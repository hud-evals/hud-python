"""Docs and examples are public API consumers.

Every HUD import shown in README, docs, and examples should keep resolving.
This catches drift that a hand-maintained symbol table can miss.
"""

from __future__ import annotations

import ast
import textwrap
from importlib import import_module
from pathlib import Path

import pytest

from hud.tests.public_api._import_contracts import (
    PYTHON_FENCE_RE,
    ImportContract,
    dedupe_contracts,
    discover_hud_imports_from_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_EXAMPLES_PATHS = (
    REPO_ROOT / "README.md",
    *sorted(path for path in (REPO_ROOT / "docs").rglob("*.mdx") if "internal" not in path.parts),
    *sorted(path for path in (REPO_ROOT / "docs").rglob("*.md") if "internal" not in path.parts),
    *sorted((REPO_ROOT / "examples").rglob("*.md")),
    *sorted((REPO_ROOT / "examples").rglob("*.py")),
)


def _discover_docs_examples_imports() -> tuple[ImportContract, ...]:
    contracts: list[ImportContract] = []
    for path in DOCS_EXAMPLES_PATHS:
        if path.exists():
            contracts.extend(discover_hud_imports_from_path(path, REPO_ROOT))
    return dedupe_contracts(contracts)


DOCS_EXAMPLES_IMPORTS = _discover_docs_examples_imports()


def _discover_docs_examples_python_snippets() -> tuple[tuple[str, str, int], ...]:
    snippets: list[tuple[str, str, int]] = []
    for path in DOCS_EXAMPLES_PATHS:
        if not path.exists():
            continue

        rel_path = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")

        if path.suffix == ".py":
            snippets.append((rel_path, text, 0))
            continue

        for index, code in enumerate(PYTHON_FENCE_RE.findall(text), start=1):
            snippets.append(
                (
                    f"{rel_path}#python-{index}",
                    textwrap.dedent(code),
                    ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )
            )

    return tuple(snippets)


DOCS_EXAMPLES_PYTHON_SNIPPETS = _discover_docs_examples_python_snippets()


def test_docs_examples_import_contract_is_not_empty() -> None:
    assert DOCS_EXAMPLES_IMPORTS


def test_docs_examples_python_snippet_contract_is_not_empty() -> None:
    assert DOCS_EXAMPLES_PYTHON_SNIPPETS


@pytest.mark.parametrize(
    "contract",
    DOCS_EXAMPLES_IMPORTS,
    ids=[contract.id for contract in DOCS_EXAMPLES_IMPORTS],
)
def test_docs_examples_hud_imports_resolve(contract: ImportContract) -> None:
    module = import_module(contract.module)
    missing = [name for name in contract.names if not hasattr(module, name)]

    assert not missing, f"{contract.source}: {contract.module} missing {missing}"


@pytest.mark.parametrize(
    ("source", "code", "flags"),
    DOCS_EXAMPLES_PYTHON_SNIPPETS,
    ids=[source for source, _, _ in DOCS_EXAMPLES_PYTHON_SNIPPETS],
)
def test_docs_examples_python_snippets_compile(source: str, code: str, flags: int) -> None:
    compile(code, source, "exec", flags=flags)
