"""Helpers for consumer-driven HUD import contract tests."""

from __future__ import annotations

import ast
import re
import textwrap
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, order=True)
class ImportContract:
    """A single import that a public consumer expects to resolve."""

    source: str
    module: str
    names: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        if self.names:
            return f"{self.source}: from {self.module} import {', '.join(self.names)}"
        return f"{self.source}: import {self.module}"


PYTHON_FENCE_RE = re.compile(r"```(?:python|py)[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
FROM_IMPORT_RE = re.compile(r"from\s+(hud(?:\.[A-Za-z_]\w*)*)\s+import\s+(.+)")
IMPORT_RE = re.compile(r"import\s+(.+)")


def _is_hud_module(module_name: str) -> bool:
    return module_name == "hud" or module_name.startswith("hud.")


def _contracts_from_ast(code: str, source: str) -> list[ImportContract]:
    tree = ast.parse(code)
    contracts: list[ImportContract] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            contracts.extend(
                ImportContract(source=source, module=alias.name)
                for alias in node.names
                if _is_hud_module(alias.name)
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and _is_hud_module(node.module)
        ):
            names = tuple(alias.name for alias in node.names if alias.name != "*")
            if names:
                contracts.append(ImportContract(source=source, module=node.module, names=names))

    return contracts


def _logical_import_lines(code: str) -> list[str]:
    lines: list[str] = []
    pending: str | None = None

    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if pending is not None:
            pending = f"{pending} {line}"
            if ")" in line:
                lines.append(pending)
                pending = None
            continue

        if line.startswith("from hud") and "(" in line and ")" not in line:
            pending = line
            continue

        if line.startswith(("from hud", "import hud")):
            lines.append(line)

    if pending is not None:
        lines.append(pending)

    return lines


def _parse_imported_names(names_part: str) -> tuple[str, ...]:
    names_part = names_part.split("#", 1)[0].strip().strip("()")
    names: list[str] = []

    for raw_name in names_part.split(","):
        name = raw_name.strip()
        if not name or name == "...":
            continue
        name = re.split(r"\s+as\s+", name, maxsplit=1)[0].strip()
        if re.fullmatch(r"[A-Za-z_]\w*", name):
            names.append(name)

    return tuple(names)


def _contracts_from_import_lines(code: str, source: str) -> list[ImportContract]:
    contracts: list[ImportContract] = []

    for line in _logical_import_lines(code):
        from_match = FROM_IMPORT_RE.match(line)
        if from_match:
            names = _parse_imported_names(from_match.group(2))
            if names:
                contracts.append(
                    ImportContract(source=source, module=from_match.group(1), names=names)
                )
            continue

        import_match = IMPORT_RE.match(line)
        if not import_match:
            continue

        for raw_module in import_match.group(1).split(","):
            module_name = re.split(r"\s+as\s+", raw_module.strip(), maxsplit=1)[0].strip()
            if _is_hud_module(module_name):
                contracts.append(ImportContract(source=source, module=module_name))

    return contracts


def discover_hud_imports_from_code(code: str, source: str) -> list[ImportContract]:
    """Discover HUD imports from complete Python or partial documentation snippets."""
    code = textwrap.dedent(code)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return _contracts_from_ast(code, source)
    except SyntaxError:
        return _contracts_from_import_lines(code, source)


def discover_hud_imports_from_path(path: Path, repo_root: Path) -> list[ImportContract]:
    try:
        rel_path = path.relative_to(repo_root).as_posix()
    except ValueError:
        rel_path = path.as_posix()
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".py":
        return discover_hud_imports_from_code(text, rel_path)

    contracts: list[ImportContract] = []
    for index, code in enumerate(PYTHON_FENCE_RE.findall(text), start=1):
        contracts.extend(discover_hud_imports_from_code(code, f"{rel_path}#python-{index}"))
    return contracts


def dedupe_contracts(contracts: list[ImportContract]) -> tuple[ImportContract, ...]:
    return tuple(sorted(set(contracts)))
