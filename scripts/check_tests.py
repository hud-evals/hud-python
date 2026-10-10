"""Fail when a test reaches into hud internals.

Tests drive the SDK through its public boundaries and fake only what lies outside
it (see "Testing" in AGENTS.md). This flags, in test modules outside the harness
(``tests/harness``, the one place allowed to configure the SDK):

- patching a hud name: ``monkeypatch.setattr``, ``patch``, ``patch.object`` or
  ``mocker.patch`` aimed at a ``"hud..."`` path or at anything imported from hud;
- importing a private name or module from hud;
- touching a private attribute of anything other than ``self`` or ``cls``.

Files that predate the rule are listed in ``check_tests_baseline.json`` with the
number of findings they still hold. That number may only fall: a file above its
baseline fails, and so does one below it until the baseline is lowered to match.
Every other file must have none.

Usage: uv run python scripts/check_tests.py [path ...]
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [ROOT / "tests", *sorted(ROOT.glob("environments/*/tests"))]
HARNESS = ROOT / "tests" / "harness"
BASELINE = Path(__file__).with_name("check_tests_baseline.json")
PATCHERS = {"setattr", "delattr", "patch"}
PATCH_METHODS = {"object", "dict", "multiple"}


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _is_patch(func: ast.expr) -> bool:
    """``setattr``, ``patch`` and ``monkeypatch.setattr`` forms, and ``patch.object`` forms."""
    if isinstance(func, ast.Name):
        return func.id in PATCHERS
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr in PATCHERS:
        return True
    owner = func.value
    owner_name = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", "")
    return func.attr in PATCH_METHODS and owner_name == "patch"


def _root(node: ast.expr) -> ast.expr:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node


class Checker(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hud_names: set[str] = set()
        self.problems: list[str] = []

    def report(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", 0)
        self.problems.append(f"{self.path.relative_to(ROOT)}:{line}: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "hud" or alias.name.startswith("hud."):
                self.hud_names.add((alias.asname or alias.name).split(".")[0])
                if any(_private(part) for part in alias.name.split(".")):
                    self.report(node, f"imports private module {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module == "hud" or module.startswith("hud."):
            private_module = any(_private(part) for part in module.split("."))
            for alias in node.names:
                self.hud_names.add(alias.asname or alias.name)
                if private_module or _private(alias.name):
                    self.report(node, f"imports private {module}.{alias.name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_patch(node.func) and node.args:
            target = node.args[0]
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                if target.value == "hud" or target.value.startswith("hud."):
                    self.report(node, f"patches {target.value}")
            else:
                root = _root(target)
                if isinstance(root, ast.Name) and root.id in self.hud_names:
                    self.report(node, f"patches {ast.unparse(target)}, which comes from hud")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _private(node.attr):
            root = _root(node)
            if not (isinstance(root, ast.Name) and root.id in {"self", "cls"}):
                self.report(node, f"touches private attribute {ast.unparse(node)}")
        self.generic_visit(node)


def check(path: Path) -> list[str]:
    checker = Checker(path)
    checker.visit(ast.parse(path.read_text("utf-8"), filename=str(path)))
    return checker.problems


def main(argv: list[str]) -> int:
    roots = [Path(arg).resolve() for arg in argv] or DEFAULT_PATHS
    files = sorted(
        file
        for root in roots
        for file in ([root] if root.is_file() else root.rglob("*.py"))
        if HARNESS not in file.parents and "tasks" not in file.relative_to(ROOT).parts
    )
    baseline: dict[str, int] = json.loads(BASELINE.read_text("utf-8")) if BASELINE.exists() else {}
    findings = {str(file.relative_to(ROOT)): check(file) for file in files}
    failed = False
    for name, problems in findings.items():
        allowed = baseline.get(name, 0)
        if len(problems) > allowed:
            failed = True
            for problem in problems:
                print(problem)
            print(f"{name}: {len(problems)} reach(es) into hud internals, {allowed} allowed\n")
        elif len(problems) < allowed:
            failed = True
            print(f"{name}: {len(problems)} reach(es) left; lower its baseline from {allowed}\n")
    counts = Counter({name: len(problems) for name, problems in findings.items()})
    if failed:
        print("See Testing in AGENTS.md; baselines live in scripts/check_tests_baseline.json.")
    elif baseline:
        print(
            f"{sum(counts[name] for name in baseline)} known reach(es) in {len(baseline)} file(s)"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
