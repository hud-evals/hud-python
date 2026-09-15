"""Local Environment identity: a source tree, or the file that defined a live env."""

from __future__ import annotations

import ast
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self


@dataclass(frozen=True)
class EnvironmentNameReference:
    """One ``Environment(...)`` constructor call found in project source.

    ``name`` is the literal string passed (positionally or as ``name=``);
    None when the call relies on the default name or passes a non-literal.
    """

    file: Path
    line: int
    text: str
    name: str | None


@dataclass(frozen=True)
class EnvironmentSource:
    """A local Environment source tree rooted at a filesystem directory."""

    root: Path

    SOURCE_EXCLUDE_DIRS: ClassVar[set[str]] = {
        ".git",
        ".venv",
        "dist",
        "build",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }

    @classmethod
    def open(cls, directory: str | Path = ".") -> Self:
        p = Path(directory).expanduser().resolve()
        if p.is_file():
            p = p.parent
        return cls(p)

    @property
    def dockerfile(self) -> Path | None:
        hud_dockerfile = self.root / "Dockerfile.hud"
        if hud_dockerfile.exists():
            return hud_dockerfile
        dockerfile = self.root / "Dockerfile"
        if dockerfile.exists():
            return dockerfile
        return None

    @property
    def is_environment(self) -> bool:
        return (
            self.root.is_dir()
            and self.dockerfile is not None
            and (self.root / "pyproject.toml").exists()
        )

    def dockerfile_instructions(self) -> list[str]:
        """Logical Dockerfile instructions, joining ``\\`` line continuations."""
        dockerfile = self.dockerfile
        if dockerfile is None:
            return []
        return _dockerfile_instructions(dockerfile.read_text(encoding="utf-8"))

    def environment_name_references(self) -> list[EnvironmentNameReference]:
        """Find ``Environment(...)`` constructor calls in project source.

        Captures the name passed positionally (``Environment("x")``) or as a
        keyword (``Environment(name="x")``); calls without a literal name are
        reported with ``name=None`` so callers can demand an explicit one.
        """
        references: list[EnvironmentNameReference] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(name for name in dirnames if name not in self.SOURCE_EXCLUDE_DIRS)
            py_files = (Path(dirpath) / name for name in sorted(filenames) if name.endswith(".py"))
            for py_file in py_files:
                try:
                    source = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(source)
                except (OSError, SyntaxError):
                    continue
                lines = source.splitlines()
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    callee = node.func
                    callee_name = (
                        callee.id
                        if isinstance(callee, ast.Name)
                        else callee.attr
                        if isinstance(callee, ast.Attribute)
                        else None
                    )
                    if callee_name != "Environment":
                        continue
                    references.append(
                        EnvironmentNameReference(
                            file=py_file,
                            line=node.lineno,
                            text=(
                                lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                            ),
                            name=_environment_call_name(node),
                        )
                    )
        return references

    def served_environment_module(self) -> str | None:
        for tokens in _dockerfile_command_tokens(self.dockerfile_instructions()):
            spec = _hud_serve_spec(tokens)
            if spec is not None:
                return spec.partition(":")[0]
        return None

    def served_environment_name(self) -> str | None:
        module = self.served_environment_module()
        if module is None:
            return None

        module_path = Path(module) if module.endswith(".py") else Path(*module.split("."))
        served_file = (self.root / module_path).with_suffix(".py").resolve()
        names = {
            ref.name
            for ref in self.environment_name_references()
            if ref.file.resolve() == served_file and ref.name is not None
        }
        return next(iter(names)) if len(names) == 1 else None


def environment_file(env: Any) -> Path:
    """The ``.py`` file that defined this live env's templates.

    ``Taskset.from_file`` already imported that module; the bound
    ``task._env`` is the same object ``Taskset.run()`` uses when no
    ``runtime=`` is passed. The CLI still serves it in a child process.
    """
    files = {
        Path(factory.func.__code__.co_filename).resolve()
        for factory in env.tasks.values()
        if getattr(getattr(factory, "func", None), "__code__", None) is not None
    }
    if len(files) != 1:
        raise ValueError(
            "local spawn needs a bound Environment from a Python source "
            "(``@env.template`` rows). Portable JSON rows require --remote, "
            "--runtime hud, or a tcp:// url"
        )
    return files.pop()


def _dockerfile_instructions(content: str) -> list[str]:
    instructions: list[str] = []
    buffer = ""
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            buffer += line[:-1].strip() + " "
            continue
        buffer += line
        instructions.append(buffer.strip())
        buffer = ""
    if buffer.strip():
        instructions.append(buffer.strip())
    return instructions


def _command_tokens(remainder: str) -> list[str]:
    """Tokens of a CMD/ENTRYPOINT body in either exec (JSON) or shell form."""
    if remainder.startswith("["):
        try:
            parsed = json.loads(remainder)
        except json.JSONDecodeError:
            return []
        return [str(token) for token in parsed] if isinstance(parsed, list) else []
    try:
        return shlex.split(remainder)
    except ValueError:
        return remainder.split()


def _dockerfile_command_tokens(instructions: list[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    for instruction in instructions:
        keyword, _, remainder = instruction.partition(" ")
        if keyword.upper() not in {"CMD", "ENTRYPOINT"}:
            continue
        tokens = _command_tokens(remainder.strip())
        if tokens:
            commands.append(tokens)
    return commands


def _hud_serve_spec(tokens: list[str]) -> str | None:
    """The serve target from a ``hud serve <spec>`` token list.

    Returns the explicit ``module[:attr]`` spec, ``"env"`` when ``hud serve`` is
    invoked with no target (the runtime default), or ``None`` when the tokens
    contain no ``hud serve`` invocation.
    """
    for index, token in enumerate(tokens):
        if Path(token).name != "hud":
            continue
        rest = tokens[index + 1 :]
        if not rest or rest[0] != "serve":
            continue
        target = rest[1] if len(rest) > 1 else None
        if target is None or target.startswith("-"):
            return "env"
        return target
    return None


def _environment_call_name(node: ast.Call) -> str | None:
    """The literal name an ``Environment(...)`` call passes, if any."""
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    for keyword in node.keywords:
        if keyword.arg == "name":
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                return keyword.value.value
            return None
    return None


__all__ = [
    "EnvironmentNameReference",
    "EnvironmentSource",
    "environment_file",
]
