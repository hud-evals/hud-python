"""Filesystem-backed Environment source and build identity."""

from __future__ import annotations

import ast
import json
import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    message: str
    file: str | None = None
    hint: str | None = None


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
        dockerfile = self.dockerfile
        if dockerfile is None:
            return None
        try:
            content = dockerfile.read_text(encoding="utf-8")
        except OSError:
            return None

        for tokens in _dockerfile_command_tokens(content):
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

    @staticmethod
    def local_source(source: str | Path) -> Path:
        """Locate environment code beside an authored task source without importing it."""
        path = Path(source).resolve()
        if path.is_dir():
            return path
        if path.suffix != ".py":
            return path.parent
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "Environment")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "Environment")
            )
            for node in ast.walk(tree)
        ):
            return path
        sibling = path.parent / "env.py"
        return sibling if sibling.is_file() else path.parent

    def relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def validate(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        issues.extend(self.validate_pyproject_references())
        issues.extend(self.validate_dockerfile())
        return issues

    def validate_pyproject_references(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        pyproject_path = self.root / "pyproject.toml"
        if not pyproject_path.exists():
            return issues

        try:
            with pyproject_path.open("rb") as file:
                data = tomllib.load(file)
        except tomllib.TOMLDecodeError as exc:
            return [
                ValidationIssue(
                    severity="error",
                    message=f"Failed to parse pyproject.toml: {exc}",
                    file="pyproject.toml",
                )
            ]

        project = data.get("project", {})
        if isinstance(project, dict):
            issues.extend(self._validate_project_references(project))

        tool = data.get("tool", {})
        if isinstance(tool, dict):
            hatch = tool.get("hatch", {})
            if isinstance(hatch, dict):
                build = hatch.get("build", {})
                if isinstance(build, dict):
                    targets = build.get("targets", {})
                    if isinstance(targets, dict):
                        issues.extend(self._validate_hatch_includes(targets))

        return issues

    def validate_dockerfile(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        dockerfile = self.dockerfile
        if dockerfile is None:
            return issues

        try:
            content = dockerfile.read_text(encoding="utf-8")
        except OSError:
            return issues

        copied_files: set[str] = set()
        has_install_before_full_copy = False
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("COPY "):
                parts = line.split()
                if len(parts) >= 3:
                    src_idx = 1
                    while src_idx < len(parts) - 1 and parts[src_idx].startswith("--"):
                        src_idx += 1
                    for src in parts[src_idx:-1]:
                        if src == ".":
                            copied_files.add("__ALL__")
                        else:
                            copied_files.add(src.removeprefix("./").rstrip("/").rstrip("*"))

            line_lower = line.lower()
            is_install_cmd = "uv sync" in line_lower or "pip install" in line_lower
            if is_install_cmd and "__ALL__" not in copied_files:
                has_install_before_full_copy = True

        if has_install_before_full_copy and (self.root / "pyproject.toml").exists():
            issues.extend(self._check_pyproject_copy_order(copied_files, dockerfile.name))

        return issues

    def _validate_project_references(self, project: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        license_info = project.get("license")
        if isinstance(license_info, dict):
            license_file = license_info.get("file")
            if isinstance(license_file, str) and not (self.root / license_file).exists():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"License file not found: {license_file}",
                        file="pyproject.toml",
                        hint=(
                            f"Create a {license_file} file or remove the "
                            "license.file reference from pyproject.toml"
                        ),
                    )
                )

        readme = project.get("readme")
        if isinstance(readme, str) and not (self.root / readme).exists():
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Readme file not found: {readme}",
                    file="pyproject.toml",
                    hint=f"Create a {readme} file or remove the readme reference",
                )
            )
        elif isinstance(readme, dict):
            readme_file = readme.get("file")
            if isinstance(readme_file, str) and not (self.root / readme_file).exists():
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=f"Readme file not found: {readme_file}",
                        file="pyproject.toml",
                        hint=f"Create a {readme_file} file or remove the readme.file reference",
                    )
                )

        return issues

    def _validate_hatch_includes(self, targets: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for target_name, target_config in targets.items():
            if not isinstance(target_config, dict):
                continue
            includes = target_config.get("include", [])
            for pattern in includes:
                is_literal = isinstance(pattern, str) and "*" not in pattern and "?" not in pattern
                if is_literal and not (self.root / pattern).exists():
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            message=f"Included file/dir not found: {pattern}",
                            file="pyproject.toml",
                            hint=f"Referenced in [tool.hatch.build.targets.{target_name}].include",
                        )
                    )
        return issues

    def _check_pyproject_copy_order(
        self,
        copied_files: set[str],
        dockerfile_name: str,
    ) -> list[ValidationIssue]:
        pyproject_path = self.root / "pyproject.toml"
        try:
            with pyproject_path.open("rb") as file:
                data = tomllib.load(file)
        except tomllib.TOMLDecodeError:
            return []

        project = data.get("project", {})
        if not isinstance(project, dict):
            return []

        issues: list[ValidationIssue] = []
        license_info = project.get("license")
        if isinstance(license_info, dict):
            license_file = license_info.get("file")
            license_missing = (
                isinstance(license_file, str)
                and license_file.removeprefix("./") not in copied_files
            )
            if license_missing:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message="LICENSE file not copied before uv sync/pip install",
                        file=dockerfile_name,
                        hint=(
                            f"Add 'COPY {license_file} ./' before the RUN command "
                            "that installs dependencies"
                        ),
                    )
                )

        readme = project.get("readme")
        if isinstance(readme, str) and readme.removeprefix("./") not in copied_files:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message="README not copied before uv sync/pip install",
                    file=dockerfile_name,
                    hint=f"Add 'COPY {readme} ./' before the RUN command, or builds may fail",
                )
            )

        return issues


def _dockerfile_instructions(content: str) -> list[str]:
    """Logical Dockerfile instructions, joining ``\\`` line continuations."""
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


def _dockerfile_command_tokens(content: str) -> list[list[str]]:
    """Token lists for each CMD/ENTRYPOINT instruction in a Dockerfile."""
    commands: list[list[str]] = []
    for instruction in _dockerfile_instructions(content):
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


__all__ = ["EnvironmentNameReference", "EnvironmentSource", "ValidationIssue"]
