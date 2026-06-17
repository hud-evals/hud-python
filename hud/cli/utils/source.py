"""Filesystem-backed Environment source, config, and build identity."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

if TYPE_CHECKING:
    from collections.abc import Iterator

LOGGER = logging.getLogger(__name__)


def normalize_environment_name(name: str, *, default: str = "environment") -> str:
    """Slugify *name* into a valid environment name (lowercase, ``[a-z0-9-]``)."""
    normalized = name.strip().lower()
    normalized = normalized.replace(" ", "-").replace("_", "-")
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-") or default


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

    HUD_DIR: ClassVar[str] = ".hud"
    CONFIG_FILENAME: ClassVar[str] = "config.json"
    LEGACY_CONFIG_FILENAME: ClassVar[str] = "deploy.json"

    SOURCE_INCLUDE_FILES: ClassVar[set[str]] = {"Dockerfile", "Dockerfile.hud", "pyproject.toml"}
    SOURCE_INCLUDE_DIRS: ClassVar[set[str]] = {"server", "mcp", "controller", "environment"}
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
    SOURCE_EXCLUDE_FILES: ClassVar[set[str]] = {"hud.lock.yaml"}
    SOURCE_EXCLUDE_SUFFIXES: ClassVar[set[str]] = {".pyc", ".log"}

    @classmethod
    def open(cls, directory: str | Path = ".") -> Self:
        p = Path(directory).expanduser().resolve()
        if p.is_file():
            p = p.parent
        return cls(p)

    @property
    def hud_dir(self) -> Path:
        return self.root / self.HUD_DIR

    @property
    def config_path(self) -> Path:
        return self.hud_dir / self.CONFIG_FILENAME

    @property
    def legacy_config_path(self) -> Path:
        return self.hud_dir / self.LEGACY_CONFIG_FILENAME

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
        py_files = list(self.root.glob("*.py")) + list(self.root.glob("*/*.py"))
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
                        text=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
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

        served_file = (self.root / module).with_suffix(".py").resolve()
        names = {
            ref.name
            for ref in self.environment_name_references()
            if ref.file.resolve() == served_file and ref.name is not None
        }
        return next(iter(names)) if len(names) == 1 else None

    def environment_name(self) -> str:
        """Directory-derived fallback name for projects without ``Environment(...)``."""
        directory_name = self.root.name or self.root.parent.name
        return normalize_environment_name(directory_name)

    def load_config(self) -> dict[str, Any]:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                LOGGER.warning("Failed to parse %s, returning empty config", self.config_path)
                return {}

        if self.legacy_config_path.exists():
            try:
                data = json.loads(self.legacy_config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
            self._migrate_legacy_config(data)
            return data

        return {}

    def save_config(self, data: dict[str, Any]) -> Path | None:
        existing = self.load_config()
        merged = {**existing, **data}

        if merged == existing and self.config_path.exists():
            return None

        self.hud_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        return self.config_path

    @property
    def taskset_id(self) -> str | None:
        value = self.load_config().get("tasksetId")
        return value if isinstance(value, str) else None

    def iter_source_files(self) -> Iterator[Path]:
        for name in self.SOURCE_INCLUDE_FILES:
            path = self.root / name
            if path.is_file():
                yield path

        for directory in self.SOURCE_INCLUDE_DIRS:
            source_dir = self.root / directory
            if not source_dir.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(source_dir):
                dirnames[:] = [name for name in dirnames if name not in self.SOURCE_EXCLUDE_DIRS]
                for filename in filenames:
                    if filename in self.SOURCE_EXCLUDE_FILES:
                        continue
                    if any(filename.endswith(suffix) for suffix in self.SOURCE_EXCLUDE_SUFFIXES):
                        continue
                    yield Path(dirpath) / filename

    def source_files(self) -> list[Path]:
        files = list(self.iter_source_files())
        files.sort(key=self.relative_path)
        return files

    def source_file_refs(self) -> list[str]:
        return [self.relative_path(path) for path in self.source_files()]

    def source_hash(self) -> str:
        hasher = hashlib.sha256()
        for path in self.source_files():
            hasher.update(self.relative_path(path).encode("utf-8"))
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(8192), b""):
                    hasher.update(chunk)
        return hasher.hexdigest()

    def relative_path(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root)).replace("\\", "/")

    def base_image(self) -> str | None:
        """The Dockerfile's first ``FROM`` image, stage name stripped."""
        dockerfile = self.dockerfile
        return _parse_base_image(dockerfile) if dockerfile is not None else None

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

    def _migrate_legacy_config(self, data: dict[str, Any]) -> None:
        try:
            self.hud_dir.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            self.legacy_config_path.unlink()
            LOGGER.info("Migrated .hud/deploy.json to .hud/config.json")
        except OSError as exc:
            LOGGER.warning("Failed to migrate deploy.json to config.json: %s", exc)


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
    """The serve target from a ``hud serve|dev <spec>`` token list.

    Returns the explicit ``module[:attr]`` spec, ``"env"`` when ``hud serve`` is
    invoked with no target (the runtime default), or ``None`` when the tokens
    contain no ``hud serve``/``hud dev`` invocation.
    """
    for index, token in enumerate(tokens):
        if Path(token).name != "hud":
            continue
        rest = tokens[index + 1 :]
        if not rest or rest[0] not in {"serve", "dev"}:
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


def _parse_base_image(dockerfile_path: Path) -> str | None:
    try:
        if not dockerfile_path.exists():
            return None
        for raw_line in dockerfile_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("FROM "):
                rest = line[5:].strip()
                lower = rest.lower()
                if " as " in lower:
                    rest = rest[: lower.index(" as ")]
                return rest.strip()
    except OSError:
        return None
    return None


__all__ = ["EnvironmentNameReference", "EnvironmentSource", "ValidationIssue"]
