"""Filesystem-backed Environment source, config, and build identity."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

if TYPE_CHECKING:
    from collections.abc import Iterator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    message: str
    file: str | None = None
    hint: str | None = None


@dataclass(frozen=True)
class EnvironmentNameReference:
    file: Path
    line: int
    text: str
    name: str


@dataclass(frozen=True)
class EnvironmentSource:
    """A local Environment source tree rooted at a filesystem directory."""

    root: Path

    HUD_DIR: ClassVar[str] = ".hud"
    CONFIG_FILENAME: ClassVar[str] = "config.json"
    LEGACY_CONFIG_FILENAME: ClassVar[str] = "deploy.json"
    LOCK_FILENAME: ClassVar[str] = "hud.lock.yaml"

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
    ENV_NAME_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r'Environment\(["\']([^"\']+)["\']\)')

    @classmethod
    def open(cls, directory: str | Path = ".") -> Self:
        return cls(Path(directory).expanduser().resolve())

    @staticmethod
    def normalize_environment_name(name: str) -> str:
        normalized = name.strip().lower()
        normalized = normalized.replace(" ", "-").replace("_", "-")
        normalized = re.sub(r"[^a-z0-9-]", "", normalized)
        normalized = re.sub(r"-+", "-", normalized)
        return normalized.strip("-") or "environment"

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
    def lock_path(self) -> Path:
        return self.root / self.LOCK_FILENAME

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

    def manifest(self) -> dict[str, Any]:
        """Read this source tree's declared Environment manifest."""
        from hud.environment import Environment
        from hud.eval import Taskset, load_module

        env_file = self.root / "env.py"
        if not env_file.exists():
            raise FileNotFoundError(f"no env.py found in {self.root}")

        module = load_module(env_file)
        envs = [value for value in vars(module).values() if isinstance(value, Environment)]
        if not envs:
            raise ValueError(f"no Environment instance defined in {env_file}")
        if len(envs) > 1:
            raise ValueError(f"multiple Environments in {env_file}; expected exactly one")

        manifest = envs[0].to_dict()
        taskset = Taskset._from_module(self.root, preloaded={env_file.resolve(): module})
        if taskset:
            manifest["tasks"] = [
                {"slug": slug, "task": task.id, "args": task.args} for slug, task in taskset.items()
            ]
        return manifest

    def environment_name_references(self) -> list[EnvironmentNameReference]:
        """Find positional ``Environment("name")`` references in project source."""
        references: list[EnvironmentNameReference] = []
        py_files = list(self.root.glob("*.py")) + list(self.root.glob("*/*.py"))
        for py_file in py_files:
            try:
                lines = py_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, 1):
                references.extend(
                    EnvironmentNameReference(
                        file=py_file,
                        line=line_no,
                        text=line.strip(),
                        name=match.group(1),
                    )
                    for match in self.ENV_NAME_PATTERN.finditer(line)
                )
        return references

    def environment_name(self, override: str | None = None) -> str:
        if override:
            return self.normalize_environment_name(override)

        directory_name = self.root.name or self.root.parent.name
        return self.normalize_environment_name(directory_name)

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

    def dockerfile_env_vars(self) -> list[str]:
        """Runtime env vars the Dockerfile requires (``ENV`` without a value)."""
        dockerfile = self.dockerfile
        return _extract_dockerfile_env_vars(dockerfile) if dockerfile is not None else []

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


def _extract_dockerfile_env_vars(dockerfile_path: Path) -> list[str]:
    required: list[str] = []

    if not dockerfile_path.exists():
        return required

    content = dockerfile_path.read_text(encoding="utf-8")
    arg_vars: set[str] = set()

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("ARG "):
            parts = line[4:].strip().split("=", 1)
            var_name = parts[0].strip()
            if len(parts) == 1 or not parts[1].strip():
                arg_vars.add(var_name)
        elif line.startswith("ENV "):
            parts = line[4:].strip().split("=", 1)
            var_name = parts[0].strip()
            if len(parts) == 2 and parts[1].strip().startswith("$"):
                ref_var = parts[1].strip()[1:]
                if ref_var in arg_vars and var_name not in required:
                    required.append(var_name)
            elif len(parts) == 2 and not parts[1].strip():
                if var_name not in required:
                    required.append(var_name)
            elif len(parts) == 1 and var_name not in required:
                required.append(var_name)

    return required


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
