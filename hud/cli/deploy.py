"""Deploy HUD environments to the platform via direct build."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import sys
import tarfile
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import httpx
import typer
import websockets
from pydantic import ValidationError
from rich.panel import Panel
from rich.table import Table
from websockets.exceptions import ConnectionClosed

from hud.cli import require_api_key
from hud.cli.config import (
    AuthScope,
    DirectoryLink,
    DirectoryState,
    parse_env_file,
    parse_key_value,
)
from hud.cli.io import (
    CliError,
    emit_json,
    map_exception,
    mark_json,
)
from hud.cli.project import (
    PROJECT_OPTION_HELP,
    Placement,
    require_writable_placement,
    resolve_placement,
)
from hud.cli.source import EnvironmentSource
from hud.cli.sync import get_registry_environment
from hud.eval.runtime import ComposeProject, RuntimeConfig
from hud.settings import settings
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.naming import normalize_environment_name
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from rich.console import Console

_VALID_RUNTIMES = {"hud", "modal"}
_COMPOSE_RECIPE_NAMES = (
    "compose.yaml",
    "compose.yml",
    "docker-compose.yaml",
    "docker-compose.yml",
)


@dataclass(frozen=True)
class _DeployPlan:
    name: str
    registry_id: str | None
    placement: Placement
    runtime: str | None
    runtime_config: RuntimeConfig | None
    env_vars: dict[str, str]
    build_args: dict[str, str]
    build_secrets: dict[str, str]
    state: DirectoryState
    dotenv_pending: bool = False
    dotenv_consent: bool | None = None
    save_link: bool = True


def _parse_key_value_flags(
    flags: list[str] | None,
    *,
    option: str,
    console: HUDConsole,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for flag in flags or []:
        parsed = parse_key_value(flag)
        if parsed is None:
            console.warning(f"Invalid {option} format: {flag} (expected KEY=VALUE)")
            continue
        values[parsed[0]] = parsed[1]
    return values


def _normalize_runtime(runtime: str | None, console: HUDConsole) -> str | None:
    if runtime is None:
        return None
    normalized = runtime.strip().lower()
    if normalized in _VALID_RUNTIMES:
        return normalized
    raise ValueError(
        f"Invalid runtime {runtime!r}; expected one of: {', '.join(sorted(_VALID_RUNTIMES))}"
    )


def _compose_recipe(context: Path) -> Path | None:
    for name in _COMPOSE_RECIPE_NAMES:
        candidate = context / name
        if candidate.is_file():
            return candidate
    return next(
        (
            candidate
            for candidate in sorted(context.glob("docker-compose.*"))
            if ".override." not in candidate.name and candidate.is_file()
        ),
        None,
    )


def _load_runtime_config(path: str | None, console: HUDConsole) -> RuntimeConfig | None:
    if path is None:
        return None
    config_path = Path(path).expanduser()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw_config = cast("dict[str, Any]", raw)
            compose = raw_config.get("compose")
            if isinstance(compose, dict):
                compose_config = cast("dict[str, Any]", compose)
                for field in ("document", "root"):
                    value = compose_config.get(field)
                    if not isinstance(value, str):
                        continue
                    candidate = Path(value).expanduser()
                    compose_config[field] = str(
                        candidate
                        if candidate.is_absolute()
                        else (config_path.parent / candidate).resolve()
                    )
        config = RuntimeConfig.model_validate(raw)
    except FileNotFoundError:
        raise ValueError(f"Runtime config file not found: {config_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid runtime config JSON in {config_path}: {exc.msg}") from exc
    except ValidationError as exc:
        raise ValueError(f"Invalid runtime config in {config_path}: {exc}") from exc
    return config


def _load_env_vars(path: Path, console: HUDConsole, *, required: bool) -> dict[str, str]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Env file not found: {path}")
        return {}

    console.info(f"Loading environment variables from {path}")
    return parse_env_file(path.read_text(encoding="utf-8"))


def collect_environment_variables(
    directory: Path,
    env_flags: list[str] | None,
    env_file: str | None,
    console: HUDConsole,
    *,
    skip_dotenv: bool = False,
) -> dict[str, str]:
    """Collect deploy environment variables from .env/--env-file plus --env overrides."""
    if env_file:
        env_vars = _load_env_vars(Path(env_file), console, required=True)
    elif not skip_dotenv:
        env_vars = _load_env_vars(directory / ".env", console, required=False)
    else:
        env_vars = {}

    env_vars.update(_parse_key_value_flags(env_flags, option="--env", console=console))
    return env_vars


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    message: str
    file: str | None = None
    hint: str | None = None


def _load_pyproject(env_source: EnvironmentSource) -> dict[str, Any] | ValidationIssue:
    pyproject_path = env_source.root / "pyproject.toml"
    if not pyproject_path.exists():
        return {}
    try:
        with pyproject_path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        return ValidationIssue(
            severity="error",
            message=f"Failed to parse pyproject.toml: {exc}",
            file="pyproject.toml",
        )
    return data


def _validate_project_references(
    env_source: EnvironmentSource, project: dict[str, Any]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    license_info = project.get("license")
    if isinstance(license_info, dict):
        license_file = license_info.get("file")
        if isinstance(license_file, str) and not (env_source.root / license_file).exists():
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
    if isinstance(readme, str) and not (env_source.root / readme).exists():
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
        if isinstance(readme_file, str) and not (env_source.root / readme_file).exists():
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Readme file not found: {readme_file}",
                    file="pyproject.toml",
                    hint=f"Create a {readme_file} file or remove the readme.file reference",
                )
            )

    return issues


def _validate_hatch_includes(
    env_source: EnvironmentSource, targets: dict[str, Any]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for target_name, target_config in targets.items():
        if not isinstance(target_config, dict):
            continue
        includes = target_config.get("include", [])
        for pattern in includes:
            is_literal = isinstance(pattern, str) and "*" not in pattern and "?" not in pattern
            if is_literal and not (env_source.root / pattern).exists():
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=f"Included file/dir not found: {pattern}",
                        file="pyproject.toml",
                        hint=f"Referenced in [tool.hatch.build.targets.{target_name}].include",
                    )
                )
    return issues


def _validate_pyproject(env_source: EnvironmentSource) -> list[ValidationIssue]:
    data = _load_pyproject(env_source)
    if isinstance(data, ValidationIssue):
        return [data]

    issues: list[ValidationIssue] = []
    project = data.get("project", {})
    if isinstance(project, dict):
        issues.extend(_validate_project_references(env_source, project))

    tool = data.get("tool", {})
    if isinstance(tool, dict):
        hatch = tool.get("hatch", {})
        if isinstance(hatch, dict):
            build = hatch.get("build", {})
            if isinstance(build, dict):
                targets = build.get("targets", {})
                if isinstance(targets, dict):
                    issues.extend(_validate_hatch_includes(env_source, targets))

    return issues


def _copied_dockerfile_sources(instruction: str) -> list[str]:
    if not instruction.upper().startswith("COPY "):
        return []
    parts = instruction.split()
    if len(parts) < 3:
        return []
    src_idx = 1
    while src_idx < len(parts) - 1 and parts[src_idx].startswith("--"):
        src_idx += 1
    return [
        "__ALL__" if src == "." else src.removeprefix("./").rstrip("/").rstrip("*")
        for src in parts[src_idx:-1]
    ]


def _check_pyproject_copy_order(
    env_source: EnvironmentSource,
    project: dict[str, Any],
    copied_files: set[str],
    dockerfile_name: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    license_info = project.get("license")
    if isinstance(license_info, dict):
        license_file = license_info.get("file")
        license_missing = (
            isinstance(license_file, str) and license_file.removeprefix("./") not in copied_files
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


def _validate_dockerfile(env_source: EnvironmentSource) -> list[ValidationIssue]:
    dockerfile = env_source.dockerfile
    if dockerfile is None:
        return []

    copied_files: set[str] = set()
    has_install_before_full_copy = False
    for instruction in env_source.dockerfile_instructions():
        copied_files.update(_copied_dockerfile_sources(instruction))
        line_lower = instruction.lower()
        if (
            "uv sync" in line_lower or "pip install" in line_lower
        ) and "__ALL__" not in copied_files:
            has_install_before_full_copy = True

    if not has_install_before_full_copy or not (env_source.root / "pyproject.toml").exists():
        return []

    data = _load_pyproject(env_source)
    if isinstance(data, ValidationIssue):
        return [data]
    project = data.get("project", {})
    if not isinstance(project, dict):
        return []
    return _check_pyproject_copy_order(env_source, project, copied_files, dockerfile.name)


def _validate_environment(env_source: EnvironmentSource) -> list[ValidationIssue]:
    return [*_validate_pyproject(env_source), *_validate_dockerfile(env_source)]


def _validate_before_deploy(env_source: EnvironmentSource, console: HUDConsole) -> None:
    console.progress_message("Validating environment...")
    validation_issues = _validate_environment(env_source)

    errors = [issue for issue in validation_issues if issue.severity == "error"]
    warnings = [issue for issue in validation_issues if issue.severity == "warning"]

    if errors:
        raise ValueError(
            "Environment validation failed: "
            + "; ".join(f"{issue.message} ({issue.file})" for issue in errors)
        )

    if warnings:
        console.warning(f"Found {len(warnings)} warning(s):")
        for issue in warnings:
            file_info = f" ({issue.file})" if issue.file else ""
            console.warning(f"  {issue.message}{file_info}")
            if issue.hint:
                console.dim_info("    Hint:", issue.hint)
        console.info("")

    if not validation_issues:
        console.success("Validation passed")


def _resolve_declared_name(env_source: EnvironmentSource, console: HUDConsole) -> str:
    """Resolve the environment name declared in code.

    Prefers the Environment served by the Dockerfile entrypoint
    (``hud serve module:attr``), so a project may define auxiliary in-process
    Environments — e.g. a verification sub-agent — without making the
    deployable identity ambiguous. Otherwise a lone declared name wins, and the
    choice is only an error when nothing disambiguates between several names.
    """
    served = env_source.served_environment_name()
    if served is not None:
        return served

    names = {ref.name for ref in env_source.environment_name_references() if ref.name is not None}
    if len(names) != 1:
        raise ValueError(
            "Declare exactly one literal Environment name "
            "or select the served module in the Dockerfile."
        )
    return next(iter(names))


def _resolve_environment_name(
    env_source: EnvironmentSource,
    registry_id: str | None,
    platform: PlatformClient,
    console: HUDConsole,
) -> str:
    """Resolve the environment name from source code.

    The name declared in ``Environment(...)`` is the environment's identity:
    the platform resolves the target registry by this name (get-or-rebuild).
    Projects must declare an ``Environment(...)`` in source.
    """
    name = _resolve_declared_name(env_source, console)

    if registry_id:
        registry_env = get_registry_environment(platform, registry_id)
        if normalize_environment_name(name) != registry_env.name:
            raise ValueError(
                f"Code declares Environment('{name}') but --registry-id targets "
                f"'{registry_env.name}'. Rename the environment in code or drop "
                "--registry-id to deploy by name."
            )
    console.info(f"Environment name: {name}")
    return name


def _collect_build_secrets(
    secret_specs: list[str] | None,
    *,
    env_dir: Path,
    console: HUDConsole,
) -> dict[str, str]:
    secrets: dict[str, str] = {}
    for secret_spec in secret_specs or []:
        parts: dict[str, str] = {}
        for part in secret_spec.split(","):
            key, sep, value = part.partition("=")
            if sep:
                parts[key.strip()] = value.strip()
        secret_id = parts.get("id")
        if not secret_id:
            raise ValueError(f"Invalid --secret format: {secret_spec} (missing id=)")

        if "env" in parts:
            env_name = parts["env"]
            value = os.environ.get(env_name)
            if value is None:
                raise ValueError(
                    f"Secret '{secret_id}': environment variable '{env_name}' is not set"
                )
            secrets[secret_id] = value
            continue

        if "src" in parts:
            src_path = Path(parts["src"]).expanduser()
            if not src_path.is_absolute():
                src_path = env_dir / src_path
            if not src_path.exists():
                raise ValueError(f"Secret '{secret_id}': file not found: {src_path}")
            try:
                secrets[secret_id] = src_path.read_text(encoding="utf-8")
            except OSError as e:
                raise ValueError(f"Secret '{secret_id}': failed to read {src_path}: {e}") from e
            continue

        raise ValueError(f"Invalid --secret format: {secret_spec} (need env= or src=)")
    return secrets


SENSITIVE_EXCLUDES = [".git", ".git/*", ".env", ".env.*", "*.env"]
DEFAULT_EXCLUDES = [
    "__pycache__",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    ".venv",
    ".venv/*",
    "venv",
    "venv/*",
    "node_modules",
    "node_modules/*",
    ".mypy_cache",
    ".mypy_cache/*",
    ".pytest_cache",
    ".pytest_cache/*",
    ".ruff_cache",
    ".ruff_cache/*",
    "*.egg-info",
    "*.egg-info/*",
    "dist",
    "dist/*",
    "build",
    "build/*",
    ".DS_Store",
    "Thumbs.db",
]


def parse_ignore_file(ignore_path: Path) -> list[str]:
    if not ignore_path.exists():
        return []
    return [
        line.strip()
        for line in ignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _matches_pattern(rel_path_str: str, path: Path, pattern: str) -> bool:
    if pattern.endswith("/"):
        pattern = pattern[:-1]
        if path.is_dir() and fnmatch.fnmatch(rel_path_str, pattern):
            return True
        return fnmatch.fnmatch(rel_path_str, f"{pattern}/*")

    if "**" in pattern:
        regex_pattern = pattern.replace("**", "*")
        if fnmatch.fnmatch(rel_path_str, regex_pattern):
            return True
        parts = rel_path_str.split("/")
        for i in range(len(parts)):
            partial = "/".join(parts[: i + 1])
            if fnmatch.fnmatch(partial, regex_pattern):
                return True
        return False

    if fnmatch.fnmatch(rel_path_str, pattern):
        return True
    if fnmatch.fnmatch(path.name, pattern):
        return True
    parts = rel_path_str.split("/")
    for i in range(len(parts)):
        partial = "/".join(parts[: i + 1])
        if fnmatch.fnmatch(partial, pattern):
            return True
    return False


def should_ignore(path: Path, base_path: Path, ignore_patterns: list[str]) -> bool:
    try:
        rel_path_str = str(path.relative_to(base_path)).replace("\\", "/")
    except ValueError:
        return False

    ignored = False
    for pattern in ignore_patterns:
        if pattern.startswith("!"):
            if ignored and _matches_pattern(rel_path_str, path, pattern[1:]):
                ignored = False
        elif _matches_pattern(rel_path_str, path, pattern):
            ignored = True
    return ignored


def create_build_context_tarball(
    directory: Path,
    dockerignore_path: Path | None = None,
    verbose: bool = False,
) -> tuple[Path, int, int, float]:
    start_time = time.time()
    hud_console = HUDConsole()
    directory = directory.resolve()

    ignore_patterns = list(DEFAULT_EXCLUDES)
    loaded_sources: list[str] = []

    gitignore_path = directory / ".gitignore"
    if gitignore_path.exists():
        gitignore_patterns = parse_ignore_file(gitignore_path)
        ignore_patterns.extend(gitignore_patterns)
        loaded_sources.append(f".gitignore ({len(gitignore_patterns)} patterns)")

    if dockerignore_path is None:
        dockerignore_path = directory / ".dockerignore"
    if dockerignore_path.exists():
        dockerignore_patterns = parse_ignore_file(dockerignore_path)
        ignore_patterns.extend(dockerignore_patterns)
        loaded_sources.append(f".dockerignore ({len(dockerignore_patterns)} patterns)")

    ignore_patterns.extend(SENSITIVE_EXCLUDES)

    if verbose and loaded_sources:
        hud_console.info(f"Loaded ignore patterns from: {', '.join(loaded_sources)}")

    temp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        suffix=".tar.gz",
        delete=False,
        prefix="hud-build-context-",
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    file_count = 0
    try:
        with tarfile.open(temp_path, "w:gz") as tar:
            for root, dirs, files in os.walk(directory):
                root_path = Path(root)
                dirs[:] = [
                    d for d in dirs if not should_ignore(root_path / d, directory, ignore_patterns)
                ]
                for child in dirs:
                    dir_path = root_path / child
                    tar.add(
                        dir_path,
                        arcname=str(dir_path.relative_to(directory)),
                        recursive=False,
                    )
                for file in files:
                    file_path = root_path / file
                    if should_ignore(file_path, directory, ignore_patterns):
                        if verbose:
                            hud_console.debug(f"Skipping: {file_path.relative_to(directory)}")
                        continue
                    arcname = str(file_path.relative_to(directory))
                    tar.add(file_path, arcname=arcname)
                    file_count += 1
                    if verbose:
                        hud_console.debug(f"Added: {arcname}")

        return temp_path, temp_path.stat().st_size, file_count, time.time() - start_time
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def format_size(size_bytes: int) -> str:
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _create_tarball(env_dir: Path, *, verbose: bool, console: HUDConsole) -> Path:
    console.progress_message("Creating build context tarball...")
    try:
        tarball_path, tarball_size, file_count, tarball_duration = create_build_context_tarball(
            env_dir,
            verbose=verbose,
        )
    except Exception as e:
        raise ValueError(f"Failed to create build context: {e}") from e

    console.success(
        f"Created tarball: {format_size(tarball_size)} ({file_count} files) "
        f"[{tarball_duration:.1f}s]"
    )
    return tarball_path


def _prepare_deploy_plan(
    env_source: EnvironmentSource,
    *,
    env_dir: Path,
    env: list[str] | None,
    env_file: str | None,
    no_env: bool,
    registry_id: str | None,
    project: str | None,
    build_args: list[str] | None,
    build_secrets: list[str] | None,
    runtime: str | None,
    runtime_config: str | None,
    verbose: bool,
    platform: PlatformClient,
    console: HUDConsole,
) -> _DeployPlan:
    state = DirectoryState(AuthScope.resolve(platform), env_dir)
    link = state.load()
    linked = None
    if link.registry_id and registry_id is None:
        linked = get_registry_environment(platform, str(link.registry_id))
    resolved_name = _resolve_environment_name(
        env_source,
        registry_id,
        platform,
        console,
    )
    placement = resolve_placement(platform, link, flag=project)
    require_writable_placement(placement)
    consent = None
    if (
        linked is not None
        and linked.name == normalize_environment_name(resolved_name)
        and (placement.project_id is None or placement.project_id == linked.project_id)
    ):
        consent = link.sync_env.get(UUID(linked.id))
    dotenv_pending = (
        not no_env and not env_file and (env_dir / ".env").is_file() and consent is None
    )
    skip_dotenv = no_env or bool(env_file) or consent is not True

    env_vars = collect_environment_variables(
        env_dir,
        env,
        env_file,
        console,
        skip_dotenv=skip_dotenv,
    )
    if env and not skip_dotenv and not env_file and env_vars and (env_dir / ".env").exists():
        console.dim_info("Env merge:", ".env + --env flags (--env values take priority)")
    if env_vars and verbose:
        console.info(f"Environment variables: {', '.join(env_vars.keys())}")

    build_args_dict = _parse_key_value_flags(build_args, option="--build-arg", console=console)
    if build_args_dict and verbose:
        console.info(f"Build arguments: {', '.join(build_args_dict.keys())}")
    normalized_runtime = _normalize_runtime(runtime, console)
    loaded_runtime_config = _load_runtime_config(runtime_config, console)
    recipe = _compose_recipe(env_dir)
    if recipe is not None:
        if loaded_runtime_config is not None and (
            loaded_runtime_config.image is not None or loaded_runtime_config.compose is not None
        ):
            raise ValueError("--runtime-config cannot set image or Compose for a Compose context")
        loaded_runtime_config = RuntimeConfig.model_validate(
            {
                **(
                    loaded_runtime_config.model_dump(exclude_unset=True)
                    if loaded_runtime_config is not None
                    else {}
                ),
                "compose": ComposeProject(document=recipe, root=env_dir),
            }
        )

    return _DeployPlan(
        state=state,
        dotenv_pending=dotenv_pending,
        save_link=registry_id is None and project is None,
        name=resolved_name,
        registry_id=registry_id,
        placement=placement,
        runtime=normalized_runtime,
        runtime_config=loaded_runtime_config,
        env_vars=env_vars,
        build_args=build_args_dict,
        build_secrets=_collect_build_secrets(build_secrets, env_dir=env_dir, console=console),
    )


def display_build_summary(
    status_response: dict[str, Any],
    registry_id: str,
    console: HUDConsole | None = None,
    platform_url: str | None = None,
    env_name: str | None = None,
) -> None:
    """Display a rich summary of a completed build."""
    if console is None:
        console = HUDConsole()

    if platform_url is None:
        platform_url = settings.hud_web_url

    rich_console = console.console

    status = status_response.get("status", "UNKNOWN")
    version = status_response.get("version", "unknown")
    duration = status_response.get("duration_seconds")
    image_name = status_response.get("image_name")
    uri = status_response.get("uri")
    lock_data = status_response.get("lock")

    duration_str = _format_duration(duration) if duration else "unknown"

    if status == "SUCCEEDED":
        status_text = "[green]✓[/green] [bold green]SUCCEEDED[/bold green]"
    elif status == "FAILED":
        status_text = "[red]✗[/red] [bold red]FAILED[/bold red]"
    else:
        status_text = f"[yellow]●[/yellow] [bold yellow]{status}[/bold yellow]"

    summary_lines = [
        f"[bold]Status:[/bold]     {status_text}",
        f"[bold]Duration:[/bold]   {duration_str}",
        f"[bold]Version:[/bold]    {version}",
    ]

    if env_name:
        summary_lines.insert(0, f"[bold]Environment:[/bold] [cyan]{env_name}[/cyan]")

    if uri:
        summary_lines.append(f"[bold]Image:[/bold]      [dim]{uri}[/dim]")
    elif image_name:
        summary_lines.append(f"[bold]Image:[/bold]      [dim]{image_name}[/dim]")

    summary_content = "\n".join(summary_lines)

    rich_console.print()
    rich_console.print(
        Panel(
            summary_content,
            title="[bold cyan]Build Summary[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if lock_data and isinstance(lock_data, dict):
        _display_lock_details(rich_console, lock_data)

    env_url = f"{platform_url}/environments/{registry_id}"
    rich_console.print()
    rich_console.print(
        Panel(
            f"[bold]View on HUD:[/bold] [link={env_url}]{env_url}[/link]",
            border_style="blue",
            padding=(0, 2),
        )
    )

    if status == "SUCCEEDED" and env_name and lock_data:
        _display_usage_example(rich_console, env_name, lock_data)

    rich_console.print()


def _display_lock_details(
    rich_console: Console,
    lock_data: dict[str, Any],
) -> None:
    tasks = lock_data.get("tasks") or []
    if tasks:
        rich_console.print()
        tasks_table = Table(
            title=f"[bold]Tasks ({len(tasks)})[/bold]",
            show_header=True,
            header_style="bold",
            border_style="dim",
        )
        tasks_table.add_column("Slug", style="cyan")
        tasks_table.add_column("Task", style="magenta")
        tasks_table.add_column("Args", style="dim")

        for task in tasks[:10]:
            if not isinstance(task, dict):
                tasks_table.add_row(str(task), "", "")
                continue
            slug = str(task.get("slug") or "")
            task_id = str(task.get("task") or task.get("id") or "")
            args = task.get("args") or {}
            args_str = ", ".join(sorted(args)) if isinstance(args, dict) and args else "No args"
            tasks_table.add_row(slug, task_id, args_str)

        if len(tasks) > 10:
            tasks_table.add_row(
                f"[dim]... and {len(tasks) - 10} more[/dim]",
                "",
                "",
            )

        rich_console.print(tasks_table)

    env_config = lock_data.get("environment") or {}
    if env_config:
        variables = env_config.get("variables") or {}
        required_vars = variables.get("required", [])
        optional_vars = variables.get("optional", [])

        if required_vars or optional_vars:
            rich_console.print()
            env_lines = []
            if required_vars:
                env_lines.append(f"[bold]Required:[/bold] {', '.join(required_vars)}")
            if optional_vars:
                env_lines.append(f"[bold]Optional:[/bold] {', '.join(optional_vars)}")

            rich_console.print(
                Panel(
                    "\n".join(env_lines),
                    title="[bold]Environment Variables[/bold]",
                    border_style="dim",
                    padding=(0, 2),
                )
            )

    capabilities = lock_data.get("capabilities") or []
    if capabilities:
        capability_names = [
            capability.get("name", str(capability))
            if isinstance(capability, dict)
            else str(capability)
            for capability in capabilities[:10]
        ]
        capabilities_str = ", ".join(capability_names)
        if len(capabilities) > 10:
            capabilities_str += f", ... and {len(capabilities) - 10} more"

        rich_console.print()
        rich_console.print(
            Panel(
                f"[bold]Capabilities ({len(capabilities)}):[/bold] {capabilities_str}",
                border_style="dim",
                padding=(0, 2),
            )
        )


def _display_usage_example(
    rich_console: Console,
    env_name: str,
    lock_data: dict[str, Any],
) -> None:
    tasks = lock_data.get("tasks") or []
    if not tasks:
        return

    first = tasks[0]
    if not isinstance(first, dict):
        return

    task_example: dict[str, Any] = {
        "env": env_name,
        "id": first.get("task") or first.get("id") or "",
    }
    if first.get("slug"):
        task_example["slug"] = first["slug"]
    args = first.get("args")
    if isinstance(args, dict) and args:
        task_example["args"] = args

    example_json = json.dumps(task_example, indent=2)
    rich_console.print()
    rich_console.print(
        Panel(
            f"[bold]Task JSON:[/bold]\n[dim]{example_json}[/dim]",
            title="[bold]Quick Start[/bold]",
            border_style="green",
            padding=(1, 2),
        )
    )


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


async def wait_for_build(
    platform: PlatformClient, build_id: str, console: HUDConsole
) -> dict[str, Any]:
    """Stream progress, then confirm the terminal result through the status API."""
    await _stream_build_logs(platform, build_id, console=console)
    return await _poll_build_status(platform, build_id, console=console)


async def _stream_build_logs(
    platform: PlatformClient,
    build_id: str,
    console: HUDConsole | None = None,
    max_reconnects: int = 3,
) -> None:
    if console is None:
        console = HUDConsole()

    ws_base = platform.base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base.rstrip('/')}/builds/{build_id}/logs?api_key={platform.api_key}"

    for attempt in range(max_reconnects + 1):
        try:
            console.info("Connecting to build logs stream...")
            async with websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10,
            ) as websocket:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")

                        if msg_type == "status":
                            console.info(data.get("message", "Connected"))

                        elif msg_type == "status_update":
                            status = data.get("status", "")
                            if status != "IN_PROGRESS":
                                console.info(f"Build status: {status}")

                        elif msg_type == "log":
                            log_message = data.get("message", "")
                            timestamp = data.get("timestamp")
                            if log_message:
                                _print_log_line(console, log_message, timestamp)

                        elif msg_type == "complete":
                            final_status = data.get("final_status", "UNKNOWN")
                            completion_msg = data.get("message", f"Build {final_status}")
                            console.info(completion_msg)
                            return

                        elif msg_type == "error":
                            error_msg = data.get("error", "Unknown error")
                            console.error(f"Build error: {error_msg}")
                            return

                    except json.JSONDecodeError:
                        console.info(str(message))

        except ConnectionClosed as e:
            if e.code == 4003:
                console.error(f"Access denied: {e.reason}")
                return

            console.warning(f"Log stream closed: {e.reason}")
        except Exception as e:
            console.warning(f"Log stream unavailable: {e}")

        if attempt < max_reconnects:
            await asyncio.sleep(min(2 ** (attempt + 1), 30))


def _print_log_line(
    console: HUDConsole,
    message: str,
    timestamp: str | int | None = None,
) -> None:
    message = message.rstrip()

    prefix = ""
    if timestamp:
        try:
            if isinstance(timestamp, int):
                dt = datetime.fromtimestamp(timestamp / 1000)
                prefix = f"[{dt.strftime('%H:%M:%S')}] "
            elif isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                prefix = f"[{dt.strftime('%H:%M:%S')}] "
        except Exception:  # noqa: S110
            pass

    lower_msg = message.lower()

    is_error_handling = any(
        pattern in lower_msg
        for pattern in [
            "2>/dev/null",
            "|| true",
            "|| echo",
            "|| :",
            "if [",
            "if aws",
            "if docker",
            "--quiet",
        ]
    )

    stripped_msg = message.strip()
    is_actual_error = not is_error_handling and (
        lower_msg.startswith(("error:", "error "))
        or "exit status 1" in lower_msg
        or "exit code: 1" in lower_msg
        or "command did not exit successfully" in lower_msg
        or "failed to" in lower_msg
        or ": FAILED" in message
        or "State: FAILED" in message
        or stripped_msg.startswith(("OSError:", "Exception:"))
    )

    if is_actual_error:
        console.error(f"{prefix}{message}")
    elif "warning" in lower_msg or "warn:" in lower_msg:
        console.warning(f"{prefix}{message}")
    elif "success" in lower_msg or "completed successfully" in lower_msg:
        console.success(f"{prefix}{message}")
    else:
        console.info(f"{prefix}{message}")


async def _poll_build_status(
    platform: PlatformClient,
    build_id: str,
    console: HUDConsole | None = None,
    poll_interval: float = 5.0,
    max_wait: float = 3600.0,
) -> dict[str, Any]:
    if console is None:
        console = HUDConsole()

    start_time = asyncio.get_event_loop().time()
    last_status = ""

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > max_wait:
            console.error(f"Build timed out after {max_wait}s")
            return {"status": "TIMED_OUT"}

        try:
            data = await platform.aget(f"/builds/{build_id}/status")

            status = data.get("status", "")
            if status != last_status:
                console.info(f"Build status: {status}")
                last_status = status

            if status in ["SUCCEEDED", "FAILED", "STOPPED", "TIMED_OUT"]:
                return data

        except HudRequestError as e:
            if e.status_code is not None and 400 <= e.status_code < 500 and e.status_code != 429:
                raise
            console.warning(f"Status check failed: {e.status_code or e}")

        await asyncio.sleep(poll_interval)


def deploy_environment(
    directory: str = ".",
    env: list[str] | None = None,
    env_file: str | None = None,
    no_env: bool = False,
    no_cache: bool = False,
    verbose: bool = False,
    registry_id: str | None = None,
    project: str | None = None,
    build_args: list[str] | None = None,
    build_secrets: list[str] | None = None,
    runtime: str | None = None,
    runtime_config: str | None = None,
    *,
    dry_run: bool = False,
) -> _DeployResult:
    """Prepare and execute one deployment, returning its complete result."""
    console = HUDConsole()
    env_source = EnvironmentSource.open(directory)
    env_dir = env_source.root
    require_api_key("deploy environments")
    if _compose_recipe(env_dir) is None and env_source.dockerfile is None:
        raise CliError(
            "failure",
            "No compose.yaml, compose.yml, docker-compose.*, or Dockerfile found",
            input={"directory": str(env_dir)},
            suggestion="Run 'hud init' to create a template.",
        )
    _validate_before_deploy(env_source, console)
    platform = PlatformClient.from_settings()
    plan = _prepare_deploy_plan(
        env_source,
        env_dir=env_dir,
        env=env,
        env_file=env_file,
        no_env=no_env,
        registry_id=registry_id,
        project=project,
        build_args=build_args,
        build_secrets=build_secrets,
        runtime=runtime,
        runtime_config=runtime_config,
        verbose=verbose,
        platform=platform,
        console=console,
    )
    if dry_run:
        return _DeployResult(
            success=True,
            name=plan.name,
            registry_id=plan.registry_id,
            dry_run=True,
            runtime=plan.runtime,
            env_var_keys=sorted(plan.env_vars),
            build_arg_keys=sorted(plan.build_args),
            dotenv_pending=plan.dotenv_pending,
        )
    if plan.dotenv_pending:
        if not sys.stdin.isatty():
            raise CliError(
                "usage",
                "Choose whether to upload .env before deploying.",
                suggestion="Pass --env-file .env to include it, or --no-env to skip it.",
            )
        consent = console.confirm("Include .env in deploy? (encrypted at rest)", default=False)
        plan = replace(
            plan,
            dotenv_consent=consent,
            env_vars=collect_environment_variables(
                env_dir,
                env,
                env_file,
                console,
                skip_dotenv=not consent,
            ),
        )
    tarball = _create_tarball(env_dir, verbose=verbose, console=console)
    try:
        return asyncio.run(
            _deploy_async(
                tarball_path=tarball,
                no_cache=no_cache,
                plan=plan,
                platform=platform,
                console=console,
                env_dir=env_dir,
            )
        )
    finally:
        tarball.unlink(missing_ok=True)


@dataclass(frozen=True)
class _DeployResult:
    success: bool
    action: str = "deploy"
    build_id: str | None = None
    registry_id: str | None = None
    status: str = ""
    name: str = ""
    dry_run: bool = False
    runtime: str | None = None
    env_var_keys: list[str] = field(default_factory=list)
    build_arg_keys: list[str] = field(default_factory=list)
    dotenv_pending: bool = False
    details: dict[str, Any] = field(default_factory=dict)


async def _upload_context(upload_url: str, tarball: Path) -> None:
    content = await asyncio.to_thread(tarball.read_bytes)
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.put(
            upload_url,
            content=content,
            headers={"Content-Type": "application/gzip"},
        )
        response.raise_for_status()


async def _trigger_build(
    platform: PlatformClient,
    *,
    build_id: str,
    plan: _DeployPlan,
    no_cache: bool,
) -> tuple[str, str]:
    payload: dict[str, Any] = {
        "source": "direct",
        "build_id": build_id,
        "name": plan.name,
        "no_cache": no_cache,
    }
    payload.update(
        {
            key: value
            for key, value in (
                ("registry_id", plan.registry_id),
                ("project_id", plan.placement.project_id),
                ("runtime_provider", plan.runtime),
                (
                    "runtime_config",
                    plan.runtime_config.model_dump(mode="json", exclude_unset=True)
                    if plan.runtime_config
                    else None,
                ),
                ("environment_variables", plan.env_vars),
                ("build_args", plan.build_args),
                ("build_secrets", plan.build_secrets),
            )
            if value
        }
    )
    data = await platform.apost("/builds/trigger", json=payload)
    return data["id"], data["registry_id"]


async def _deploy_async(
    tarball_path: Path,
    no_cache: bool,
    plan: _DeployPlan,
    platform: PlatformClient,
    console: HUDConsole,
    env_dir: Path | None = None,
) -> _DeployResult:
    """Narrate the library-owned exchange, stream logs, and save the link."""
    console.progress_message("Getting upload URL...")
    step_start = time.time()

    upload = await platform.apost("/builds/upload-url")
    upload_url, reserved_id = upload["upload_url"], upload["build_id"]

    console.success(f"Got upload URL [{time.time() - step_start:.1f}s]")
    console.info(f"Build ID: {reserved_id}")

    console.progress_message("Uploading build context...")
    step_start = time.time()

    await _upload_context(upload_url, tarball_path)
    console.success(f"Upload complete [{time.time() - step_start:.1f}s]")
    console.progress_message("Triggering build...")
    step_start = time.time()
    build_id, registry_id = await _trigger_build(
        platform,
        build_id=reserved_id,
        plan=plan,
        no_cache=no_cache,
    )
    if registry_id and plan.save_link:
        changes = DirectoryLink(registry_id=UUID(registry_id))
        if plan.dotenv_consent is not None:
            changes.sync_env = {UUID(registry_id): plan.dotenv_consent}
        plan.state.update(changes)

    console.success(f"Build triggered [{time.time() - step_start:.1f}s]")
    console.info(f"Build ID: {build_id}")
    console.info("")

    console.section_title("Build Logs")
    status_data = await wait_for_build(platform, build_id, console)
    final_status = status_data["status"]

    return _DeployResult(
        success=final_status == "SUCCEEDED",
        details=status_data,
        name=plan.name,
        build_id=build_id,
        registry_id=registry_id,
        status=final_status,
    )


def discover_environments(directory: Path) -> list[Path]:
    """Find immediate child directories that contain a HUD environment."""
    if not directory.is_dir():
        return []
    return [
        child
        for child in sorted(directory.iterdir())
        if child.is_dir()
        and (EnvironmentSource.open(child).is_environment or _compose_recipe(child) is not None)
    ]


def deploy_command(
    directory: str = typer.Argument(".", help="Environment directory or env.py file"),
    all_envs: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Deploy all HUD environments found in directory",
    ),
    env: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--env",
        "-e",
        help="Environment variable (KEY=VALUE, repeatable)",
    ),
    env_file: str | None = typer.Option(
        None,
        "--env-file",
        help="Path to .env file (default: .env in directory)",
    ),
    no_env: bool = typer.Option(
        False,
        "--no-env",
        help="Skip .env file loading for this deploy (does not change saved preference)",
    ),
    build_args: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--build-arg",
        help="Docker build argument (KEY=VALUE, repeatable)",
    ),
    secrets: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--secret",
        help="Docker build secret, e.g. --secret id=GITHUB_TOKEN,env=GITHUB_TOKEN",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable build cache",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
    registry_id: str | None = typer.Option(
        None,
        "--registry-id",
        help="Existing registry ID for rebuilds (advanced)",
        hidden=True,
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help=PROJECT_OPTION_HELP,
    ),
    runtime: str | None = typer.Option(
        None,
        "--runtime",
        help="Persist a registry default runtime for tasks that do not specify one: hud or modal",
    ),
    runtime_config: str | None = typer.Option(
        None,
        "--runtime-config",
        help="Path to a JSON RuntimeConfig for hosted runs",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Write JSON to stdout.", callback=mark_json, is_eager=True
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
) -> None:
    """Deploy HUD environment to the platform.

    Accepts a directory or an env.py file — if a file is given, its parent
    directory is used. The environment name comes from the ``Environment(...)``
    declaration in code. Builds from the local Dockerfile and streams remote
    build logs.

    [not dim]Examples:
        hud deploy
        hud deploy --dry-run --json
        hud deploy --all --json[/not dim]
    """
    directories = (
        discover_environments(Path(directory).resolve()) if all_envs else [Path(directory)]
    )
    if not directories:
        raise CliError("not_found", f"No HUD environments found in {directory}")
    succeeded: list[str] = []
    failed: list[str] = []
    entries: list[dict[str, Any]] = []
    for target in directories:
        try:
            result = deploy_environment(
                directory=str(target),
                env=env,
                env_file=env_file,
                no_env=no_env,
                no_cache=no_cache,
                verbose=verbose,
                registry_id=registry_id,
                project=project,
                build_args=build_args,
                build_secrets=secrets,
                runtime=runtime,
                runtime_config=runtime_config,
                dry_run=dry_run,
            )
            payload = asdict(result)
            success = result.success
            if not dry_run and json_output is not True:
                display_build_summary(
                    status_response=result.details,
                    registry_id=result.registry_id or "",
                    env_name=result.name,
                    console=HUDConsole(),
                )
        except Exception as exc:
            if not all_envs:
                raise
            error = map_exception(exc)
            payload = {"success": False, **error.to_payload()}
            success = False
            HUDConsole().error(f"{target.name}: {error.message}")
        (succeeded if success else failed).append(target.name)
        entries.append({"directory": target.name, **payload})
    if json_output is True:
        emit_json(
            {"succeeded": succeeded, "failed": failed, "dry_run": dry_run, "environments": entries}
            if all_envs
            else entries[0]
        )
    elif dry_run:
        for entry in entries:
            HUDConsole().info(f"Would deploy {entry.get('name', entry['directory'])}")
            if entry.get("dotenv_pending"):
                HUDConsole().info("Uploading .env requires an explicit choice before deployment.")
    if failed:
        raise typer.Exit(1)
