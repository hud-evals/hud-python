"""Deploy HUD environments to the platform via direct build."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, NotRequired
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import httpx
import typer
import websockets
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, with_config
from typing_extensions import TypedDict
from websockets.exceptions import ConnectionClosed, WebSocketException

from hud.cli import (
    CLI,
    AuthScope,
    CliError,
    DirectoryLink,
    DirectoryState,
    Result,
    parse_key_value,
)
from hud.cli.harbor import ensure_taskset, follow_import, pack_import, upload_and_start
from hud.cli.project import PROJECT_OPTION_HELP, Placement
from hud.cli.sync import RegistryEnvironment
from hud.eval.runtime import RuntimeConfig
from hud.settings import settings
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.naming import normalize_environment_name
from hud.utils.platform import PlatformClient

hud_console = HUDConsole()

SENSITIVE_EXCLUDES = [".git", ".env", ".env.*", "*.env"]
"""Never uploaded. Applied last, so a ``.dockerignore`` negation cannot re-include them."""

DEFAULT_EXCLUDES = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".DS_Store",
    "Thumbs.db",
]
"""Local junk no image needs. Applied first, so ``.dockerignore`` can re-include any of it."""

_UNSEARCHED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
"""Directories skipped when looking for ``Environment(...)`` declarations."""

_REGISTRY_RUNTIMES = ("hud", "modal")
"""Default runtimes a registry accepts (the platform's ``RequestedRuntimeProvider``)."""

HARBOR_TASK_FILE = "task.toml"
"""Marks a Harbor task folder. One that declares no ``Environment(...)`` deploys through a Harbor
import into a taskset, then builds the environment version the import registers."""

_LIVE_BUILD_STATUSES = frozenset({"IN_PROGRESS", "FINALIZING", "MIGRATING", "CANCELLING"})


def _dockerignore_re(pattern: str) -> re.Pattern[str]:
    """Compile one ``.dockerignore`` glob: ``*`` is one segment, ``**`` is any depth."""
    if pattern.startswith("/"):
        pattern, anchored = pattern[1:], True
    else:
        anchored = "/" in pattern
    if pattern in {"", "**"}:
        return re.compile(r"\A.*\Z")
    if not anchored:
        pattern = f"**/{pattern}"
    out = [r"\A"]
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i) and (i == 0 or pattern[i - 1] == "/"):
            after = i + 2
            if after == len(pattern):
                out.append(".*")
                i = after
                continue
            if pattern[after] == "/":
                out.append("(?:.*/)?")
                i = after + 1
                continue
        char = pattern[i]
        out.append("[^/]*" if char == "*" else "[^/]" if char == "?" else re.escape(char))
        i += 1
    out.append(r"\Z")
    return re.compile("".join(out))


class _BuildUpload(TypedDict):
    build_id: str
    upload_url: str


class _Build(BaseModel):
    id: str
    registry_id: UUID


@with_config(ConfigDict(extra="allow"))
class _ManifestTask(TypedDict):
    id: str


@with_config(ConfigDict(extra="allow"))
class _ManifestCapability(TypedDict):
    name: str


@with_config(ConfigDict(extra="allow"))
class _Manifest(TypedDict):
    tasks: NotRequired[list[_ManifestTask] | None]
    capabilities: NotRequired[list[_ManifestCapability] | None]


@with_config(ConfigDict(extra="allow"))
class _BuildStatus(TypedDict):
    status: str
    manifest: NotRequired[_Manifest | None]


@dataclass(frozen=True)
class _DeployResult:
    """JSON document for one deploy; a dry run fills the plan fields, a build the build fields."""

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
    # Set for a Harbor task folder: the import that registered it and the task it holds.
    import_id: str | None = None
    taskset_id: str | None = None
    task_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


async def deploy_command(
    directory: str = typer.Argument(".", help="Environment directory or env.py file"),
    name: str | None = typer.Option(
        None,
        "--name",
        help="Environment name when the tree declares more than one.",
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
        help="Upsert registry secrets from this file. Keys omitted from the file are kept.",
    ),
    no_env: bool = typer.Option(
        False,
        "--no-env",
        help="Skip seeding registry secrets from a local .env on first deploy.",
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
    taskset: str | None = typer.Option(
        None,
        "--taskset",
        "-t",
        help=(
            "Harbor task folder only: taskset name or ID to add the task to; a new name "
            "creates it. Defaults to the current directory's taskset, then a taskset named "
            "after the folder."
        ),
    ),
    upload_only: bool = typer.Option(
        False,
        "--upload-only",
        help="Harbor task folder only: add the task and upload its folder without building it.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts (required in non-interactive terminals).",
    ),
) -> Any:
    """Deploy HUD environment to the platform.

    Accepts a directory or an env.py file. The environment name comes from the
    ``Environment(...)`` declaration in code; pass ``--name`` when the tree
    declares more than one. A file supplies the declaration by itself while its
    parent directory is still the uploaded tree, so a nested source tree with
    its own declaration cannot make the name ambiguous. Streams the remote
    build. Compose and other run settings belong in ``--runtime-config`` or on
    the task, not inferred from filenames.

    A Harbor task folder (one holding ``task.toml`` and no ``Environment(...)``)
    is imported as it is into a taskset, as ``hud harbor import`` imports it:
    the folder becomes a task of the taskset and a version of the environment
    named after the folder. Then HUD adapts and builds that version on Modal,
    and the task runs the adapted task. ``--upload-only`` stops after the
    import. It takes no ``--name``, ``--registry-id``, ``--runtime-config``,
    ``--build-arg``, ``--secret``, or ``--runtime`` other than ``modal``.

    [not dim]Examples:
        hud deploy
        hud deploy --name judge
        hud deploy tasks/hello-world --taskset tb2
        hud deploy tasks/hello-world --upload-only
        hud deploy --yes --json
        hud deploy --dry-run --json[/not dim]
    """
    platform = PlatformClient.from_settings()
    source = Path(directory).expanduser().resolve()  # noqa: ASYNC240
    if source.is_file():
        env_dir = source.parent
        modules = [source]
    else:
        env_dir = source
        modules = []
        for dirpath, dirnames, filenames in os.walk(env_dir):
            dirnames[:] = [d for d in dirnames if d not in _UNSEARCHED_DIRS]
            modules.extend(Path(dirpath) / f for f in filenames if f.endswith(".py"))
    names: set[str] = set()
    for path in modules:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            callee = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if callee != "Environment":
                continue
            name_node = (
                node.args[0]
                if node.args
                else next((kw.value for kw in node.keywords if kw.arg == "name"), None)
            )
            if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
                names.add(name_node.value)
    found = ", ".join(sorted(names))
    harbor_folder = not names and source.is_dir() and (source / HARBOR_TASK_FILE).is_file()
    if harbor_folder:
        refused = [
            flag
            for flag, given in (
                ("--name", name is not None),
                ("--registry-id", registry_id is not None),
                ("--runtime-config", runtime_config is not None),
                ("--build-arg", bool(build_args)),
                ("--secret", bool(secrets)),
                ("--runtime", runtime is not None and runtime.lower() != "modal"),
            )
            if given
        ]
        if refused:
            raise CliError(
                "usage",
                f"A Harbor task folder deploys without {', '.join(refused)}.",
                suggestion="HUD builds a Harbor task folder on Modal from the folder itself.",
                input={"directory": directory},
            )
        unbuilt = [
            flag
            for flag, given in (
                ("--env", bool(env)),
                ("--env-file", env_file is not None),
                ("--no-cache", no_cache),
            )
            if given
        ]
        if upload_only and unbuilt:
            raise CliError(
                "usage",
                f"--upload-only builds nothing, so it takes no {', '.join(unbuilt)}.",
                suggestion="Pass them to the deploy that builds the folder.",
                input={"directory": directory},
            )
        return await _deploy_harbor_folder(
            platform,
            source,
            taskset=taskset,
            project=project,
            env_vars=_read_env_vars(Path(env_file) if env_file else None, env),
            no_cache=no_cache,
            upload_only=upload_only,
            dry_run=dry_run,
            yes=yes,
        )
    if taskset is not None or upload_only:
        raise CliError(
            "usage",
            "--taskset and --upload-only deploy a Harbor task folder only.",
            suggestion="Add an environment's tasks to a taskset with hud sync.",
            input={"directory": directory},
        )
    if not names:
        raise CliError(
            "usage",
            f"No environment found in {source}.",
            suggestion="Declare the environment with Environment(name=...) in a .py file.",
        )
    if name is not None:
        if name not in names:
            raise ValueError(f"No environment named {name!r} in {source}. Found: {found}.")
    elif len(names) > 1:
        raise ValueError(f"Multiple environments in {source}: {found}. Pass --name to choose one.")
    else:
        name = names.pop()
    if registry_id is not None:
        try:
            registered = platform.get(f"/registry/{registry_id}", RegistryEnvironment).name
        except HudRequestError as exc:
            raise CliError.from_http(
                exc, resource="Environment", input={"registry_id": registry_id}
            ) from exc
        if normalize_environment_name(name) != registered:
            raise CliError(
                "usage",
                f"Code declares Environment({name!r}) but --registry-id targets {registered!r}.",
                suggestion="Rename the environment in code, or drop --registry-id to deploy "
                "by name.",
                input={"registry_id": registry_id, "name": name},
            )
    state = DirectoryState(AuthScope.resolve(platform), env_dir)
    link = state.load()
    placement = Placement.resolve(platform, link, flag=project)
    placement.require_writable()
    first_deploy = link.registry_id is None and registry_id is None
    env_file_path = Path(env_file) if env_file else None
    dotenv_pending = (
        (env_dir / ".env").is_file() and not no_env and env_file_path is None and first_deploy
    )
    resolved_runtime = runtime.lower() if runtime is not None else None
    if resolved_runtime is not None and resolved_runtime not in _REGISTRY_RUNTIMES:
        raise CliError(
            "usage",
            f"Unknown runtime {runtime!r}. Choose one of: {', '.join(_REGISTRY_RUNTIMES)}.",
            input={"runtime": runtime},
        )
    config_path = Path(runtime_config).expanduser() if runtime_config else None  # noqa: ASYNC240
    if config_path is None:
        resolved_runtime_config = None
    else:
        resolved_runtime_config = RuntimeConfig.model_validate(
            json.loads(config_path.read_text(encoding="utf-8")),
            context={"base_path": config_path.parent},
        ).model_dump(mode="json", exclude_unset=True)
        if not resolved_runtime_config:
            raise ValueError("--runtime-config must set at least one field.")
    parsed_build_args: dict[str, str] = {}
    for flag in build_args or []:
        parsed = parse_key_value(flag)
        if parsed is None:
            raise ValueError(f"Invalid --build-arg format: {flag} (expected KEY=VALUE)")
        parsed_build_args[parsed[0]] = parsed[1]

    hud_console.info(f"Environment name: {name}")
    if env_file:
        hud_console.info(f"Loading environment variables from {env_file}")
    if dotenv_pending and not dry_run:
        if not sys.stdin.isatty():
            raise CliError(
                "usage",
                "Choose whether to seed registry secrets from .env before deploying.",
                suggestion="Pass --env-file .env to upsert them, or --no-env to skip.",
            )
        if hud_console.confirm(
            "Seed registry secrets from .env? (upsert; encrypted at rest)",
            default=False,
        ):
            env_file_path = env_dir / ".env"

    env_vars = _read_env_vars(env_file_path, env)

    build_secrets: dict[str, str] = {}
    for secret_spec in secrets or []:
        spec: dict[str, str] = {}
        for part in secret_spec.split(","):
            key, sep, value = part.partition("=")
            if sep:
                spec[key.strip()] = value.strip()
        secret_id = spec.get("id")
        if not secret_id:
            raise ValueError(f"Invalid --secret format: {secret_spec} (missing id=)")
        if "env" in spec:
            env_name = spec["env"]
            value = os.environ.get(env_name)
            if value is None:
                raise ValueError(
                    f"Secret '{secret_id}': environment variable '{env_name}' is not set"
                )
            build_secrets[secret_id] = value
        elif "src" in spec:
            src_path = env_dir / Path(spec["src"]).expanduser()  # noqa: ASYNC240
            try:
                build_secrets[secret_id] = src_path.read_text(encoding="utf-8")
            except OSError as e:
                raise ValueError(f"Secret '{secret_id}': failed to read {src_path}: {e}") from e
        else:
            raise ValueError(f"Invalid --secret format: {secret_spec} (need env= or src=)")

    if not dotenv_pending and not no_env and env_file is None and (env_dir / ".env").is_file():
        hud_console.dim_info("Registry secrets:", "kept. Pass --env-file .env to upsert.")
    if dry_run:
        hud_console.info(f"Would deploy {name}")
        if dotenv_pending:
            hud_console.info(
                "Seeding registry secrets from .env requires --env-file or confirmation."
            )
        return asdict(
            _DeployResult(
                success=True,
                name=name,
                registry_id=registry_id,
                dry_run=True,
                runtime=resolved_runtime,
                env_var_keys=sorted(env_vars),
                build_arg_keys=sorted(parsed_build_args),
                dotenv_pending=dotenv_pending,
            )
        )

    CLI.confirm_or_abort(f"Deploy {name}?", yes=yes, default=True)
    hud_console.progress_message("Creating build context tarball...")
    dockerignore = env_dir / ".dockerignore"
    rules: list[tuple[re.Pattern[str], bool, bool]] = []
    for line in (
        *DEFAULT_EXCLUDES,
        *(dockerignore.read_text(encoding="utf-8").splitlines() if dockerignore.is_file() else ()),
        *SENSITIVE_EXCLUDES,
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        body = line.removeprefix("!")
        rules.append((_dockerignore_re(body.rstrip("/")), negate, body.endswith("/")))

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        path = info.name.strip("/")
        parts = path.split("/")
        ignored = False
        for regex, negate, dir_only in rules:
            if ((not dir_only or info.isdir()) and regex.fullmatch(path)) or any(
                regex.fullmatch("/".join(parts[:depth])) for depth in range(1, len(parts))
            ):
                ignored = not negate
        return None if ignored else info

    fd, temp_name = tempfile.mkstemp(suffix=".tar.gz", prefix="hud-build-context-")
    os.close(fd)
    tarball = Path(temp_name)
    try:
        with tarfile.open(tarball, "w:gz") as tar:
            for child in env_dir.iterdir():
                tar.add(child, arcname=child.name, filter=keep)
    except BaseException:
        tarball.unlink(missing_ok=True)  # noqa: ASYNC240
        raise
    hud_console.success(f"Created tarball: {tarball.stat().st_size} bytes")  # noqa: ASYNC240
    try:
        hud_console.progress_message("Getting upload URL...")
        started = time.time()
        upload = await platform.apost("/builds/upload-url", _BuildUpload)
        hud_console.success(f"Got upload URL [{time.time() - started:.1f}s]")
        hud_console.info(f"Build ID: {upload['build_id']}")

        hud_console.progress_message("Uploading build context...")
        started = time.time()
        content = await asyncio.to_thread(tarball.read_bytes)
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.put(
                upload["upload_url"],
                content=content,
                headers={"Content-Type": "application/gzip"},
            )
            response.raise_for_status()
        hud_console.success(f"Upload complete [{time.time() - started:.1f}s]")

        optional: dict[str, Any] = {
            "registry_id": registry_id,
            "project_id": placement.project_id,
            "runtime_provider": resolved_runtime,
            "runtime_config": resolved_runtime_config,
            "environment_variables": env_vars,
            "build_args": parsed_build_args,
            "build_secrets": build_secrets,
        }
        hud_console.progress_message("Triggering build...")
        started = time.time()
        build = await platform.apost(
            "/builds/trigger",
            _Build,
            json={
                "source": "direct",
                "build_id": upload["build_id"],
                "name": name,
                "no_cache": no_cache,
                **{key: value for key, value in optional.items() if value},
            },
        )
        hud_console.success(f"Build triggered [{time.time() - started:.1f}s]")
    finally:
        tarball.unlink(missing_ok=True)  # noqa: ASYNC240
    if registry_id is None and project is None:
        state.update(DirectoryLink(registry_id=build.registry_id))
    hud_console.info(f"Build ID: {build.id}")
    hud_console.info("")
    await _stream_build_logs(platform, build.id)
    return _concluded(await _built(platform, name, build.id, str(build.registry_id)))


def _read_env_vars(env_file: Path | None, env: list[str] | None) -> dict[str, str]:
    """The variables in ``env_file``, with each ``--env KEY=VALUE`` over them."""
    env_vars: dict[str, str] = {}
    if env_file is not None:
        if not env_file.is_file():
            raise FileNotFoundError(f"Env file not found: {env_file}")
        env_vars = {
            key: value
            for key, value in dotenv_values(env_file, interpolate=False).items()
            if value is not None
        }
    for flag in env or []:
        parsed = parse_key_value(flag)
        if parsed is None:
            raise ValueError(f"Invalid --env format: {flag} (expected KEY=VALUE)")
        env_vars[parsed[0]] = parsed[1]
    return env_vars


async def _deploy_harbor_folder(
    platform: PlatformClient,
    folder: Path,
    *,
    taskset: str | None,
    project: str | None,
    env_vars: dict[str, str],
    no_cache: bool,
    upload_only: bool,
    dry_run: bool,
    yes: bool,
) -> Any:
    """Import ``folder`` into a taskset, then build the environment version the import registered.

    The import registers unchanged content as the version that already holds it, so a version
    that is building is followed, and one whose build ended is reported without building again.
    """
    name = folder.name
    hud_console.info(f"Harbor task folder: {name}")
    link = DirectoryState(AuthScope.resolve(platform)).load()
    with tempfile.TemporaryDirectory(prefix="hud-harbor-") as workdir:
        bundle = pack_import(
            platform, link, [folder], taskset=taskset, project=project, workdir=Path(workdir)
        )
        if dry_run:
            hud_console.info(
                f"Would import {name} into {bundle.target.name}"
                + ("" if upload_only else " and build it")
            )
            return asdict(
                _DeployResult(
                    success=True,
                    name=name,
                    dry_run=True,
                    runtime="modal",
                    env_var_keys=sorted(env_vars),
                    details=bundle.plan(),
                )
            )
        CLI.confirm_or_abort(f"Deploy {name}?", yes=yes, default=True)
        taskset_id = ensure_taskset(platform, bundle.target)
        imported = upload_and_start(platform, bundle.path, taskset_id)

    imported, report = follow_import(platform, imported)
    task = next(
        (
            task
            for task in (report.tasks if report is not None else [])
            if task.source == "bundle" and task.path == name
        ),
        None,
    )
    if (
        imported.status != "completed"
        or task is None
        or task.task_id is None
        or task.build_id is None
        or task.environment_id is None
    ):
        reason = f": {task.reason}" if task is not None and task.reason else ""
        hud_console.error(f"The import did not add {name} to the taskset{reason}")
        return Result(
            asdict(
                _DeployResult(
                    success=False,
                    name=name,
                    status=imported.status,
                    import_id=imported.id,
                    taskset_id=taskset_id,
                    details={"task": task.model_dump(mode="json")} if task is not None else {},
                )
            )
        )

    status = await platform.aget(f"/builds/{task.build_id}/status", _BuildStatus)
    if upload_only:
        hud_console.success(f"Uploaded {name} as version {task.build_version} of its environment")
        if status["status"] == "UPLOADED":
            hud_console.hint(f"Build it with: hud deploy {folder} --taskset {bundle.target.name}")
        return asdict(
            _DeployResult(
                success=True,
                name=name,
                build_id=task.build_id,
                registry_id=task.environment_id,
                status=status["status"],
                import_id=imported.id,
                taskset_id=taskset_id,
                task_id=task.task_id,
                details=dict(status),
            )
        )

    if status["status"] == "UPLOADED":
        hud_console.progress_message("Triggering build...")
        await platform.apost(
            "/builds/trigger",
            _Build,
            json={
                "source": "direct",
                "registry_id": task.environment_id,
                "build_id": task.build_id,
                "no_cache": no_cache,
                **({"environment_variables": env_vars} if env_vars else {}),
            },
        )
        hud_console.success("Build triggered")
    elif status["status"] not in _LIVE_BUILD_STATUSES:
        hud_console.info(
            f"Version {task.build_version} already holds this content and its build ended "
            f"{status['status']}; change the folder to build a new version."
        )
        if env_vars or no_cache:
            hud_console.warning("--env, --env-file, and --no-cache apply to a new build only.")
    if status["status"] == "UPLOADED" or status["status"] in _LIVE_BUILD_STATUSES:
        hud_console.info(f"Build ID: {task.build_id}")
        hud_console.info("")
        await _stream_build_logs(platform, task.build_id)
    result = await _built(platform, name, task.build_id, task.environment_id)
    return _concluded(
        replace(result, import_id=imported.id, taskset_id=taskset_id, task_id=task.task_id)
    )


async def _stream_build_logs(platform: PlatformClient, build_id: str) -> None:
    """Print the build's log as it streams, until the build ends or the stream closes."""
    hud_console.section_title("Build Logs")
    http_url = platform.url(f"/builds/{build_id}/logs", params={"api_key": platform.api_key})
    parts = urlsplit(http_url)
    ws_url = urlunsplit(
        (
            "wss" if parts.scheme == "https" else "ws",
            parts.netloc,
            parts.path,
            parts.query,
            parts.fragment,
        )
    )
    try:
        hud_console.info("Connecting to build logs stream...")
        async with websockets.connect(ws_url, ping_interval=30, ping_timeout=10) as websocket:
            async for message in websocket:
                frame = json.loads(message)
                match frame["type"]:
                    case "status":
                        hud_console.info(frame["message"])
                    case "status_update" if frame.get("status") != "IN_PROGRESS":
                        hud_console.info(f"Build status: {frame.get('status', '')}")
                    case "log" if frame.get("message"):
                        timestamp_ms = frame.get("timestamp")
                        prefix = (
                            f"[{datetime.fromtimestamp(timestamp_ms / 1000):%H:%M:%S}] "
                            if timestamp_ms is not None
                            else ""
                        )
                        hud_console.info(f"{prefix}{frame['message'].rstrip()}")
                    case "complete":
                        hud_console.info(frame["message"])
                        break
                    case "error":
                        hud_console.error(f"Build error: {frame['error']}")
                        break
    except ConnectionClosed as e:
        if e.code == 4003:
            hud_console.error(f"Access denied: {e.reason}")
        else:
            hud_console.warning(f"Log stream closed: {e.reason}")
    except (OSError, WebSocketException) as e:
        hud_console.warning(f"Log stream unavailable: {e}")


async def _built(
    platform: PlatformClient, name: str, build_id: str, registry_id: str
) -> _DeployResult:
    """Wait for the build to end, then print and return how it ended."""
    status = await platform.aget(f"/builds/{build_id}/status", _BuildStatus)
    while status["status"] in _LIVE_BUILD_STATUSES:
        await asyncio.sleep(5)
        status = await platform.aget(f"/builds/{build_id}/status", _BuildStatus)

    manifest = status.get("manifest") or {}
    tasks = [task["id"] for task in manifest.get("tasks") or []]
    capabilities = [capability["name"] for capability in manifest.get("capabilities") or []]
    summary: dict[str, str | int | float] = {
        "Environment": name,
        "Status": status["status"],
        "Version": status.get("version") or "unknown",
    }
    if uri := status.get("uri"):
        summary["Image"] = uri
    if error_message := status.get("error_message"):
        summary["Error"] = error_message
    if tasks:
        summary["Tasks"] = ", ".join(tasks)
    if capabilities:
        summary["Capabilities"] = ", ".join(capabilities)
    hud_console.section_title("Build")
    hud_console.key_value_table(summary)
    hud_console.link(f"{settings.hud_web_url}/environments/{registry_id}")
    return _DeployResult(
        success=status["status"] == "SUCCEEDED",
        details=dict(status),
        name=name,
        build_id=build_id,
        registry_id=registry_id,
        status=status["status"],
    )


def _concluded(result: _DeployResult) -> Any:
    """The result's JSON document; a deploy that did not succeed exits non-zero."""
    payload = asdict(result)
    return Result(payload) if not result.success else payload
