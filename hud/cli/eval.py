"""``hud eval`` — run an agent over a taskset and report the graded job.

Config precedence: CLI arguments > ``.hud_eval.toml`` > defaults.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import tomllib
from enum import StrEnum
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Annotated, Any, assert_never, cast
from uuid import UUID

import typer
from pydantic import AliasChoices, AnyUrl, BaseModel, ConfigDict, Field, UrlConstraints
from rich import box
from rich.table import Table

from hud.agents import resolve_agent_model
from hud.cli import CLI, CliError, parse_key_value
from hud.eval import (
    DaytonaRuntime,
    DockerRuntime,
    HostedRuntime,
    HUDRuntime,
    ModalRuntime,
    Runtime,
    SubprocessRuntime,
    Taskset,
)
from hud.settings import settings
from hud.types import AgentType
from hud.utils.gateway import list_gateway_models
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from hud.eval import Provider, Task
    from hud.utils.gateway import GatewayModelInfo

hud_console = HUDConsole()

_CONFIG_PATH = Path(".hud_eval.toml")


class Placement(StrEnum):
    """Named ``--runtime`` choices; a ``tcp://`` url attaches to a served env instead.

    ``LOCAL`` is composed per row in the command; every other name is a provider
    each row's ``runtime_config`` is handed to as-is.
    """

    LOCAL = "local"
    HUD = "hud"
    HOSTED = "hosted"
    DOCKER = "docker"
    MODAL = "modal"
    DAYTONA = "daytona"


TcpUrl = Annotated[AnyUrl, UrlConstraints(allowed_schemes=["tcp"])]


def _substitute_env(value: Any, mapping: dict[str, Any]) -> Any:
    """Expand ``${VAR}`` placeholders; an unset variable is a config error."""
    if isinstance(value, dict):
        return {key: _substitute_env(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_env(item, mapping) for item in value]
    if not isinstance(value, str):
        return value
    try:
        return Template(value).substitute(mapping)
    except KeyError as exc:
        raise ValueError(f"{_CONFIG_PATH}: ${{{exc.args[0]}}} is not set") from None


class EvalConfig(BaseModel):
    """``[eval]`` settings plus the per-agent ``[claude]``/``[openai]``/... sections."""

    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    agent_type: AgentType | None = Field(
        default=None, validation_alias=AliasChoices("agent_type", "agent")
    )
    model: str | None = None
    task_ids: list[str] | None = None
    all: bool = False
    max_concurrent: int = 30
    max_steps: int = 10
    verbose: bool = False
    very_verbose: bool = False
    auto_respond: bool = False
    group_size: int = 1
    gateway: bool = False
    #: ``LOCAL`` spawns each row's env (Docker for container rows, a subprocess
    #: loading the task source otherwise); other names hand rows to that
    #: provider; a ``tcp://`` url attaches to an already-served env. ``None``
    #: infers from the source: a file on disk runs locally, a platform taskset hosted.
    runtime: Placement | TcpUrl | None = None
    agent_config: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = _CONFIG_PATH) -> EvalConfig:
        if not path.exists():
            return cls()
        fields = {key: value for key, value in settings.model_dump().items() if value is not None}
        mapping: dict[str, Any] = {**os.environ, **fields}
        mapping.update({key.upper(): value for key, value in fields.items()})
        if settings.api_key:
            mapping["HUD_API_KEY"] = settings.api_key
        with path.open("rb") as stream:
            data = _substitute_env(tomllib.load(stream), mapping)
        agent_config = {agent.value: data.pop(agent.value) for agent in AgentType if agent in data}
        eval_section = data.pop("eval", {})
        if data:
            raise ValueError(f"{path}: unknown sections: {', '.join(sorted(data))}")
        remote = eval_section.pop("remote", False)
        if not isinstance(remote, bool):
            raise ValueError(f"{path}: remote must be a boolean")
        if remote:
            eval_section.setdefault("runtime", Placement.HOSTED)
        return cls.model_validate({**eval_section, "agent_config": agent_config})

    def merge(self, overrides: dict[str, Any]) -> EvalConfig:
        """Layer ``overrides`` on this config; agent sections merge one level deep."""
        data = self.model_dump()
        for name, params in overrides.get("agent_config", {}).items():
            data["agent_config"][name] = {**data["agent_config"].get(name, {}), **params}
        return self.model_validate(
            {**data, **{k: v for k, v in overrides.items() if k != "agent_config"}}
        )


def eval_command(
    source: str | None = typer.Argument(None, help="Tasks file (.py, .json) or platform taskset"),
    agent: str | None = typer.Argument(
        None,
        help="Model name (e.g. claude-sonnet-4-6) or agent type (claude, openai, gemini, openai_compatible)",  # noqa: E501
    ),
    all: bool = typer.Option(False, "--all", help="Run all problems instead of just 1"),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run the entire dataset. Shortcut for --all --auto-respond --max-steps 100",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name"),
    config: list[str] | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Agent config: key=value"
    ),
    max_concurrent: int | None = typer.Option(
        None, "--max-concurrent", help="Max concurrent tasks"
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Max steps per task"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    very_verbose: bool = typer.Option(False, "--very-verbose", "-vv", help="Debug logs"),
    auto_respond: bool = typer.Option(
        False,
        "--auto-respond",
        help="Automatically prompt the agent to continue if it does not respond with a tool call",
    ),
    group_size: int | None = typer.Option(None, "--group", "--group-size", help="Runs per task"),
    task_ids: str | None = typer.Option(
        None,
        "--task-ids",
        help="Comma-separated task slugs (or 0-based indices) to run",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts (required in non-interactive terminals).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
    gateway: bool = typer.Option(
        False, "--gateway", "-g", help="Route LLM API calls through HUD Gateway"
    ),
    runtime: str | None = typer.Option(
        None,
        "--runtime",
        help="Placement: local (subprocess/Docker per row), hud (runtime tunnel), hosted (whole "
        "rollout on the platform), docker, modal, daytona, or a tcp:// url. "
        "Default: local for a tasks file; hosted for a platform taskset.",
    ),
    remote: bool = typer.Option(
        False,
        "--remote",
        help="Run the whole rollout on the HUD platform (same as --runtime hosted)",
    ),
) -> dict[str, Any]:
    """Run evaluation on datasets or individual tasks with agents.

    A tasks file (tasks.py, a directory, or JSON/JSONL beside its env source) runs
    locally, each rollout in a fresh subprocess; a platform taskset runs hosted.

    Examples:
        hud eval tasks.py claude-sonnet-4-6
        hud eval tasks.py claude
        hud eval "My Tasks" claude-sonnet-4-6 --full   # Platform taskset, run on the platform
        hud eval tasks.py claude --config max_tokens=32768
        hud eval tasks.py claude --gateway             # Route LLM calls through HUD Gateway
        hud eval tasks.json claude-sonnet-4-6 --runtime hud  # Use the HUD runtime tunnel
        hud eval tasks.json claude-sonnet-4-6 --remote       # Execute the rollout remotely
        hud eval tasks.py claude --yes --json
        hud eval tasks.py claude --dry-run --json
    """
    hud_console.info("Initializing evaluation...")
    cfg = EvalConfig.load()

    if runtime is not None and remote:
        raise ValueError("--runtime and --remote are mutually exclusive placement options")
    overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "source": source,
            "model": model,
            "max_concurrent": max_concurrent,
            "max_steps": max_steps,
            "group_size": group_size,
            "runtime": Placement.HOSTED if remote else runtime,
        }.items()
        if value is not None
    }
    overrides.update(
        {
            key: True
            for key, value in {
                "all": all or full,
                "verbose": verbose,
                "very_verbose": very_verbose,
                "auto_respond": auto_respond or full,
                "gateway": gateway,
            }.items()
            if value
        }
    )
    if full:
        overrides.setdefault("max_steps", 100)
    if agent is not None:
        agent_type, model_id = resolve_agent_model(agent)
        overrides["agent_type"] = agent_type
        if model_id != agent_type.value:
            overrides.setdefault("model", model_id)
    if task_ids is not None:
        overrides["task_ids"] = [t.strip() for t in task_ids.split(",") if t.strip()]
    if config:
        # ``claude.max_tokens=1`` targets one agent; a bare key applies to the selected agent.
        selected = overrides.get("agent_type", cfg.agent_type)
        sections: dict[str, dict[str, Any]] = {}
        for item in config:
            parsed = parse_key_value(item)
            if parsed is None:
                raise ValueError(f"--config expects key=value, got {item!r}")
            key, value = parsed
            section, sep, param = key.partition(".")
            if not sep:
                if selected is None:
                    raise ValueError(
                        f"--config {key}=... needs an agent; pass one or write <agent>.{key}=..."
                    )
                section, param = selected.value, key
            parsed_value: bool | int | float | str = value
            if value.lower() in ("true", "false"):
                parsed_value = value.lower() == "true"
            else:
                for number in (int, float):
                    try:
                        parsed_value = number(value)
                        break
                    except ValueError:
                        continue
            sections.setdefault(AgentType(section).value, {})[param] = parsed_value
        overrides["agent_config"] = sections
    cfg = cfg.merge(overrides)

    if dry_run:
        agent_type = cfg.agent_type
        if cfg.source is None or agent_type is None:
            raise CliError(
                "usage",
                "Dry-run requires an explicit task source and agent (or configured defaults).",
            )
        planned = cfg.runtime or (
            Placement.LOCAL if Path(cfg.source).exists() else Placement.HOSTED
        )
        hud_console.info("--dry-run: no evaluation started")
        return {
            "dry_run": True,
            "action": "eval",
            "source": cfg.source,
            "agent": agent_type.value,
            "model": cfg.model,
            "runtime": planned,
            "remote": planned is Placement.HOSTED,
            "all": cfg.all,
            "max_steps": cfg.max_steps,
            "max_concurrent": cfg.max_concurrent,
            "group_size": cfg.group_size,
            "task_ids": cfg.task_ids,
        }

    if cfg.source is None:
        names = sorted(
            path.name
            for path in Path.cwd().iterdir()
            if path.suffix in (".json", ".jsonl") and not path.name.startswith(".")
        )
        if not names:
            raise FileNotFoundError("No task JSON or JSONL files found in current directory")
        chosen = names[0] if len(names) == 1 else hud_console.select("Select a tasks file", names)
        cfg = cfg.merge({"source": chosen})
        hud_console.success(f"Selected: {cfg.source}")

    if cfg.agent_type is None:
        # Pick the agent type, then a current catalog model of that type, newest first.
        picked_type = AgentType(
            hud_console.select(
                "Select an agent:", choices=[agent.value for agent in AgentType], default=0
            )
        )
        picked: dict[str, Any] = {"agent_type": picked_type}
        if settings.api_key:
            models = sorted(
                (
                    catalog_model
                    for catalog_model in list_gateway_models()
                    if catalog_model.sdk_agent_type == picked_type.value
                    and catalog_model.deprecated_at is None
                    and catalog_model.model_name
                ),
                key=lambda catalog_model: catalog_model.recency,
                reverse=True,
            )
            if not models:
                raise ValueError(f"The model catalog has no {picked_type.value} models.")
            picked_model = cast(
                "GatewayModelInfo",
                hud_console.select(
                    "Select a model:",
                    choices=[
                        {"name": f"{m.name or m.model_name} ({m.model_name})", "value": m}
                        for m in models
                    ],
                    default=0,
                ),
            )
            picked["model"] = picked_model.model_name
            if picked_model.name:
                picked["agent_config"] = {picked_type.value: {"model_name": picked_model.name}}
        else:
            hud_console.info(
                f"Using the {picked_type.value} agent's default model "
                f"({picked_type.config_cls().model}); set HUD_API_KEY to pick from the catalog."
            )
        cfg = cfg.merge(picked)

    agent_type, source = cfg.agent_type, cfg.source
    assert agent_type is not None and source is not None
    # A file on disk spawns locally; a platform taskset runs on the platform.
    if cfg.runtime is None:
        is_file = Path(source).exists()
        cfg = cfg.merge({"runtime": Placement.LOCAL if is_file else Placement.HOSTED})
    assert cfg.runtime is not None

    if cfg.gateway or cfg.runtime in (Placement.HUD, Placement.HOSTED):
        PlatformClient.from_settings()
    if (
        agent_type == AgentType.OPENAI_COMPATIBLE
        and cfg.model is None
        and "model" not in cfg.agent_config.get("openai_compatible", {})
    ):
        raise ValueError("Model name is required for OpenAI compatible agent; use --model.")

    if Path(source).exists():
        hud_console.info(f"Loading tasks from: {source}")
        taskset = Taskset.from_file(source)
    else:
        hud_console.info(f"Loading platform taskset: {source}")
        taskset = Taskset.from_api(source)
    if not taskset:
        raise ValueError(
            f"No runnable Tasks found in {source}. Define a `hud.Environment` with "
            "`@env.template` and expose Tasks (for example, `t = my_task(arg=...)`)."
        )
    if cfg.task_ids:
        wanted = set(cfg.task_ids)
        taskset = taskset.filter(
            slug
            for index, (slug, task) in enumerate(taskset.items())
            if slug in wanted or task.id in wanted or str(index) in wanted
        )
        if not taskset:
            raise ValueError(f"No tasks matching: {', '.join(cfg.task_ids)}")
        hud_console.info(f"Filtered to {len(taskset)} task(s)")
    elif not cfg.all:
        total = len(taskset)
        taskset = taskset.filter([next(iter(taskset.tasks))])
        if total > 1:
            hud_console.warning(
                f"Running only 1 of {total} tasks (the first). "
                f"Add --full to run all {total}, or --task-ids to pick specific ones."
            )
    hud_console.info(f"Loaded {len(taskset)} task(s)")

    placement: Provider | HostedRuntime
    match cfg.runtime:
        case AnyUrl():
            placement = Runtime(str(cfg.runtime))
        case Placement.LOCAL:
            rows = list(taskset)
            rows.extend(
                task.verifier
                for task in taskset
                if task.verifier is not None and not task.shares_verifier_runtime
            )
            if not Path(source).exists() and any(
                not (
                    task.runtime_config
                    and (task.runtime_config.image or task.runtime_config.compose)
                )
                for task in rows
            ):
                raise ValueError(
                    f"{source} is a platform taskset, so there is no env source to spawn "
                    "locally. Run it with --remote, --runtime hud, or --runtime tcp://host:port."
                )
            docker = DockerRuntime()
            source_path = Path(source).resolve()
            beside = SubprocessRuntime(source_path if source_path.is_dir() else source_path.parent)

            def spawn(task: Task) -> AbstractAsyncContextManager[Runtime]:
                config = task.runtime_config
                if config and (config.image or config.compose):
                    return docker(task)
                if task._env is not None:
                    return SubprocessRuntime(source_path)(task)
                return beside(task)

            placement = spawn
        case Placement.HUD:
            placement = HUDRuntime()
        case Placement.HOSTED:
            placement = HostedRuntime()
        case Placement.DOCKER:
            placement = DockerRuntime()
        case Placement.MODAL:
            placement = ModalRuntime()
        case Placement.DAYTONA:
            placement = DaytonaRuntime()
        case _:
            assert_never(cfg.runtime)

    # The agent's config kwargs: its TOML section, then --model on top.
    agent_kwargs = dict(cfg.agent_config.get(agent_type.value, {}))
    if cfg.model:
        agent_kwargs["model"] = cfg.model
    agent_kwargs["max_steps"] = cfg.max_steps
    if cfg.auto_respond:
        agent_kwargs["auto_respond"] = True
    if cfg.gateway:
        agent_kwargs["gateway"] = True

    table = Table(title="Evaluation Settings", title_style="bold cyan", box=box.ROUNDED)
    table.add_column("Setting", style="yellow")
    table.add_column("Value", style="green")
    table.add_row("source", cfg.source)
    table.add_row("runtime", str(cfg.runtime))
    table.add_row("agent", agent_type.value)
    if cfg.task_ids:
        shown = ", ".join(cfg.task_ids[:5])
        table.add_row("task_ids", shown + ("..." if len(cfg.task_ids) > 5 else ""))
    table.add_row("all", str(cfg.all))
    table.add_row("max_steps", str(cfg.max_steps))
    table.add_row("max_concurrent", str(cfg.max_concurrent))
    if cfg.group_size > 1:
        table.add_row("group_size", str(cfg.group_size))
    for flag in ("auto_respond", "very_verbose", "verbose", "gateway"):
        if getattr(cfg, flag):
            table.add_row(flag, "[bold green]True[/bold green]")
    table.add_row("", "")
    table.add_row(f"[dim]{agent_type.value} config[/dim]", "")
    for name, value in agent_kwargs.items():
        if name in ("max_steps", "auto_respond"):
            continue
        table.add_row(f"  {name}", "****" if name == "api_key" else str(value))
    hud_console.print(table)
    CLI.confirm_or_abort("Proceed?", yes=yes, default=True)

    single_run = len(taskset) == 1 and cfg.group_size == 1
    if cfg.very_verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(message)s")
        logging.getLogger("hud.agents").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    elif cfg.verbose or single_run:
        logging.getLogger("hud.agents").setLevel(logging.INFO)
    if not single_run:
        hud_console.info(
            f"Running evaluation (max_concurrent: {cfg.max_concurrent}, "
            f"group_size: {cfg.group_size})"
        )

    # cls/config_cls are matched unions; the pairing is correct by construction.
    agent_instance = cast("Any", agent_type.cls)(config=agent_type.config_cls(**agent_kwargs))

    started = time.monotonic()
    job = asyncio.run(
        taskset.run(
            agent_instance,
            runtime=placement,
            group=cfg.group_size,
            max_concurrent=cfg.max_concurrent,
        )
    )
    elapsed = time.monotonic() - started
    if job.runs and settings.telemetry_enabled and settings.api_key:
        hud_console.info(f"{settings.hud_web_url}/jobs/{UUID(job.id)}")

    if job.runs:
        errors = set(map(id, job.errors))
        hud_console.print(f"\n[bold]'{cfg.source}' Results[/bold]")
        hud_console.print(f"  [dim]Runs:[/dim] {len(job.runs)}")
        hud_console.print(f"  [dim]Time:[/dim] {elapsed:.1f}s")
        hud_console.print(f"  [dim]Mean reward:[/dim] [green]{job.reward:.3f}[/green]")
        if errors:
            hud_console.print(f"  [dim]Errors:[/dim] [red]{len(errors)}[/red]")
        if len(job.runs) <= 50:
            details = Table(title="Details", show_header=True, header_style="bold")
            details.add_column("#", style="dim", justify="right", width=4)
            details.add_column("Prompt", style="dim", max_width=35)
            details.add_column("Answer", style="dim", max_width=35)
            details.add_column("Reward", justify="right", style="green", width=8)
            for index, run in enumerate(job.runs):
                cells = []
                for text in (run.prompt, run.trace.content):
                    flat = str(text).replace("\n", " ").strip() if text else ""
                    cells.append((flat[:33] + ".." if len(flat) > 35 else flat) or "—")
                details.add_row(
                    str(index),
                    *cells,
                    "[red]error[/red]" if id(run) in errors else f"{run.reward:.3f}",
                )
            hud_console.print(details)
        hud_console.print()

    return {
        "job_id": job.id,
        "source": cfg.source,
        "run_count": len(job.runs),
        "mean_reward": job.reward,
        "error_count": len(job.errors),
        "elapsed_seconds": elapsed,
        "runs": [
            {
                "task_id": run.task_id,
                "slug": run.slug,
                "reward": run.reward,
                "is_error": run.trace.is_error,
                "trace_id": run.trace_id,
            }
            for run in job.runs
        ],
    }
