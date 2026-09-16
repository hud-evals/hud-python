"""``hud eval`` — run an agent over a taskset and report the graded job.

Config precedence: CLI arguments > ``.hud_eval.toml`` > defaults.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, cast

import typer
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from rich import box
from rich.table import Table

from hud.agents import resolve_agent_model
from hud.cli import CLI, CliError, parse_key_value
from hud.eval import DockerRuntime, HostedRuntime, HUDRuntime, Runtime, SubprocessRuntime, Taskset
from hud.settings import settings
from hud.types import AgentType
from hud.utils.gateway import build_model_client, normalize_gateway_model_id
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient, canonical_record_id

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from hud.agents.base import Agent
    from hud.eval import Job, Provider, Task

hud_console = HUDConsole()

_CONFIG_PATH = Path(".hud_eval.toml")
_PLACEMENTS = ("local", "hud", "hosted")
_SECRET_MARKERS = ("key", "secret", "token", "password")


@dataclass(frozen=True)
class AgentPreset:
    """An interactive-picker entry: agent type, model, and its display name."""

    name: str
    agent_type: AgentType
    model: str
    model_name: str | None = None

    def overrides(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {"agent_type": self.agent_type, "model": self.model}
        if self.model_name is not None:
            overrides["agent_config"] = {self.agent_type.value: {"model_name": self.model_name}}
        return overrides


_AGENT_PRESETS: list[AgentPreset] = [
    AgentPreset("Claude Sonnet 4.6", AgentType.CLAUDE, "claude-sonnet-4-6"),
    AgentPreset("Claude Opus 4.8", AgentType.CLAUDE, "claude-opus-4-8"),
    AgentPreset("GPT-5.6", AgentType.OPENAI, "gpt-5.6"),
    AgentPreset("GPT-5.5", AgentType.OPENAI, "gpt-5.5"),
    AgentPreset("Gemini 3.1 Pro (Preview)", AgentType.GEMINI, "gemini-3.1-pro-preview"),
    AgentPreset(
        "Grok 4-1 Fast (xAI)", AgentType.OPENAI_COMPATIBLE, "grok-4-1-fast", "Grok 4-1 Fast"
    ),
    AgentPreset("GLM 5.2 (Z.ai)", AgentType.OPENAI_COMPATIBLE, "z-ai/glm-5.2", "GLM 5.2"),
    AgentPreset(
        "Kimi K2.6 (Moonshot)", AgentType.OPENAI_COMPATIBLE, "moonshotai/kimi-k2.6", "Kimi K2.6"
    ),
    AgentPreset("MiniMax M3", AgentType.OPENAI_COMPATIBLE, "MiniMax-M3", "MiniMax M3"),
]


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


def _env_mapping() -> dict[str, Any]:
    """Process environment plus ``hud.settings`` fields in either case."""
    fields = settings.model_dump()
    mapping: dict[str, Any] = {**os.environ, **fields}
    mapping.update({key.upper(): value for key, value in fields.items()})
    if settings.api_key:
        mapping["HUD_API_KEY"] = settings.api_key
    return mapping


def _parse_config_value(value: str) -> bool | int | float | str:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _agent_config_updates(
    items: list[str], agent_type: AgentType | None
) -> dict[str, dict[str, Any]]:
    """Parse ``--config key=value`` into per-agent sections.

    ``claude.max_tokens=1`` targets one agent explicitly; a bare key applies
    to the selected agent.
    """
    updates: dict[str, dict[str, Any]] = {}
    for item in items:
        parsed = parse_key_value(item)
        if parsed is None:
            raise ValueError(f"--config expects key=value, got {item!r}")
        key, value = parsed
        section, sep, param = key.partition(".")
        if not sep:
            if agent_type is None:
                raise ValueError(
                    f"--config {key}=... needs an agent; pass one or write <agent>.{key}=..."
                )
            section, param = agent_type.value, key
        updates.setdefault(AgentType(section).value, {})[param] = _parse_config_value(value)
    return updates


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
    #: Placement: ``local`` (spawn each row's env — Docker for container rows,
    #: a subprocess serving the bound env's source otherwise), ``hud`` (runtime
    #: tunnel, agent loop here), ``hosted`` (whole rollout on the platform), or
    #: a ``tcp://`` url of an already-served env. ``None`` infers from the
    #: source: a file on disk runs locally, a platform taskset hosted.
    runtime: str | None = None
    agent_config: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("runtime")
    @classmethod
    def _known_placement(cls, value: str | None) -> str | None:
        if value is None or value in _PLACEMENTS or value.startswith("tcp://"):
            return value
        raise ValueError(f"Unknown runtime {value!r}. Use local, hud, hosted, or a tcp:// url.")

    @classmethod
    def load(cls, path: Path = _CONFIG_PATH) -> EvalConfig:
        if not path.exists():
            return cls()
        with path.open("rb") as stream:
            data = _substitute_env(tomllib.load(stream), _env_mapping())
        agent_config = {agent.value: data.pop(agent.value) for agent in AgentType if agent in data}
        eval_section = data.pop("eval", {})
        if data:
            raise ValueError(f"{path}: unknown sections: {', '.join(sorted(data))}")
        return cls.model_validate({**eval_section, "agent_config": agent_config})

    def merge(self, overrides: dict[str, Any]) -> EvalConfig:
        """Layer ``overrides`` on this config; agent sections merge one level deep."""
        data = self.model_dump()
        for name, params in overrides.get("agent_config", {}).items():
            data["agent_config"][name] = {**data["agent_config"].get(name, {}), **params}
        return self.model_validate(
            {**data, **{k: v for k, v in overrides.items() if k != "agent_config"}}
        )

    @property
    def source_is_file(self) -> bool:
        return self.source is not None and Path(self.source).exists()

    def with_placement(self) -> EvalConfig:
        """Pin ``runtime``: a local file spawns locally, a platform taskset runs hosted."""
        if self.runtime is None:
            return self.model_copy(update={"runtime": "local" if self.source_is_file else "hosted"})
        if self.runtime == "local" and not self.source_is_file:
            raise ValueError(
                f"--runtime local needs a local env source, but {self.source!r} is a "
                "platform taskset with no env source on disk. Run it on the platform "
                "by omitting --runtime or passing --remote, export it first "
                "(hud sync tasks <name> --export tasks.json) and run that file, "
                "or attach to a served env with --runtime tcp://host:port."
            )
        return self

    def require_credentials(self) -> None:
        if self.gateway or self.runtime in ("hud", "hosted"):
            PlatformClient.from_settings()
        if (
            self.agent_type == AgentType.OPENAI_COMPATIBLE
            and self.model is None
            and "model" not in self.agent_config.get("openai_compatible", {})
        ):
            raise ValueError("Model name is required for OpenAI compatible agent; use --model.")

    def agent_kwargs(self) -> dict[str, Any]:
        """The agent's config kwargs: its TOML section, then ``--model`` on top."""
        assert self.agent_type is not None
        kwargs = dict(self.agent_config.get(self.agent_type.value, {}))
        if self.model:
            kwargs["model"] = self.model
        if isinstance(kwargs.get("model"), str):
            kwargs["model"] = normalize_gateway_model_id(kwargs["model"])
        kwargs["max_steps"] = self.max_steps
        if self.auto_respond:
            kwargs["auto_respond"] = True
        return kwargs

    def display(self) -> None:
        table = Table(title="Evaluation Settings", title_style="bold cyan", box=box.ROUNDED)
        table.add_column("Setting", style="yellow")
        table.add_column("Value", style="green")
        table.add_row("source", self.source or "-")
        table.add_row("runtime", self.runtime or "-")
        table.add_row("agent", self.agent_type.value if self.agent_type else "-")
        if self.task_ids:
            shown = ", ".join(self.task_ids[:5])
            table.add_row("task_ids", shown + ("..." if len(self.task_ids) > 5 else ""))
        table.add_row("all", str(self.all))
        table.add_row("max_steps", str(self.max_steps))
        table.add_row("max_concurrent", str(self.max_concurrent))
        if self.group_size > 1:
            table.add_row("group_size", str(self.group_size))
        for flag in ("auto_respond", "very_verbose", "verbose", "gateway"):
            if getattr(self, flag):
                table.add_row(flag, "[bold green]True[/bold green]")
        if self.agent_type is not None:
            table.add_row("", "")
            table.add_row(f"[dim]{self.agent_type.value} config[/dim]", "")
            for name, value in self.agent_kwargs().items():
                if name in ("max_steps", "auto_respond"):
                    continue
                shown = str(value)
                if any(marker in name for marker in _SECRET_MARKERS) and shown:
                    shown = f"{shown[:4]}****" if len(shown) > 4 else "****"
                table.add_row(f"  {name}", shown)
        hud_console.print(table)


def _build_agent(cfg: EvalConfig) -> Agent:
    """Construct the agent from the eval config.

    The CLI prefers a provider's own key over the HUD gateway unless
    ``--gateway`` forces routing. ``openai_compatible`` has no provider of
    its own: its agent uses the gateway unless ``api_key``/``base_url`` are
    configured. Hosted rollouts leave the client unset so the platform
    rebuilds it remotely.
    """
    assert cfg.agent_type is not None
    config = cfg.agent_type.config_cls(**cfg.agent_kwargs())
    if (
        cfg.runtime != "hosted"
        and config.model_client is None
        and cfg.agent_type != AgentType.OPENAI_COMPATIBLE
    ):
        config.model_client = build_model_client(
            cfg.agent_type.gateway_provider, model=config.model, prefer_provider=not cfg.gateway
        )
    # cls/config_cls are matched unions; the pairing is correct by construction.
    return cast("Any", cfg.agent_type.cls)(config=config)


def _local_placement() -> Provider:
    """Spawn each row's own substrate from what the row declares."""
    docker = DockerRuntime()

    def spawn(task: Task) -> AbstractAsyncContextManager[Runtime]:
        config = task.runtime_config
        if config is not None and (config.image is not None or config.compose is not None):
            return docker(task)
        if task._env is None:
            raise ValueError(
                "no placement: these rows have no bound Environment. Pass a Python "
                "tasks/env module, or use --remote / --runtime hud / a tcp:// url."
            )
        return SubprocessRuntime(task._env)(task)

    return spawn


def _placement(cfg: EvalConfig) -> Provider | HostedRuntime:
    match cfg.runtime:
        case "hosted":
            return HostedRuntime()
        case "hud":
            return HUDRuntime()
        case "local":
            return _local_placement()
        case url:
            assert url is not None and url.startswith("tcp://")
            return Runtime(url)


def _load_taskset(cfg: EvalConfig) -> Taskset:
    assert cfg.source is not None
    if cfg.source_is_file:
        hud_console.info(f"Loading tasks from: {cfg.source}")
        taskset = Taskset.from_file(cfg.source)
    else:
        hud_console.info(f"Loading platform taskset: {cfg.source}")
        taskset = Taskset.from_api(cfg.source)
    if not taskset:
        raise ValueError(
            f"No runnable Tasks found in {cfg.source}. Define a `hud.Environment` with "
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
    return taskset


def _configure_logging(cfg: EvalConfig, *, single_run: bool) -> None:
    if cfg.very_verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(message)s")
        logging.getLogger("hud.agents").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    elif cfg.verbose or single_run:
        logging.getLogger("hud.agents").setLevel(logging.INFO)


def _truncate(text: str | list[Any] | None, max_len: int) -> str:
    if not text:
        return "—"
    flat = str(text).replace("\n", " ").strip()
    return flat[: max_len - 2] + ".." if len(flat) > max_len else flat


def _display_job(job: Job, *, source: str, elapsed: float) -> None:
    errors = set(map(id, job.errors))
    hud_console.print(f"\n[bold]'{source}' Results[/bold]")
    hud_console.print(f"  [dim]Runs:[/dim] {len(job.runs)}")
    hud_console.print(f"  [dim]Time:[/dim] {elapsed:.1f}s")
    hud_console.print(f"  [dim]Mean reward:[/dim] [green]{job.reward:.3f}[/green]")
    if errors:
        hud_console.print(f"  [dim]Errors:[/dim] [red]{len(errors)}[/red]")

    if len(job.runs) <= 50:
        table = Table(title="Details", show_header=True, header_style="bold")
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Prompt", style="dim", max_width=35)
        table.add_column("Answer", style="dim", max_width=35)
        table.add_column("Reward", justify="right", style="green", width=8)
        for index, run in enumerate(job.runs):
            table.add_row(
                str(index),
                _truncate(run.prompt, 35),
                _truncate(run.trace.content, 35),
                "[red]error[/red]" if id(run) in errors else f"{run.reward:.3f}",
            )
        hud_console.print(table)
    hud_console.print()


def _pick_tasks_file() -> str:
    """The tasks JSON/JSONL in the working directory, prompting when there are several."""
    names = sorted(
        path.name
        for path in Path.cwd().iterdir()
        if path.suffix in (".json", ".jsonl") and not path.name.startswith(".")
    )
    if not names:
        raise FileNotFoundError("No task JSON or JSONL files found in current directory")
    if len(names) == 1:
        return names[0]
    return hud_console.select("Select a tasks file", choices=names)


def eval_command(
    source: str | None = typer.Argument(None, help="Taskset slug or task JSON file"),
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
        help="Placement: local, hud (runtime tunnel), hosted (whole rollout on the platform), "
        "or a tcp:// url. Default: local for a tasks file; hosted for a platform taskset.",
    ),
    remote: bool = typer.Option(
        False,
        "--remote",
        help="Run the whole rollout on the HUD platform (same as --runtime hosted)",
    ),
) -> dict[str, Any]:
    """Run evaluation on datasets or individual tasks with agents.

    Examples:
        hud eval tasks.json claude-sonnet-4-6
        hud eval tasks.json claude
        hud eval "My Tasks" claude-sonnet-4-6 --full   # Platform taskset, run on the platform
        hud eval tasks.json claude --config max_tokens=32768
        hud eval tasks.json claude --gateway           # Route LLM calls through HUD Gateway
        hud eval tasks.json claude-sonnet-4-6 --runtime hud  # Use HUD runtime tunnel
        hud eval tasks.json claude-sonnet-4-6 --remote       # Execute rollout remotely
        hud eval tasks.json claude --yes --json
        hud eval tasks.json claude --dry-run --json
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
            "runtime": "hosted" if remote else runtime,
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
        overrides["agent_config"] = _agent_config_updates(
            config, overrides.get("agent_type", cfg.agent_type)
        )
    cfg = cfg.merge(overrides)

    if dry_run:
        agent_type = cfg.agent_type
        if cfg.source is None or agent_type is None:
            raise CliError(
                "usage",
                "Dry-run requires an explicit task source and agent (or configured defaults).",
            )
        cfg = cfg.with_placement()
        hud_console.info("--dry-run: no evaluation started")
        return {
            "dry_run": True,
            "action": "eval",
            "source": cfg.source,
            "agent": agent_type.value,
            "model": cfg.model,
            "runtime": cfg.runtime,
            "remote": cfg.runtime == "hosted",
            "all": cfg.all,
            "max_steps": cfg.max_steps,
            "max_concurrent": cfg.max_concurrent,
            "group_size": cfg.group_size,
            "task_ids": cfg.task_ids,
        }

    if cfg.source is None:
        cfg = cfg.merge({"source": _pick_tasks_file()})
        hud_console.success(f"Selected: {cfg.source}")
    if cfg.agent_type is None:
        preset = cast(
            "AgentPreset",
            hud_console.select(
                "Select an agent:",
                choices=[{"name": preset.name, "value": preset} for preset in _AGENT_PRESETS],
                default=0,
            ),
        )
        cfg = cfg.merge(preset.overrides())
    cfg = cfg.with_placement()
    cfg.require_credentials()
    cfg.display()
    CLI.confirm_or_abort("Proceed?", yes=yes, default=True)

    taskset = _load_taskset(cfg)
    single_run = len(taskset) == 1 and cfg.group_size == 1
    _configure_logging(cfg, single_run=single_run)
    if not single_run:
        hud_console.info(
            f"Running evaluation (max_concurrent: {cfg.max_concurrent}, "
            f"group_size: {cfg.group_size})"
        )
    started = time.monotonic()
    job = asyncio.run(
        taskset.run(
            _build_agent(cfg),
            runtime=_placement(cfg),
            group=cfg.group_size,
            max_concurrent=cfg.max_concurrent,
        )
    )
    elapsed = time.monotonic() - started
    if job.runs and settings.telemetry_enabled and settings.api_key:
        hud_console.info(f"{settings.hud_web_url}/jobs/{canonical_record_id(job.id)}")
    if job.runs:
        _display_job(job, source=cfg.source or "", elapsed=elapsed)
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
