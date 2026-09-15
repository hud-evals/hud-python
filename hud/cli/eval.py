"""HUD evaluation command for running tasks and datasets.

Config Override Order: CLI arguments > .hud_eval.toml > defaults
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from string import Template
from typing import TYPE_CHECKING, Any, ClassVar, cast

import typer
from pydantic import BaseModel, Field, field_validator
from rich import box
from rich.table import Table

from hud.cli.config import parse_key_value
from hud.cli.io import CliError, confirm_or_abort, json_option, read_text_arg, report
from hud.cli.source import environment_file
from hud.settings import settings
from hud.types import AgentType
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient, canonical_record_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.eval.run import Run

_BEDROCK_ARN_PATTERN = re.compile(r"^arn:aws:bedrock:[a-z0-9-]+:\d+:inference-profile/.+$")
_SUCCESS_THRESHOLD = 0.7


def _is_bedrock_arn(model: str | None) -> bool:
    """Check if a model string is a Bedrock inference profile ARN."""
    return model is not None and bool(_BEDROCK_ARN_PATTERN.match(model))


logger = logging.getLogger(__name__)
hud_console = HUDConsole()

_CONFIG_PATH = ".hud_eval.toml"
_PLACEMENT_CONFLICT_ERROR = "--runtime and --remote are mutually exclusive placement options"


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ``${VAR_NAME}`` placeholders in config values.

    Sources values from ``os.environ`` and ``hud.settings`` (uppercase aliases
    included, so both ``${api_key}`` and ``${API_KEY}`` work). Missing
    variables resolve to empty strings.
    """
    mapping: dict[str, Any] = dict(os.environ)
    settings_dict = settings.model_dump()
    mapping.update(settings_dict)
    mapping.update({key.upper(): val for key, val in settings_dict.items()})
    if settings.api_key:
        mapping["HUD_API_KEY"] = settings.api_key

    safe_mapping: defaultdict[str, Any] = defaultdict(str, mapping)

    def substitute(value: Any) -> Any:
        if isinstance(value, str):
            return Template(value).substitute(safe_mapping)
        if isinstance(value, dict):
            return {k: substitute(v) for k, v in value.items()}
        if isinstance(value, list):
            return [substitute(item) for item in value]
        return value

    return substitute(obj)


def _require_bedrock_credentials() -> None:
    missing_aws = (
        not settings.aws_access_key_id
        or not settings.aws_secret_access_key
        or not settings.aws_region
    )
    if missing_aws:
        raise HudAuthenticationError(
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are required for AWS Bedrock"
        )


@dataclass(frozen=True)
class AgentPreset:
    """A preset agent configuration combining agent type, model, and optional config."""

    name: str
    agent_type: AgentType
    model: str | None = None
    agent_config: dict[str, Any] | None = None


_AGENT_PRESETS: list[AgentPreset] = [
    AgentPreset("Claude Sonnet 4.6", AgentType.CLAUDE, "claude-sonnet-4-6"),
    AgentPreset("Claude Opus 4.8", AgentType.CLAUDE, "claude-opus-4-8"),
    AgentPreset("GPT-5.6", AgentType.OPENAI, "gpt-5.6"),
    AgentPreset("GPT-5.5", AgentType.OPENAI, "gpt-5.5"),
    AgentPreset("Gemini 3.1 Pro (Preview)", AgentType.GEMINI, "gemini-3.1-pro-preview"),
    AgentPreset(
        "Grok 4-1 Fast (xAI)",
        AgentType.OPENAI_COMPATIBLE,
        "grok-4-1-fast",
        {
            "openai_compatible": {
                "base_url": settings.hud_gateway_url,
                "model_name": "Grok 4-1 Fast",
            }
        },
    ),
    AgentPreset(
        "GLM 5.2 (Z.ai)",
        AgentType.OPENAI_COMPATIBLE,
        "z-ai/glm-5.2",
        {"openai_compatible": {"base_url": settings.hud_gateway_url, "model_name": "GLM 5.2"}},
    ),
    AgentPreset(
        "Kimi K2.6 (Moonshot)",
        AgentType.OPENAI_COMPATIBLE,
        "moonshotai/kimi-k2.6",
        {"openai_compatible": {"base_url": settings.hud_gateway_url, "model_name": "Kimi K2.6"}},
    ),
    AgentPreset(
        "MiniMax M3",
        AgentType.OPENAI_COMPATIBLE,
        "MiniMax-M3",
        {"openai_compatible": {"base_url": settings.hud_gateway_url, "model_name": "MiniMax M3"}},
    ),
]


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


def _merge_agent_config(
    current: dict[str, Any],
    *,
    selected_agent: AgentType | str | None,
    updates: list[str] | None,
) -> dict[str, Any] | None:
    if not updates:
        return None
    if isinstance(selected_agent, str):
        try:
            selected_agent = AgentType(selected_agent)
        except ValueError:
            selected_agent = None

    merged = dict(current)
    for item in updates:
        parsed = parse_key_value(item)
        if parsed is None:
            continue
        key, value = parsed
        parsed_value = _parse_config_value(value)

        if "." in key:
            agent_name, param = key.split(".", 1)
        elif selected_agent is not None:
            agent_name, param = selected_agent.value, key
        else:
            continue

        existing = merged.get(agent_name, {})
        agent_config = dict(existing) if isinstance(existing, dict) else {}
        agent_config[param] = parsed_value
        merged[agent_name] = agent_config
    return merged


class EvalConfig(BaseModel):
    """Configuration for hud eval command."""

    _EVAL_FIELDS: ClassVar[set[str]] = {
        "source",
        "agent_type",
        "task_ids",
        "all",
        "max_concurrent",
        "max_steps",
        "verbose",
        "very_verbose",
        "group_size",
        "auto_respond",
        "gateway",
        "runtime",
        "remote",
    }
    source: str | None = None
    agent_type: AgentType | None = None
    model: str | None = None
    task_ids: list[str] | None = None
    all: bool = False
    max_concurrent: int = 30
    max_steps: int = 10
    verbose: bool = False
    very_verbose: bool = False
    auto_respond: bool | None = None
    group_size: int = 1
    gateway: bool = False
    #: Placement: "local" (spawn each row's env from the source), "hud"
    #: (HUD runtime tunnel), or a tcp:// url of an already-served env.
    #: ``None`` means "infer from the source": a local file runs locally, a
    #: platform taskset (slug/id, no env source on disk) runs remotely.
    runtime: str | None = None
    #: Run the whole rollout remotely on the HUD platform.
    remote: bool = False

    agent_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_type", mode="before")
    @classmethod
    def _parse_agent_type(cls, v: Any) -> AgentType | None:
        if v is None:
            return None
        if isinstance(v, AgentType):
            return v
        if isinstance(v, str):
            try:
                return AgentType(v)
            except ValueError:
                valid = [e.value for e in AgentType]
                raise ValueError(
                    f"Invalid agent: {v}. Must be one of: {', '.join(valid)}"
                ) from None
        return v

    def source_is_local_file(self) -> bool:
        """Whether ``source`` points at an on-disk taskset (vs. a platform slug/id)."""
        return self.source is not None and Path(self.source).exists()

    def resolve_runtime(self) -> EvalConfig:
        """Pin the effective placement from the source type.

        A local file/dir has its env source on disk, so it defaults to spawning
        envs locally; a platform taskset (slug or id) has no env source on disk,
        so it defaults to whole-rollout remote execution. An explicit
        ``--runtime`` is always honored, except ``local`` against a platform
        taskset, which has no env to spawn.
        """
        if self.runtime is None:
            if self.source_is_local_file():
                return self.model_copy(update={"runtime": "local"})
            return self.model_copy(update={"remote": True})
        if self.runtime == "local" and not self.source_is_local_file():
            raise ValueError(
                f"--runtime local needs a local env source, but {self.source!r} is a "
                "platform taskset with no env source on disk. Run it on the platform "
                "by omitting --runtime or passing --remote, export it first "
                "(hud sync tasks <name> --export tasks.json) and run that file, "
                "or attach to a served env with --runtime tcp://host:port."
            )
        return self

    def validate_api_keys(self) -> None:
        if self.remote or self.runtime == "hud" or self.gateway:
            PlatformClient.from_settings()
        if self.agent_type == AgentType.OPENAI_COMPATIBLE:
            config_model = self.agent_config.get("openai_compatible", {}).get("model")
            if not self.model and not config_model:
                raise ValueError("Model name is required for OpenAI compatible agent; use --model.")

    def get_agent_kwargs(self) -> dict[str, Any]:
        """Build agent kwargs from config.

        Model precedence:
        1. CLI --model (highest priority)
        2. [agent_type].model in TOML (per-agent config)
        """
        if self.agent_type is None:
            raise ValueError("agent_type must be set before calling get_agent_kwargs()")

        kwargs: dict[str, Any] = {}

        agent_key = self.agent_type.value
        if agent_key in self.agent_config:
            agent_cfg = dict(self.agent_config[agent_key])
            kwargs.update(agent_cfg)

        if self.model:
            kwargs["model"] = self.model

        if isinstance(kwargs.get("model"), str):
            from hud.utils.gateway import normalize_gateway_model_id

            kwargs["model"] = normalize_gateway_model_id(kwargs["model"])

        if self.agent_type == AgentType.OPENAI_COMPATIBLE and "api_key" not in kwargs:
            base_url = kwargs.get("base_url", "")
            if settings.hud_gateway_url in base_url and settings.api_key:
                kwargs["api_key"] = settings.api_key

        kwargs["verbose"] = self.verbose or self.very_verbose
        kwargs["max_steps"] = self.max_steps

        return kwargs

    @classmethod
    def load(cls, path: str = _CONFIG_PATH) -> EvalConfig:
        """Load config from TOML file."""
        p = Path(path)
        if not p.exists():
            return cls()

        with p.open("rb") as f:
            toml_data = tomllib.load(f)

        toml_data = _resolve_env_vars(toml_data)

        eval_section = toml_data.get("eval", {})
        data: dict[str, Any] = {}

        if "agent" in eval_section:
            data["agent_type"] = eval_section["agent"]
        for key in cls._EVAL_FIELDS:
            if key in eval_section:
                data[key] = eval_section[key]

        agent_config: dict[str, Any] = {}
        for agent_type in AgentType:
            if agent_type.value in toml_data:
                agent_config[agent_type.value] = toml_data[agent_type.value]
        data["agent_config"] = agent_config

        return cls.model_validate(data)

    def merge_cli(
        self,
        *,
        source: str | None = None,
        agent: str | None = None,
        model: str | None = None,
        all: bool = False,
        full: bool = False,
        max_concurrent: int | None = None,
        max_steps: int | None = None,
        verbose: bool = False,
        very_verbose: bool = False,
        auto_respond: bool = False,
        group_size: int | None = None,
        gateway: bool = False,
        config: list[str] | None = None,
        task_ids: str | None = None,
        runtime: str | None = None,
        remote: bool = False,
    ) -> EvalConfig:
        """Merge CLI args (non-None values override config)."""
        if runtime is not None and remote:
            raise ValueError(_PLACEMENT_CONFLICT_ERROR)

        overrides: dict[str, Any] = {
            key: value
            for key, value in {
                "source": source,
                "model": model,
                "max_concurrent": max_concurrent,
                "max_steps": max_steps,
                "group_size": group_size,
                "runtime": runtime,
            }.items()
            if value is not None
        }
        if agent is not None:
            try:
                AgentType(agent)
                overrides["agent_type"] = agent
            except ValueError:
                from hud.agents import resolve_agent_model

                agent_type, model_name = resolve_agent_model(agent)
                overrides["agent_type"] = agent_type.value
                overrides.setdefault("model", model_name)

        if task_ids is not None:
            overrides["task_ids"] = [t.strip() for t in task_ids.split(",") if t.strip()]

        if runtime is not None:
            overrides["remote"] = False

        for key, value in {
            "all": all,
            "verbose": verbose,
            "very_verbose": very_verbose,
            "auto_respond": auto_respond,
            "gateway": gateway,
            "remote": remote,
        }.items():
            if value:
                overrides[key] = True

        if full:
            overrides["all"] = True
            if "auto_respond" not in overrides:
                overrides["auto_respond"] = True
            if "max_steps" not in overrides:
                overrides["max_steps"] = 100

        merged_agent_config = _merge_agent_config(
            self.agent_config,
            selected_agent=overrides.get("agent_type") or self.agent_type,
            updates=config,
        )
        if merged_agent_config is not None:
            overrides["agent_config"] = merged_agent_config

        return self.model_validate({**self.model_dump(), **overrides})

    def resolve_agent_interactive(self) -> EvalConfig:
        """Prompt user to select an agent preset if not set. Returns updated config."""
        if self.agent_type is not None:
            return self

        choices: list[str | dict[str, Any]] = [
            {"name": preset.name, "value": preset} for preset in _AGENT_PRESETS
        ]

        selected = cast(
            "AgentPreset",
            hud_console.select("Select an agent:", choices=choices, default=0),
        )

        updates: dict[str, Any] = {"agent_type": selected.agent_type}
        if selected.model:
            updates["model"] = selected.model
        if selected.agent_config:
            merged = dict(self.agent_config)
            for key, value in selected.agent_config.items():
                if key in merged:
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
            updates["agent_config"] = merged

        return self.model_validate({**self.model_dump(), **updates})

    def display(self) -> None:
        """Display settings in a table."""
        table = Table(title="Evaluation Settings", title_style="bold cyan", box=box.ROUNDED)
        table.add_column("Setting", style="yellow")
        table.add_column("Value", style="green")

        table.add_row("source", str(self.source or "-"))
        table.add_row("runtime", str(self.runtime or "-"))
        table.add_row("agent", self.agent_type.value if self.agent_type else "-")
        if self.task_ids:
            table.add_row(
                "task_ids", ", ".join(self.task_ids[:5]) + ("..." if len(self.task_ids) > 5 else "")
            )
        table.add_row("all", str(self.all))
        table.add_row("max_steps", str(self.max_steps))
        table.add_row("max_concurrent", str(self.max_concurrent))
        if self.group_size > 1:
            table.add_row("group_size", str(self.group_size))
        if self.auto_respond:
            table.add_row("auto_respond", "[bold green]True[/bold green]")
        if self.very_verbose:
            table.add_row("very_verbose", "[bold green]True[/bold green]")
        elif self.verbose:
            table.add_row("verbose", "[bold green]True[/bold green]")
        if self.gateway:
            table.add_row("gateway", "[bold green]True[/bold green] (routing via HUD Gateway)")
        if self.remote:
            table.add_row("remote", "[bold green]True[/bold green]")

        if self.agent_type:
            table.add_row("", "")
            table.add_row(f"[dim]{self.agent_type.value} config[/dim]", "")

            config_cls = self.agent_type.config_cls
            defaults = config_cls()
            overrides = self.agent_config.get(self.agent_type.value, {})
            skip = {
                "model_client",
                "model_name",
                "model_config",
                "system_prompt",
            }

            sensitive_fields = {"api_key", "api_secret", "token", "password", "secret"}

            for name in config_cls.model_fields:
                if name in skip:
                    continue
                if name == "model":
                    if self.model:
                        value = self.model
                    elif overrides.get("model"):
                        value = overrides["model"]
                    else:
                        value = getattr(defaults, "model", None)
                    table.add_row("  model", str(value) if value else "-")
                elif name in overrides:
                    value = overrides[name]
                    if name in sensitive_fields and value:
                        display_value = f"{str(value)[:4]}****" if len(str(value)) > 4 else "****"
                    else:
                        display_value = str(value)
                    table.add_row(f"  {name}", display_value)

        hud_console.console.print(table)


def _build_agent(cfg: EvalConfig) -> Any:
    """Construct a new-flow agent (``agent(run)``) from the eval config."""
    if cfg.agent_type is None:
        raise ValueError("agent_type must be set")
    agent_kwargs = cfg.get_agent_kwargs()
    if cfg.auto_respond:
        agent_kwargs["auto_respond"] = True

    from hud.agents.types import OpenAIChatConfig
    from hud.utils.gateway import build_gateway_client, build_model_client

    config = cfg.agent_type.config_cls(**agent_kwargs)
    custom_endpoint = isinstance(config, OpenAIChatConfig) and (
        config.api_key is not None or config.base_url is not None
    )
    if not cfg.remote and config.model_client is None and not custom_endpoint:
        if cfg.agent_type == AgentType.CLAUDE and _is_bedrock_arn(config.model):
            _require_bedrock_credentials()

            from anthropic import AsyncAnthropicBedrock

            config.model_client = AsyncAnthropicBedrock(
                aws_access_key=settings.aws_access_key_id,
                aws_secret_key=settings.aws_secret_access_key,
                aws_region=settings.aws_region,
            )
        elif cfg.gateway:
            config.model_client = build_gateway_client(cfg.agent_type.gateway_provider)
        else:
            config.model_client = build_model_client(
                cfg.agent_type.gateway_provider, prefer_provider=True
            )

    # cls/config_cls are matched unions; the pairing is correct by construction.
    return cast("Any", cfg.agent_type.cls)(config=config)


def _local_subprocess(task: Any) -> Any:
    """Serve the Environment bound on ``task`` (``SubprocessRuntime``)."""
    from hud.eval import SubprocessRuntime

    env = task._env
    if env is None:
        raise ValueError(
            "no placement: these rows have no bound Environment. "
            "Pass a Python tasks/env module, or use --remote / --runtime hud / a tcp:// url."
        )
    return SubprocessRuntime(environment_file(env))(task)


def _resolve_placement(cfg: EvalConfig, source_path: Path | None, taskset: Any) -> Any:
    """Map the config's ``runtime`` onto a placement for ``Taskset.run``.

    "local" uses the Environment already bound on each row (the same
    inference ``Taskset.run()`` does): container rows get ``DockerRuntime``,
    otherwise that env's defining file is served in a subprocess;
    "hud" opens the HUD runtime tunnel while keeping the agent loop local;
    ``--remote`` submits every rollout for platform-hosted execution; a
    ``tcp://`` url attaches to an env served elsewhere.
    """
    from hud.eval import DockerRuntime, HostedRuntime, HUDRuntime, Runtime

    if cfg.remote:
        return HostedRuntime()
    if cfg.runtime == "local":
        if source_path is None:
            raise ValueError("local placement requires a local source path")
        docker = DockerRuntime()

        def local(task: Any) -> Any:
            config = task.runtime_config
            return (
                docker(task)
                if config is not None and (config.image is not None or config.compose is not None)
                else _local_subprocess(task)
            )

        return local
    if cfg.runtime == "hud":
        return HUDRuntime()
    if cfg.runtime is not None and cfg.runtime.startswith("tcp://"):
        return Runtime(cfg.runtime)
    raise ValueError(
        f"Unknown runtime {cfg.runtime!r}. Use 'local', 'hud', a tcp:// url, or --remote."
    )


async def _run_evaluation(cfg: EvalConfig) -> Any:
    """Run evaluation on the Env/Task/Taskset/Run flow.

    Loads a ``Taskset`` from a Python source or JSON/JSONL taskset and runs it
    on the configured placement (default: spawned local substrates — each
    rollout serves its own row's env, so mixed-env tasksets are one job).
    Returns the ``Job`` receipt containing the live execution ``Run`` results.
    """
    if cfg.source is None or cfg.agent_type is None:
        raise ValueError("source and agent_type must be set")

    from hud.eval import Taskset

    source_path = Path(cfg.source)
    is_local = await asyncio.to_thread(source_path.exists)
    if is_local:
        hud_console.info(f"Loading tasks from: {cfg.source}")
        taskset = Taskset.from_file(source_path)
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

    if len(taskset) == 1 and cfg.group_size == 1:
        logging.getLogger("hud.agents").setLevel(logging.INFO)
    else:
        hud_console.info(
            f"Running evaluation (max_concurrent: {cfg.max_concurrent}, "
            f"group_size: {cfg.group_size})"
        )

    agent = _build_agent(cfg)
    placement = _resolve_placement(cfg, source_path if is_local else None, taskset)

    job = await taskset.run(
        agent,
        runtime=placement,
        group=cfg.group_size,
        max_concurrent=cfg.max_concurrent,
    )
    if job.runs and settings.telemetry_enabled and settings.api_key:
        hud_console.info(f"{settings.hud_web_url}/jobs/{canonical_record_id(job.id)}")

    return job


def _truncate(text: str | list[Any] | None, max_len: int) -> str:
    if not text:
        return "—"
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\n", " ").strip()
    return text[: max_len - 2] + ".." if len(text) > max_len else text


def display_runs(
    runs: Sequence[Run],
    *,
    name: str = "",
    elapsed: float | None = None,
    show_details: bool = True,
) -> None:
    """Print a summary (+ per-run details table) for a batch of runs."""
    if not runs:
        print("No results to display")  # noqa: T201
        return

    rewards = [r.reward for r in runs]
    errors = [r for r in runs if r.trace.is_error]
    mean_reward = mean(rewards)
    std_reward = pstdev(rewards) if len(rewards) > 1 else 0.0
    success_rate = sum(1 for r in rewards if r > _SUCCESS_THRESHOLD) / len(runs)

    console = hud_console.console
    title = f"'{name}' Results" if name else "Evaluation Complete"
    console.print(f"\n[bold]{title}[/bold]")
    console.print(f"  [dim]Runs:[/dim] {len(runs)}")
    if elapsed:
        rate = len(runs) / elapsed if elapsed > 0 else 0
        console.print(f"  [dim]Time:[/dim] {elapsed:.1f}s ({rate:.1f}/s)")
    console.print(
        f"  [dim]Mean reward:[/dim] [green]{mean_reward:.3f}[/green] +/- {std_reward:.3f}"
    )
    console.print(f"  [dim]Success rate:[/dim] [yellow]{success_rate * 100:.1f}%[/yellow]")
    if errors:
        console.print(f"  [dim]Errors:[/dim] [red]{len(errors)}[/red]")

    if show_details and len(runs) <= 50:
        table = Table(title="Details", show_header=True, header_style="bold")
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Prompt", style="dim", max_width=35)
        table.add_column("Answer", style="dim", max_width=35)
        table.add_column("Reward", justify="right", style="green", width=8)
        table.add_column("", justify="center", width=3)
        for i, run in enumerate(runs):
            if run.trace.is_error:
                status = "[red]✗[/red]"
            elif run.reward > _SUCCESS_THRESHOLD:
                status = "[green]✓[/green]"
            else:
                status = "[yellow]○[/yellow]"
            row: list[Any] = [
                str(i),
                _truncate(run.prompt, 35),
                _truncate(run.trace.content, 35),
                f"{run.reward:.3f}",
                status,
            ]
            table.add_row(*row)
        console.print(table)

    if std_reward > 0.3:
        console.print(f"\n[yellow]High variance (std={std_reward:.3f})[/yellow]")
    console.print()


def find_tasks_file(tasks_file: str | None, msg: str = "Select a tasks file") -> str:
    """Return a local tasks file, prompting when more than one exists."""
    if tasks_file:
        return tasks_file

    cwd = Path.cwd()
    names = [
        path.name
        for path in (*cwd.glob("*.json"), *cwd.glob("*.jsonl"))
        if not path.name.startswith(".")
    ]
    if not names:
        raise FileNotFoundError("No task JSON or JSONL files found in current directory")
    if len(names) == 1:
        return names[0]
    return hud_console.select(msg, choices=names)


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
        help="Run the entire dataset. Shortcut for --all --auto-respond  --max-steps 100",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name"),
    config: list[str] | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Agent config: key=value"
    ),
    from_json: Path | None = typer.Option(  # noqa: B008
        None,
        "--from-json",
        help="Load full eval configuration from a JSON file (e.g. exported from a HUD job).",
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
    json_output: bool = json_option(),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
    gateway: bool = typer.Option(
        False, "--gateway", "-g", help="Route LLM API calls through HUD Gateway"
    ),
    runtime: str | None = typer.Option(
        None,
        "--runtime",
        help="Placement: local, hud (runtime tunnel), or a tcp:// url. "
        "Default: local for a tasks file; remote for a platform taskset.",
    ),
    remote: bool = typer.Option(
        False,
        "--remote",
        help="Run the whole rollout remotely on the HUD platform",
    ),
) -> None:
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

    if from_json is not None:
        cfg = EvalConfig.model_validate_json(read_text_arg(str(from_json)))
    else:
        cfg = EvalConfig.load()

    cfg = cfg.merge_cli(
        source=source,
        agent=agent,
        model=model,
        all=all,
        full=full,
        max_concurrent=max_concurrent,
        max_steps=max_steps,
        task_ids=task_ids,
        verbose=verbose,
        very_verbose=very_verbose,
        auto_respond=auto_respond,
        group_size=group_size,
        config=config,
        gateway=gateway,
        runtime=runtime,
        remote=remote,
    )

    if dry_run and (cfg.source is None or cfg.agent_type is None):
        raise CliError(
            "usage", "Dry-run requires an explicit task source and agent (or configured defaults)."
        )
    if dry_run:
        cfg = cfg.resolve_runtime()
        plan = {
            "dry_run": True,
            "action": "eval",
            "source": cfg.source,
            "agent": cfg.agent_type.value if cfg.agent_type else None,
            "model": cfg.model,
            "runtime": cfg.runtime,
            "remote": cfg.remote,
            "all": cfg.all,
            "max_steps": cfg.max_steps,
            "max_concurrent": cfg.max_concurrent,
            "group_size": cfg.group_size,
            "task_ids": cfg.task_ids,
        }
        report(
            plan,
            json_output=json_output,
            render=lambda _saved: hud_console.info("--dry-run: no evaluation started"),
        )
        return

    if cfg.source is None:
        cfg = cfg.model_copy(update={"source": find_tasks_file(None, msg="Select a tasks file")})
        hud_console.success(f"Selected: {cfg.source}")

    cfg = cfg.resolve_agent_interactive()
    cfg = cfg.resolve_runtime()

    if cfg.very_verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(message)s")
        logging.getLogger("hud.agents").setLevel(logging.DEBUG)
        # Suppress noisy HTTP client logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    elif cfg.verbose:
        logging.getLogger("hud.agents").setLevel(logging.INFO)

    cfg.validate_api_keys()

    cfg.display()

    confirm_or_abort("Proceed?", yes=yes, default=True)

    start_time = time.time()
    job = asyncio.run(_run_evaluation(cfg))
    elapsed = time.time() - start_time

    runs = job.runs
    saved = {
        "job_id": job.id,
        "source": cfg.source,
        "run_count": len(runs),
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
            for run in runs
        ],
    }

    def _render(row: dict[str, Any]) -> None:
        if row["runs"]:
            display_runs(runs, name=str(row["source"] or ""), elapsed=float(row["elapsed_seconds"]))

    report(saved, json_output=json_output, render=_render)
