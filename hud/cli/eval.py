"""HUD evaluation command for running tasks and datasets.

Config Override Order: CLI arguments > .hud_eval.toml > defaults
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, ClassVar, cast

import typer
from pydantic import BaseModel, Field, field_validator
from rich import box
from rich.table import Table

from hud.cli.utils.api import require_api_key
from hud.cli.utils.config import parse_key_value
from hud.settings import settings
from hud.types import AgentType
from hud.utils.hud_console import HUDConsole

_BEDROCK_ARN_PATTERN = re.compile(r"^arn:aws:bedrock:[a-z0-9-]+:\d+:inference-profile/.+$")


def _is_bedrock_arn(model: str | None) -> bool:
    """Check if a model string is a Bedrock inference profile ARN."""
    return model is not None and bool(_BEDROCK_ARN_PATTERN.match(model))


def _resolve_model_from_catalog(model_id: str) -> tuple[AgentType, str] | None:
    """Look up a model in the gateway catalog and return (agent_type, model_name).

    Returns None if the model isn't found or the catalog is unreachable.
    """
    try:
        from hud.utils.gateway import list_gateway_models, normalize_gateway_model_id

        model_id = normalize_gateway_model_id(model_id)
        models = list_gateway_models()
    except Exception:
        return None
    for m in models:
        if (m.model_name == model_id or m.id == model_id) and m.sdk_agent_type:
            try:
                return AgentType(m.sdk_agent_type), m.model_name or model_id
            except ValueError:
                pass
    return None


logger = logging.getLogger(__name__)
hud_console = HUDConsole()

_CONFIG_PATH = ".hud_eval.toml"
_PLACEMENT_CONFLICT_ERROR = "--runtime and --remote are mutually exclusive placement options"
_SOURCE_FORMATS = ("hud", "harbor")


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
        hud_console.error(
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are required for AWS Bedrock"
        )
        raise typer.Exit(1)


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

_DEFAULT_CONFIG_TEMPLATE = """# HUD Eval Configuration
# Command-line arguments override these settings

[eval]
# source = "hud-evals/SheetBench-50"
# agent = "claude"
# all = false  # Run all problems instead of just 1
# max_concurrent = 30
# max_steps = 10
# group_size = 1
# task_ids = ["checkout-smoke", "0"]  # slugs or 0-based indices
# verbose = true
# very_verbose = true
# auto_respond = true
# gateway = false  # Route LLM API calls through HUD Gateway
# format = "hud"  # hud or harbor
# runtime = "local"  # local, hud, or tcp://host:port
# remote = false  # Run the whole rollout remotely on HUD

[claude]
# model = "claude-sonnet-4-6"
# max_tokens = 16384
# use_computer_beta = true

[openai]
# model = "gpt-5.5"
# temperature = 0.7
# max_output_tokens = 4096

[gemini]
# model = "gemini-2.5-pro"
# temperature = 1.0
# top_p = 0.95

[openai_compatible]
# base_url = "http://localhost:8000/v1"
# model = "my-model"
"""

# Agent type -> (settings attr, env var name)
_API_KEY_REQUIREMENTS: dict[AgentType, tuple[str, str]] = {
    AgentType.CLAUDE: ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    AgentType.GEMINI: ("gemini_api_key", "GEMINI_API_KEY"),
    AgentType.OPENAI: ("openai_api_key", "OPENAI_API_KEY"),
}


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
        "format",
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
    #: Source format. ``None``/``hud`` means normal HUD task source loading;
    #: ``harbor`` opts into the Harbor integration loader/runtime.
    format: str | None = None
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

    @field_validator("format", mode="before")
    @classmethod
    def _parse_format(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            return v
        normalized = v.strip().lower()
        if normalized in ("", "hud"):
            return None
        if normalized in _SOURCE_FORMATS:
            return normalized
        raise ValueError(f"Invalid format: {v}. Must be one of: {', '.join(_SOURCE_FORMATS)}")

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
        if self.format == "harbor":
            if not self.source_is_local_file():
                hud_console.error("--format harbor requires a local Harbor task directory")
                raise typer.Exit(1)
            if self.remote or (self.runtime is not None and self.runtime != "local"):
                hud_console.error("--format harbor currently supports only local runtime placement")
                raise typer.Exit(1)
        if self.runtime is None:
            if self.source_is_local_file():
                return self.model_copy(update={"runtime": "local"})
            return self.model_copy(update={"remote": True})
        if self.runtime == "local" and not self.source_is_local_file():
            hud_console.error(
                f"--runtime local needs a local env source, but {self.source!r} is a "
                "platform taskset with no env source on disk. Run it on the platform "
                "by omitting --runtime or passing --remote, export it first "
                "(hud sync tasks <name> --export tasks.json) and run that file, "
                "or attach to a served env with --runtime tcp://host:port."
            )
            raise typer.Exit(1)
        return self

    def validate_api_keys(self) -> None:
        if self.agent_type is None:
            return

        # Hosted placement runs the agent on the platform, where LLM calls
        # always route through the HUD gateway — no local provider key is
        # involved, and a local gateway model_client could not travel with
        # the submission anyway. Only HUD_API_KEY matters.
        if self.remote:
            require_api_key("run remote hosted evals")
            if self.gateway:
                self.gateway = False
                hud_console.info(
                    "--gateway is implied by --remote (the hosted runner always "
                    "routes through the HUD gateway); ignoring the flag locally."
                )
            return

        if self.runtime == "hud":
            require_api_key("run HUD runtime tunnel evals")

        # Gateway by default: when the provider key is missing but HUD_API_KEY is
        # set, route via the HUD gateway instead of erroring — the out-of-the-box
        # path needs only one key.
        if (
            not self.gateway
            and self.agent_type in _API_KEY_REQUIREMENTS
            and not _is_bedrock_arn(self.model)
            and settings.api_key
        ):
            attr, env_var = _API_KEY_REQUIREMENTS[self.agent_type]
            if not getattr(settings, attr, None):
                self.gateway = True
                hud_console.info(
                    f"No {env_var} set — routing via the HUD Gateway with your HUD_API_KEY. "
                    f"Set {env_var} to call the provider directly."
                )

        if self.gateway:
            require_api_key("use gateway mode")
            return

        if self.agent_type == AgentType.OPENAI_COMPATIBLE:
            config_model = self.agent_config.get("openai_compatible", {}).get("model")
            if not self.model and not config_model:
                hud_console.error(
                    "Model name is required for OpenAI compatible agent. "
                    "Use --model or set model in [openai_compatible] section of .hud_eval.toml"
                )
                raise typer.Exit(1)
        elif self.agent_type == AgentType.CLAUDE and _is_bedrock_arn(self.model):
            _require_bedrock_credentials()
        elif self.agent_type in _API_KEY_REQUIREMENTS:
            attr, env_var = _API_KEY_REQUIREMENTS[self.agent_type]
            if not getattr(settings, attr, None):
                hud_console.error(f"{env_var} is required for {self.agent_type.value} agent")
                hud_console.info(f"Set it: hud set {env_var}=your-key-here")
                raise typer.Exit(1)

        if not settings.api_key:
            hud_console.warning("HUD_API_KEY not set. Some features may be limited.")

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

        bedrock_arn_detected = _is_bedrock_arn(kwargs.get("model")) or _is_bedrock_arn(
            kwargs.get("checkpoint_name")
        )
        if self.agent_type == AgentType.CLAUDE and bedrock_arn_detected:
            _require_bedrock_credentials()

            from anthropic import AsyncAnthropicBedrock

            kwargs["model_client"] = AsyncAnthropicBedrock(
                aws_access_key=settings.aws_access_key_id,
                aws_secret_key=settings.aws_secret_access_key,
                aws_region=settings.aws_region or "us-east-1",
            )
            hud_console.info("Using AWS Bedrock (detected ARN in model)")

        kwargs["verbose"] = self.verbose or self.very_verbose
        kwargs["max_steps"] = self.max_steps

        return kwargs

    @classmethod
    def load(cls, path: str = _CONFIG_PATH) -> EvalConfig:
        """Load config from TOML file."""
        p = Path(path)
        if not p.exists():
            p.write_text(_DEFAULT_CONFIG_TEMPLATE)
            hud_console.info(f"Generated {_CONFIG_PATH}")
            return cls()

        try:
            with open(p, "rb") as f:
                toml_data = tomllib.load(f)
        except Exception as e:
            hud_console.warning(f"Failed to parse {path}: {e}")
            return cls()

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

        try:
            return cls.model_validate(data)
        except Exception as e:
            hud_console.warning(f"Invalid config: {e}")
            return cls()

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
        format: str | None = None,
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
                "format": format,
                "runtime": runtime,
            }.items()
            if value is not None
        }
        if agent is not None:
            try:
                AgentType(agent)
                overrides["agent_type"] = agent
            except ValueError:
                resolved = _resolve_model_from_catalog(agent)
                if resolved is not None:
                    agent_type, model_name = resolved
                    overrides["agent_type"] = agent_type.value
                    if "model" not in overrides:
                        overrides["model"] = model_name
                else:
                    overrides["agent_type"] = agent  # let validator surface the error

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
        if self.format:
            table.add_row("format", self.format)
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

    if cfg.gateway:
        from hud.utils.gateway import build_gateway_client

        agent_kwargs.setdefault(
            "model_client", build_gateway_client(cfg.agent_type.gateway_provider)
        )
        hud_console.info(f"Using HUD Gateway for {cfg.agent_type.gateway_provider} API")

    config = cfg.agent_type.config_cls(**agent_kwargs)
    # cls/config_cls are matched unions; the pairing is correct by construction.
    return cast("Any", cfg.agent_type.cls)(config=config)


def _python_defines_environment(path: Path) -> bool:
    """Return True when ``path`` constructs a v6 :class:`~hud.environment.Environment`."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
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
        if callee_name == "Environment":
            return True
    return False


def _spawn_target(source: Path) -> Path:
    """The path the ``LocalRuntime`` provider serves.

    Directories and env-defining ``.py`` files are served as-is. Task-only
    sources (``tasks.py`` importing from ``env.py``) resolve to a sibling
    ``env.py`` or the containing directory. JSON/JSONL data files use the
    surrounding directory (the env source lives next to the tasks file).
    """
    resolved = source.resolve()
    if resolved.is_dir():
        return resolved
    if resolved.suffix != ".py":
        return resolved.parent
    if _python_defines_environment(resolved):
        return resolved
    env_py = resolved.parent / "env.py"
    if env_py.is_file():
        return env_py
    return resolved.parent


def _load_local_taskset(source_path: Path, source_format: str | None) -> Any:
    from hud.eval import Taskset

    format_name = source_format or "hud"
    if format_name == "hud":
        taskset = Taskset.from_file(source_path)
        if len(taskset) == 0:
            from integrations.harbor import detect

            if detect(source_path):
                hud_console.hint(
                    f"{source_path} looks like a Harbor task directory; "
                    "rerun with --format harbor to load it."
                )
        return taskset
    if format_name == "harbor":
        from integrations.harbor import load

        return load(source_path)
    raise ValueError(f"unsupported task source format: {format_name}")


def _resolve_placement(cfg: EvalConfig, source_path: Path | None) -> Any:
    """Map the config's ``runtime`` onto a placement for ``Taskset.run``.

    "local" spawns each row's env from the source next to the tasks file;
    "hud" opens the HUD runtime tunnel while keeping the agent loop local;
    ``--remote`` submits every rollout for platform-hosted execution; a
    ``tcp://`` url attaches to an env served elsewhere.
    """
    from hud.eval import HostedRuntime, HUDRuntime, LocalRuntime, Runtime

    if cfg.remote:
        require_api_key("run remote hosted evals")
        return HostedRuntime()
    if cfg.runtime == "local":
        if source_path is None:
            raise ValueError("local placement requires a local source path")
        if cfg.format == "harbor":
            from integrations.harbor import HarborRuntime

            return HarborRuntime(source_path)
        return LocalRuntime(_spawn_target(source_path))
    if cfg.runtime == "hud":
        require_api_key("run HUD runtime tunnel evals")
        return HUDRuntime()
    if cfg.runtime is not None and cfg.runtime.startswith("tcp://"):
        return Runtime(cfg.runtime)
    hud_console.error(
        f"Unknown runtime {cfg.runtime!r}. Use 'local', 'hud', a tcp:// url, or --remote."
    )
    raise typer.Exit(1)


async def _run_evaluation(cfg: EvalConfig) -> Any:
    """Run evaluation on the Env/Task/Taskset/Run flow.

    Loads a ``Taskset`` from a Python source or JSON/JSONL taskset and runs it
    on the configured placement (default: spawned local substrates — each
    rollout serves its own row's env, so mixed-env tasksets are one job).
    Returns the ``Job`` receipt containing the live execution ``Run`` results.
    """
    if cfg.source is None or cfg.agent_type is None:
        raise ValueError("source and agent_type must be set")

    source_path = Path(cfg.source)
    is_local = source_path.exists()
    if is_local:
        hud_console.info(f"Loading tasks from: {cfg.source}")
        try:
            taskset = _load_local_taskset(source_path, cfg.format)
        except Exception as e:
            hud_console.error(f"Failed to load tasks from {cfg.source}: {e}")
            raise typer.Exit(1) from e
    else:
        from hud.eval import Taskset

        hud_console.info(f"Loading platform taskset: {cfg.source}")
        try:
            taskset = Taskset.from_api(cfg.source)
        except ValueError as e:
            hud_console.error(
                f"Task source not found: {cfg.source}. It is neither a local file nor a "
                "platform taskset (by name or id). Pass a tasks file (.py/.json/.jsonl) "
                "or an existing taskset name."
            )
            raise typer.Exit(1) from e

    if not taskset:
        hud_console.error(
            f"No runnable Tasks found in {cfg.source}. Define a `hud.Environment` with "
            "`@env.template` and expose Tasks (for example, `t = my_task(arg=...)`)."
        )
        raise typer.Exit(1)

    if cfg.task_ids:
        wanted = set(cfg.task_ids)
        taskset = Taskset(
            taskset.name,
            (
                task
                for index, (slug, task) in enumerate(taskset.items())
                if slug in wanted or task.id in wanted or str(index) in wanted
            ),
        )
        if not taskset:
            hud_console.error(f"No tasks matching: {', '.join(cfg.task_ids)}")
            raise typer.Exit(1)
        hud_console.info(f"Filtered to {len(taskset)} task(s)")
    elif not cfg.all:
        tasks = list(taskset)
        total = len(tasks)
        taskset = Taskset(taskset.name, [tasks[0]])
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
    placement = _resolve_placement(cfg, source_path if is_local else None)

    job = await taskset.run(
        agent,
        runtime=placement,
        group=cfg.group_size,
        max_concurrent=cfg.max_concurrent,
    )
    if job.runs and settings.telemetry_enabled and settings.api_key:
        hud_console.info(f"{settings.hud_web_url}/jobs/{job.id}")

    return job


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
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    gateway: bool = typer.Option(
        False, "--gateway", "-g", help="Route LLM API calls through HUD Gateway"
    ),
    format: str | None = typer.Option(
        None,
        "--format",
        help="Task source format: hud (default) or harbor.",
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
        hud eval ./harbor_tasks claude --format harbor # Run Harbor task dirs locally
        hud eval tasks.json claude-sonnet-4-6 --runtime hud  # Use HUD runtime tunnel
        hud eval tasks.json claude-sonnet-4-6 --remote       # Execute rollout remotely
    """
    hud_console.info("Initializing evaluation...")

    if from_json is not None:
        try:
            cfg = EvalConfig.model_validate_json(from_json.read_text(encoding="utf-8"))
        except Exception as e:
            hud_console.error(f"Failed to load JSON config from {from_json}: {e}")
            raise typer.Exit(1) from None
    else:
        cfg = EvalConfig.load()

    try:
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
            format=format,
            runtime=runtime,
            remote=remote,
        )
    except ValueError as e:
        hud_console.error(str(e))
        raise typer.Exit(1) from None

    if cfg.source is None:
        try:
            from hud.cli.utils.tasks import find_tasks_file

            cfg = cfg.model_copy(
                update={"source": find_tasks_file(None, msg="Select a tasks file")}
            )
            hud_console.success(f"Selected: {cfg.source}")
        except Exception:
            hud_console.error("No source provided and no task files found")
            raise typer.Exit(1) from None

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

    if not yes and not hud_console.confirm("Proceed?"):
        hud_console.info("Cancelled.")
        raise typer.Exit(1)

    start_time = time.time()
    try:
        job = asyncio.run(_run_evaluation(cfg))
    except ValueError as e:
        hud_console.error(str(e))
        raise typer.Exit(1) from None
    elapsed = time.time() - start_time

    runs = job.runs
    if runs:
        from hud.cli.utils.display import display_runs

        display_runs(runs, name=cfg.source or "", elapsed=elapsed)
