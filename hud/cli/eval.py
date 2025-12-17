"""HUD evaluation command for running tasks and datasets.

Config Override Order: CLI arguments > .hud_eval.toml > defaults
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import questionary
import typer
from pydantic import BaseModel, Field, field_validator
from rich import box
from rich.table import Table

from hud.settings import settings
from hud.types import AgentType
from hud.utils.env import resolve_env_vars
from hud.utils.hud_console import HUDConsole

# Pattern to detect AWS Bedrock inference profile ARNs
_BEDROCK_ARN_PATTERN = re.compile(r"^arn:aws:bedrock:[a-z0-9-]+:\d+:inference-profile/.+$")


def _is_bedrock_arn(model: str | None) -> bool:
    """Check if a model string is a Bedrock inference profile ARN."""
    return model is not None and bool(_BEDROCK_ARN_PATTERN.match(model))


if TYPE_CHECKING:
    from hud.agents.base import MCPAgent

logger = logging.getLogger(__name__)
hud_console = HUDConsole()

_CONFIG_PATH = ".hud_eval.toml"


@dataclass(frozen=True)
class AgentPreset:
    """A preset agent configuration combining agent type, model, and optional config."""

    name: str
    agent_type: AgentType
    model: str | None = None
    agent_config: dict[str, Any] | None = None


# Built-in presets for the interactive picker
_AGENT_PRESETS: list[AgentPreset] = [
    # Native agents (use provider SDKs directly)
    AgentPreset("Claude Sonnet 4.5", AgentType.CLAUDE, "claude-sonnet-4-5"),
    AgentPreset("GPT-5", AgentType.OPENAI, "gpt-5"),
    AgentPreset("Operator (OpenAI Computer Use)", AgentType.OPERATOR, "computer-use-preview"),
    AgentPreset("Gemini 3 Pro Preview", AgentType.GEMINI, "gemini-3-pro-preview"),
    AgentPreset(
        "Gemini CUA (Gemini Computer Use)",
        AgentType.GEMINI_CUA,
        "gemini-2.5-computer-use-preview",
    ),
    # HUD Gateway presets (models via HUD Inference API)
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
        "GLM-4.5V (Z-AI)",
        AgentType.OPENAI_COMPATIBLE,
        "z-ai/glm-4.5v",
        {"openai_compatible": {"base_url": settings.hud_gateway_url, "model_name": "GLM-4.5V"}},
    ),
]

_DEFAULT_CONFIG_TEMPLATE = """# HUD Eval Configuration
# Command-line arguments override these settings

[eval]
# source = "hud-evals/SheetBench-50"
# agent = "claude"
# full = false
# max_concurrent = 30
# max_steps = 10
# group_size = 1
# task_ids = ["task_1", "task_2"]
# verbose = true
# very_verbose = true
# auto_respond = true
# gateway = false  # Route LLM API calls through HUD Gateway

[agent]
# allowed_tools = ["computer", "playwright"]
# disallowed_tools = []

[claude]
# model = "claude-sonnet-4-5"
# max_tokens = 16384
# use_computer_beta = true

[openai]
# model = "gpt-4o"
# temperature = 0.7
# max_output_tokens = 4096

[gemini]
# model = "gemini-2.5-pro"
# temperature = 1.0
# top_p = 0.95

[gemini_cua]
# model = "gemini-2.5-computer-use-preview"
# temperature = 1.0
# top_p = 0.95
# excluded_predefined_functions = []

[openai_compatible]
# base_url = "http://localhost:8000/v1"
# model = "my-model"
"""

# Agent type -> (settings attr, env var name)
_API_KEY_REQUIREMENTS: dict[AgentType, tuple[str, str]] = {
    AgentType.CLAUDE: ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    AgentType.GEMINI: ("gemini_api_key", "GEMINI_API_KEY"),
    AgentType.GEMINI_CUA: ("gemini_api_key", "GEMINI_API_KEY"),
    AgentType.OPENAI: ("openai_api_key", "OPENAI_API_KEY"),
    AgentType.OPERATOR: ("openai_api_key", "OPENAI_API_KEY"),
}


class EvalConfig(BaseModel):
    """Configuration for hud eval command."""

    # Class-level registry
    _agent_classes: ClassVar[dict[AgentType, type["MCPAgent"]]] = {}

    # Fields loaded from [eval] section
    _EVAL_FIELDS: ClassVar[set[str]] = {
        "source",
        "agent_type",
        "task_ids",
        "full",
        "max_concurrent",
        "max_steps",
        "verbose",
        "very_verbose",
        "group_size",
        "remote",
        "auto_respond",
        "quiet",
        "gateway",
    }
    # Fields loaded from [agent] section
    _AGENT_FIELDS: ClassVar[set[str]] = {"allowed_tools", "disallowed_tools"}

    # Eval settings
    source: str | None = None
    agent_type: AgentType | None = None
    model: str | None = None
    task_ids: list[str] | None = None
    full: bool = False
    max_concurrent: int = 30
    max_steps: int | None = None
    verbose: bool = False
    very_verbose: bool = False
    auto_respond: bool | None = None  # Continue without prompting (default: True for --full)
    group_size: int = 1
    remote: bool = False
    quiet: bool = False  # Suppress opening browser for eval links
    gateway: bool = False  # Use HUD Gateway for LLM API calls

    # Base agent config (these merge with task's agent_config)
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None

    agent_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_type", mode="before")
    @classmethod
    def _parse_agent_type(cls, v: Any) -> AgentType | None:
        """Convert string agent name to AgentType enum."""
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

    def validate_api_keys(self) -> None:
        """Validate required API keys for the selected agent. Raises typer.Exit on failure."""
        if self.agent_type is None:
            return

        if self.remote:
            if not settings.api_key:
                hud_console.error("HUD_API_KEY is required for remote execution")
                hud_console.info("Set it: hud set HUD_API_KEY=your-key-here")
                raise typer.Exit(1)
            return

        # Gateway mode only requires HUD_API_KEY
        if self.gateway:
            if not settings.api_key:
                hud_console.error("HUD_API_KEY is required for gateway mode")
                hud_console.info("Set it: hud set HUD_API_KEY=your-key-here")
                raise typer.Exit(1)
            return

        if self.agent_type == AgentType.OPENAI_COMPATIBLE:
            # Check both CLI --model and config file model
            config_model = self.agent_config.get("openai_compatible", {}).get("model")
            if not self.model and not config_model:
                hud_console.error(
                    "Model name is required for OpenAI compatible agent. "
                    "Use --model or set model in [openai_compatible] section of .hud_eval.toml"
                )
                raise typer.Exit(1)
        elif self.agent_type == AgentType.CLAUDE and _is_bedrock_arn(self.model):
            missing_aws = (
                not settings.aws_access_key_id
                or not settings.aws_secret_access_key
                or not settings.aws_region
            )
            if missing_aws:
                hud_console.error(
                    "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION "
                    "are required for AWS Bedrock"
                )
                raise typer.Exit(1)
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

        if self.allowed_tools:
            kwargs["allowed_tools"] = self.allowed_tools
        if self.disallowed_tools:
            kwargs["disallowed_tools"] = self.disallowed_tools

        # Apply agent-specific config
        agent_key = self.agent_type.value
        if agent_key in self.agent_config:
            agent_cfg = dict(self.agent_config[agent_key])
            kwargs.update(agent_cfg)

        # CLI --model always wins
        if self.model:
            kwargs["model"] = self.model

        if self.agent_type == AgentType.OPENAI_COMPATIBLE:
            base_url = kwargs.get("base_url", "")
            if "api_key" not in kwargs:
                # Use HUD API key for gateway, otherwise fall back to OpenAI API key
                if settings.hud_gateway_url in base_url:
                    kwargs["api_key"] = settings.api_key
                elif settings.openai_api_key:
                    kwargs["api_key"] = settings.openai_api_key

        # Auto-detect Bedrock when Claude is selected with a Bedrock ARN
        # Check both model and checkpoint_name for ARN patterns
        bedrock_arn_detected = _is_bedrock_arn(kwargs.get("model")) or _is_bedrock_arn(
            kwargs.get("checkpoint_name")
        )
        if self.agent_type == AgentType.CLAUDE and bedrock_arn_detected:
            missing_aws = (
                not settings.aws_access_key_id
                or not settings.aws_secret_access_key
                or not settings.aws_region
            )
            if missing_aws:
                hud_console.error(
                    "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION "
                    "are required for AWS Bedrock"
                )
                raise typer.Exit(1)

            from anthropic import AsyncAnthropicBedrock

            kwargs["model_client"] = AsyncAnthropicBedrock(
                aws_access_key=settings.aws_access_key_id,
                aws_secret_key=settings.aws_secret_access_key,
                aws_region=settings.aws_region or "us-east-1",
            )
            hud_console.info("🔧 Using AWS Bedrock (detected ARN in model)")

        kwargs["verbose"] = self.verbose or self.very_verbose

        if self.agent_type in (
            AgentType.CLAUDE,
            AgentType.OPENAI,
            AgentType.OPERATOR,
            AgentType.GEMINI,
            AgentType.GEMINI_CUA,
        ):
            kwargs["validate_api_key"] = False

        # Configure gateway mode - route LLM API calls through HUD gateway
        if self.gateway:
            hud_api_key = settings.api_key
            if not hud_api_key:
                raise typer.Exit(1)  # Already validated in validate_api_keys()

            if self.agent_type == AgentType.CLAUDE:
                from anthropic import AsyncAnthropic

                kwargs["model_client"] = AsyncAnthropic(
                    api_key=hud_api_key,
                    base_url=settings.hud_gateway_url,
                )
                hud_console.info("🌐 Using HUD Gateway for Claude API")
            elif self.agent_type in (AgentType.OPENAI, AgentType.OPERATOR):
                from openai import AsyncOpenAI

                kwargs["model_client"] = AsyncOpenAI(
                    api_key=hud_api_key,
                    base_url=settings.hud_gateway_url,
                )
                hud_console.info("🌐 Using HUD Gateway for OpenAI API")
            elif self.agent_type == AgentType.OPENAI_COMPATIBLE:
                from openai import AsyncOpenAI

                kwargs["openai_client"] = AsyncOpenAI(
                    api_key=hud_api_key,
                    base_url=settings.hud_gateway_url,
                )
                hud_console.info("🌐 Using HUD Gateway for OpenAI-compatible API")
            elif self.agent_type in (AgentType.GEMINI, AgentType.GEMINI_CUA):
                from google import genai
                from google.genai.types import HttpOptions

                kwargs["model_client"] = genai.Client(
                    api_key="PLACEHOLDER",
                    http_options=HttpOptions(
                        api_version="v1beta",
                        base_url=settings.hud_gateway_url,
                        headers={"Authorization": f"Bearer {hud_api_key}"},
                    ),
                )
                hud_console.info("🌐 Using HUD Gateway for Gemini API")

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

        toml_data = resolve_env_vars(toml_data)

        # Extract sections
        eval_section = toml_data.get("eval", {})
        agent_section = toml_data.get("agent", {})

        # Build config data
        data: dict[str, Any] = {}

        # Eval settings (map 'agent' -> 'agent_type')
        if "agent" in eval_section:
            data["agent_type"] = eval_section["agent"]
        for key in cls._EVAL_FIELDS:
            if key in eval_section:
                data[key] = eval_section[key]

        # Agent base config
        for key in cls._AGENT_FIELDS:
            if key in agent_section:
                data[key] = agent_section[key]

        # Agent-specific configs (claude, openai, gemini, etc.)
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
        agent: str | None = None,
        config: list[str] | None = None,
        allowed_tools: str | None = None,
        disallowed_tools: str | None = None,
        task_ids: str | None = None,
        **cli_args: Any,
    ) -> EvalConfig:
        """Merge CLI args (non-None values override config)."""
        overrides: dict[str, Any] = {}

        if agent is not None:
            overrides["agent_type"] = agent

        # Parse comma-separated lists
        if allowed_tools is not None:
            overrides["allowed_tools"] = [t.strip() for t in allowed_tools.split(",") if t.strip()]
        if disallowed_tools is not None:
            overrides["disallowed_tools"] = [
                t.strip() for t in disallowed_tools.split(",") if t.strip()
            ]
        if task_ids is not None:
            overrides["task_ids"] = [t.strip() for t in task_ids.split(",") if t.strip()]

        overrides.update({k: v for k, v in cli_args.items() if v is not None and v is not False})

        for k in ("full", "verbose", "very_verbose", "remote", "quiet", "gateway"):
            if cli_args.get(k) is True:
                overrides[k] = True
            elif k in overrides and cli_args.get(k) is False:
                del overrides[k]

        if config:
            merged_agent_config = dict(self.agent_config)
            for item in config:
                if "=" in item:
                    key, value = item.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Parse value
                    if value.lower() == "true":
                        parsed_value: Any = True
                    elif value.lower() == "false":
                        parsed_value = False
                    else:
                        try:
                            parsed_value = int(value)
                        except ValueError:
                            try:
                                parsed_value = float(value)
                            except ValueError:
                                parsed_value = value

                    # Handle namespaced keys (e.g., claude.max_tokens)
                    if "." in key:
                        agent_name, param = key.split(".", 1)
                        if agent_name not in merged_agent_config:
                            merged_agent_config[agent_name] = {}
                        merged_agent_config[agent_name][param] = parsed_value
                    else:
                        # Non-namespaced: apply to current agent if set
                        if self.agent_type:
                            agent_name = self.agent_type.value
                            if agent_name not in merged_agent_config:
                                merged_agent_config[agent_name] = {}
                            merged_agent_config[agent_name][key] = parsed_value

            overrides["agent_config"] = merged_agent_config

        return self.model_validate({**self.model_dump(), **overrides})

    def resolve_agent_interactive(self) -> EvalConfig:
        """Prompt user to select an agent preset if not set. Returns updated config."""
        if self.agent_type is not None:
            return self

        # Build choices from presets
        choices: list[dict[str, Any]] = [
            {"name": preset.name, "value": preset} for preset in _AGENT_PRESETS
        ]

        selected: AgentPreset = hud_console.select("Select an agent:", choices=choices, default=0)  # type: ignore[arg-type]

        # Merge preset into config
        updates: dict[str, Any] = {"agent_type": selected.agent_type}
        if selected.model:
            updates["model"] = selected.model
        if selected.agent_config:
            # Merge preset's agent_config with existing
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

        # Core settings
        table.add_row("source", str(self.source or "—"))
        table.add_row("agent", self.agent_type.value)  # type: ignore[union-attr]
        if self.task_ids:
            table.add_row(
                "task_ids", ", ".join(self.task_ids[:5]) + ("..." if len(self.task_ids) > 5 else "")
            )
        table.add_row("full", str(self.full))
        table.add_row("max_steps", str(self.max_steps or (100 if self.full else 10)))
        if not self.remote:
            table.add_row("max_concurrent", str(self.max_concurrent))
        if self.group_size > 1:
            table.add_row("group_size", str(self.group_size))
        # Show auto_respond when it will be true (explicit or via --full)
        effective_auto_respond = self.auto_respond if self.auto_respond is not None else self.full
        if effective_auto_respond:
            table.add_row("auto_respond", "[bold green]True[/bold green]")
        if self.very_verbose:
            table.add_row("very_verbose", "[bold green]True[/bold green]")
        elif self.verbose:
            table.add_row("verbose", "[bold green]True[/bold green]")
        if self.remote:
            table.add_row("remote", "[bold green]True[/bold green] (submitting to platform)")
        if self.gateway:
            table.add_row("gateway", "[bold green]True[/bold green] (routing via HUD Gateway)")

        # Tool filters (only if set)
        if self.allowed_tools:
            table.add_row("allowed_tools", ", ".join(self.allowed_tools))
        if self.disallowed_tools:
            table.add_row("disallowed_tools", ", ".join(self.disallowed_tools))

        # Agent config section
        if self.agent_type:
            table.add_row("", "")
            table.add_row(f"[dim]{self.agent_type.value} config[/dim]", "")

            config_cls = self.agent_type.cls.config_cls
            defaults = config_cls()
            overrides = self.agent_config.get(self.agent_type.value, {})
            skip = {
                "model_client",
                "model_name",
                "validate_api_key",
                "model_config",
                "allowed_tools",
                "disallowed_tools",
                "system_prompt",
                "response_tool_name",
                "append_setup_output",
                "initial_screenshot",
            }

            sensitive_fields = {"api_key", "api_secret", "token", "password", "secret"}

            for name in config_cls.model_fields:
                if name in skip:
                    continue
                # Always show model
                if name == "model":
                    if self.model:
                        value = self.model
                    elif overrides.get("model"):
                        value = overrides["model"]
                    else:
                        value = getattr(defaults, "model", None)
                    table.add_row("  model", str(value) if value else "—")
                elif name in overrides:
                    value = overrides[name]
                    if name in sensitive_fields and value:
                        display_value = f"{str(value)[:4]}****" if len(str(value)) > 4 else "****"
                    else:
                        display_value = str(value)
                    table.add_row(f"  {name}", display_value)

        hud_console.console.print(table)


# =============================================================================
# Evaluation runner
# =============================================================================


async def _run_evaluation(cfg: EvalConfig) -> tuple[list[Any], list[Any]]:
    """Run evaluation with the given config using run_dataset()."""
    from hud.datasets import load_tasks, run_dataset

    if cfg.source is None or cfg.agent_type is None:
        raise ValueError("source and agent_type must be set")

    # Load tasks using unified loader (handles v4→v5 conversion automatically)
    hud_console.info(f"📊 Loading tasks from: {cfg.source}…")
    tasks = load_tasks(cfg.source)

    if not tasks:
        hud_console.error(f"No tasks found in: {cfg.source}")
        raise typer.Exit(1)

    # Filter by task IDs if provided
    if cfg.task_ids:
        id_set = set(cfg.task_ids)
        # Match by task.id or index
        filtered = [t for i, t in enumerate(tasks) if t.id in id_set or str(i) in id_set]
        if not filtered:
            hud_console.error(f"No tasks found matching IDs: {', '.join(cfg.task_ids)}")
            raise typer.Exit(1)
        hud_console.info(f"Filtered to {len(filtered)} task(s) by ID")
        tasks = filtered
    elif not cfg.full:
        # Single task mode (no --full, no --task-ids)
        tasks = [tasks[0]]
        hud_console.info("Using first task (run with --full or --task-ids for more)…")

    hud_console.info(f"Loaded {len(tasks)} task(s)")

    # Prepare agent kwargs
    agent_kwargs = cfg.get_agent_kwargs()
    auto_respond = cfg.auto_respond if cfg.auto_respond is not None else cfg.full
    if auto_respond:
        agent_kwargs = {**agent_kwargs, "auto_respond": True}

    max_steps = cfg.max_steps or (100 if cfg.full else 10)

    # Remote execution - submit to HUD platform
    if cfg.remote:
        # Create a job ID for tracking
        import uuid

        from hud.datasets.utils import submit_rollouts

        job_id = str(uuid.uuid4())
        hud_console.info(
            f"Submitting {len(tasks)} task(s) for remote execution (job_id: {job_id})…"
        )

        await submit_rollouts(
            tasks=tasks,
            job_id=job_id,
            agent_type=cfg.agent_type,
            agent_params=agent_kwargs,
            max_steps=max_steps,
            group_size=cfg.group_size,
        )

        hud_console.success(f"Tasks submitted. View at: https://hud.ai/job/{job_id}")
        return [], tasks

    # Single task mode - show extra info
    if len(tasks) == 1 and cfg.group_size == 1:
        logging.getLogger("hud.agents").setLevel(logging.INFO)
        logging.getLogger("hud.agents.base").setLevel(logging.INFO)
        # Get prompt from args (v4 tasks) or show scenario name
        prompt = tasks[0].args.get("prompt") if tasks[0].args else tasks[0].scenario
        if prompt:
            hud_console.info(f"Prompt: {prompt}")
    else:
        hud_console.info(
            f"🚀 Running evaluation (max_concurrent: {cfg.max_concurrent}, "
            f"group_size: {cfg.group_size})…"
        )

    # Run using run_dataset
    results = await run_dataset(
        tasks,
        cfg.agent_type,
        agent_params=agent_kwargs,
        max_steps=max_steps,
        max_concurrent=cfg.max_concurrent,
        group_size=cfg.group_size,
        quiet=cfg.quiet,
    )

    # Show reward for single task
    if len(tasks) == 1 and cfg.group_size == 1 and results:
        hud_console.success(f"Reward: {results[0].reward}")

    return results, tasks


# =============================================================================
# CLI command
# =============================================================================


def eval_command(
    source: str | None = typer.Argument(None, help="HuggingFace dataset or task JSON file"),
    agent: str | None = typer.Argument(
        None,
        help="Agent: claude, openai, operator, gemini, gemini_cua, openai_compatible, integration_test",  # noqa: E501
    ),
    full: bool = typer.Option(False, "--full", help="Run entire dataset"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name"),
    config: list[str] | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Agent config: key=value"
    ),
    # Task-overridable settings
    allowed_tools: str | None = typer.Option(
        None, "--allowed-tools", help="Comma-separated allowed tools"
    ),
    disallowed_tools: str | None = typer.Option(
        None, "--disallowed-tools", help="Comma-separated disallowed tools"
    ),
    # Eval settings
    max_concurrent: int | None = typer.Option(
        None, "--max-concurrent", help="Max concurrent tasks"
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Max steps per task"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    very_verbose: bool = typer.Option(False, "--very-verbose", "-vv", help="Debug logs"),
    auto_respond: bool | None = typer.Option(
        None,
        "--auto-respond",
        help="Continue without prompting after tool calls (default: True for --full)",
    ),
    group_size: int | None = typer.Option(None, "--group-size", help="Runs per task"),
    task_ids: str | None = typer.Option(None, "--task-ids", help="Comma-separated task IDs to run"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    remote: bool = typer.Option(
        False, "--remote", help="Submit tasks to platform for remote execution"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress opening browser for eval links"
    ),
    gateway: bool = typer.Option(
        False, "--gateway", "-g", help="Route LLM API calls through HUD Gateway"
    ),
) -> None:
    """🚀 Run evaluation on datasets or individual tasks with agents.

    Examples:
        hud eval tasks.json claude
        hud eval hud-evals/SheetBench-50 claude --full
        hud eval tasks.json claude --config max_tokens=32768
        hud eval tasks.json openai --config temperature=0.7
        hud eval tasks.json claude --full --remote  # Remote execution
        hud eval tasks.json claude --gateway  # Route LLM calls through HUD Gateway
    """
    hud_console.info("🔧 Initializing evaluation...")

    # Load config and merge CLI args
    cfg = EvalConfig.load().merge_cli(
        source=source,
        agent=agent,
        model=model,
        full=full,
        max_concurrent=max_concurrent,
        max_steps=max_steps,
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        task_ids=task_ids,
        verbose=verbose,
        very_verbose=very_verbose,
        auto_respond=auto_respond,
        group_size=group_size,
        config=config,
        remote=remote,
        quiet=quiet,
        gateway=gateway,
    )

    # Find source if not provided
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

    # Resolve agent interactively if needed
    cfg = cfg.resolve_agent_interactive()

    # Configure logging
    if cfg.very_verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(message)s")
        logging.getLogger("hud.agents").setLevel(logging.DEBUG)
        # Suppress noisy HTTP client logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    elif cfg.verbose:
        logging.getLogger("hud.agents").setLevel(logging.INFO)

    # Validate API keys
    cfg.validate_api_keys()

    # Display and confirm
    cfg.display()

    if not yes and not questionary.confirm("Proceed?", default=True, qmark="").ask():
        hud_console.info("Cancelled.")
        raise typer.Exit(1)

    # Run
    start_time = time.time()
    try:
        results, tasks = asyncio.run(_run_evaluation(cfg))
    except ValueError as e:
        hud_console.error(str(e))
        raise typer.Exit(1) from None
    elapsed = time.time() - start_time

    if cfg.remote:
        return

    from hud.datasets import display_results

    display_results(results, tasks=tasks, elapsed=elapsed, show_details=len(results) <= 50)
