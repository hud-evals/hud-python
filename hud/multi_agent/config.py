"""YAML configuration loader for agent definitions.

This module loads agent configurations from YAML files, supporting:
- Main orchestrator agent with sub-agent tools
- Specialist sub-agents with tool configurations
- Structured return schemas for typed sub-agent outputs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from hud.multi_agent.schemas import (
    CodeResult,
    GenericResult,
    PlanResult,
    ResearchResult,
    ReviewResult,
)

logger = logging.getLogger(__name__)

# Map of schema name strings to actual classes
SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "ResearchResult": ResearchResult,
    "CodeResult": CodeResult,
    "ReviewResult": ReviewResult,
    "PlanResult": PlanResult,
    "GenericResult": GenericResult,
}


def register_schema(name: str, schema_cls: type[BaseModel]) -> None:
    """Register a custom schema for use in YAML configs.

    Args:
        name: Schema name to use in YAML
        schema_cls: Pydantic model class
    """
    SCHEMA_REGISTRY[name] = schema_cls


class AgentToolConfig(BaseModel):
    """Configuration for a sub-agent exposed as a tool."""

    name: str
    agent: str  # Reference to another agent config
    description: str = ""


class ReturnsConfig(BaseModel):
    """Return schema configuration.
    
    Specifies which schema to use for structured sub-agent output.
    The schema's fields are automatically extracted and included in
    the agent's system prompt to guide JSON output.
    """

    schema_name: str = Field(alias="schema", default="GenericResult")


class AgentConfig(BaseModel):
    """Configuration for a single agent.
    
    Note: This intentionally doesn't extend BaseAgentConfig because:
    1. It's for YAML config loading with different defaults
    2. BaseAgentConfig has extra="forbid" while this needs flexibility
    3. The fields are converted to SubAgentConfig when creating agents
    """

    name: str
    type: str = "specialist"  # "orchestrator" or "specialist"
    model: str = "anthropic/claude-sonnet-4-5"

    system_prompt: str = ""  # Different default than BaseAgentConfig (None vs "")

    # For orchestrator: sub-agents as tools
    agent_tools: list[AgentToolConfig] = []

    # For specialist: tools available
    tools: list[str] = []

    # Return schema - specifies structured output format for sub-agent results
    returns: ReturnsConfig | None = None

    # Execution limits
    max_steps: int = 10

    def get_return_schema(self) -> type[BaseModel]:
        """Get the return schema class."""
        if self.returns:
            schema_name = self.returns.schema_name
        else:
            schema_name = "GenericResult"

        return SCHEMA_REGISTRY.get(schema_name, GenericResult)


class MultiAgentConfig(BaseModel):
    """Configuration for the entire multi-agent system."""

    # Main orchestrator
    main: str = "main"  # Name of main agent

    # All agents
    agents: dict[str, AgentConfig] = {}

    # Global settings
    workspace: str = "./workspace"
    log_dir: str = "./.logs"


class ConfigLoader:
    """Load multi-agent configuration from YAML files.

    Supports:
    - Single file with all agents
    - Directory with one file per agent

    Example directory structure:
        agents/
          main.yaml       # Orchestrator
          researcher.yaml # Sub-agent
          coder.yaml      # Sub-agent

    Example single file:
        agents:
          main:
            type: orchestrator
            ...
          researcher:
            type: specialist
            ...
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize the config loader.

        Args:
            config_path: Path to config file or directory
        """
        self.config_path = Path(config_path)

    def load(self) -> MultiAgentConfig:
        """Load the configuration.

        Returns:
            MultiAgentConfig with all agents loaded
        """
        if self.config_path.is_dir():
            return self._load_from_directory()
        else:
            return self._load_from_file()

    def _load_from_file(self) -> MultiAgentConfig:
        """Load from a single YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Parse agents
        agents = {}
        for name, agent_data in data.get("agents", {}).items():
            agent_data["name"] = name
            agents[name] = AgentConfig.model_validate(agent_data)

        return MultiAgentConfig(
            main=data.get("main", "main"),
            agents=agents,
            workspace=data.get("workspace", "./workspace"),
            log_dir=data.get("log_dir", "./.logs"),
        )

    def _load_from_directory(self) -> MultiAgentConfig:
        """Load from a directory of YAML files."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_path}")

        agents = {}
        main_agent = None

        for yaml_file in self.config_path.glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            if data is None:
                continue

            name = data.get("name", yaml_file.stem)
            data["name"] = name
            agents[name] = AgentConfig.model_validate(data)

            # First orchestrator becomes main
            if data.get("type") == "orchestrator" and main_agent is None:
                main_agent = name

        # Also check for .yml files
        for yml_file in self.config_path.glob("*.yml"):
            with open(yml_file) as f:
                data = yaml.safe_load(f)

            if data is None:
                continue

            name = data.get("name", yml_file.stem)
            data["name"] = name
            agents[name] = AgentConfig.model_validate(data)

            if data.get("type") == "orchestrator" and main_agent is None:
                main_agent = name

        return MultiAgentConfig(
            main=main_agent or "main",
            agents=agents,
        )

    def validate(self) -> list[str]:
        """Validate the configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            config = self.load()
        except Exception as e:
            return [f"Failed to load config: {e}"]

        # Check main agent exists
        if config.main not in config.agents:
            errors.append(f"Main agent '{config.main}' not found in agents")

        # Check agent tool references
        for name, agent in config.agents.items():
            for tool in agent.agent_tools:
                if tool.agent not in config.agents:
                    errors.append(f"Agent '{name}' references unknown agent '{tool.agent}'")

        return errors


def load_config(path: str | Path) -> MultiAgentConfig:
    """Convenience function to load config.

    Args:
        path: Path to config file or directory

    Returns:
        MultiAgentConfig
    """
    loader = ConfigLoader(path)
    return loader.load()


__all__ = [
    "AgentConfig",
    "AgentToolConfig",
    "ConfigLoader",
    "MultiAgentConfig",
    "ReturnsConfig",
    "SCHEMA_REGISTRY",
    "load_config",
    "register_schema",
]

