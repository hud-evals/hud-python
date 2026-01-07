"""Multi-Agent System

This module implements a hierarchical multi-agent system with:
- Agent-as-Tool pattern: Sub-agents exposed as tools to main agent
- YAML configuration: Define agents in YAML files
- Structured returns: Type-safe sub-agent results with make_schema()
- Execution logging: Full trace in YAML log files

Quick Start:
    ```python
    import hud
    from hud.multi_agent import MultiAgentRunner

    async def main():
        async with hud.eval(task, name="multi-agent") as ctx:
            runner = MultiAgentRunner(
                config_dir=Path("agents/"),
                ctx=ctx,
                workspace=Path("./workspace"),
            )

            result = await runner.run(
                task="Build a REST API with authentication",
                max_steps=50,
            )

            print(f"Reward: {result.reward}")
            print(f"Files: {result.files}")
    ```
"""

# Schemas - structured return types for sub-agents
from hud.multi_agent.schemas import (
    AgentResultBase,
    CodeIssue,
    CodeResult,
    FileChange,
    GenericResult,
    IssueSeverity,
    PlannedTask,
    PlanResult,
    ResearchResult,
    ReviewResult,
    Source,
    SubAgentResult,
    TaskStatus,
    make_schema,
)

# SubAgent - factory for creating sub-agents
from hud.multi_agent.sub_agent import SubAgentConfig, create_sub_agent

# Config - YAML configuration loading
from hud.multi_agent.config import (
    AgentConfig,
    AgentToolConfig,
    ConfigLoader,
    MultiAgentConfig,
    ReturnsConfig,
    SCHEMA_REGISTRY,
    load_config,
    register_schema,
)

# Logger - execution trace logging
from hud.multi_agent.logger import StepLogger

# Runner - main orchestration
from hud.multi_agent.runner import MultiAgentRunner, RunResult

__all__ = [
    # Schemas - structured returns
    "AgentResultBase",
    "CodeIssue",
    "CodeResult",
    "FileChange",
    "GenericResult",
    "IssueSeverity",
    "PlannedTask",
    "PlanResult",
    "ResearchResult",
    "ReviewResult",
    "Source",
    "SubAgentResult",
    "TaskStatus",
    "make_schema",
    # SubAgent
    "SubAgentConfig",
    "create_sub_agent",
    # Config
    "AgentConfig",
    "AgentToolConfig",
    "ConfigLoader",
    "MultiAgentConfig",
    "ReturnsConfig",
    "SCHEMA_REGISTRY",
    "load_config",
    "register_schema",
    # Logger
    "StepLogger",
    # Runner
    "MultiAgentRunner",
    "RunResult",
]
