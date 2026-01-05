"""Multi-Agent System

This module implements a hierarchical multi-agent system with:
- Agent-as-Tool pattern: Sub-agents exposed as tools to main agent
- CodeAct: Agent writes Python code instead of JSON tool calls
- Filesystem as Memory: grep/glob search, not vector DB
- Append-only Context: KV cache optimization

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

# Schemas
from hud.multi_agent.schemas import (
    AgentResultBase,
    Checkpoint,
    CodeIssue,
    CodeResult,
    ContextEntry,
    ContextEntryType,
    FileChange,
    GenericResult,
    IssueSeverity,
    PlannedTask,
    PlanResult,
    ResearchResult,
    ReviewResult,
    Source,
    StepLog,
    TaskStatus,
)

# Context
from hud.multi_agent.context import AppendOnlyContext

# Memory
from hud.multi_agent.memory import FilesystemMemory, MemoryEntry, SearchResult

# Compaction
from hud.multi_agent.compaction import ContextCompactor, ContextOffloader

# CodeAct
from hud.multi_agent.codeact import CodeActExecutor, ExecutionResult, SandboxExecutor

# Agent-as-Tool
from hud.multi_agent.agent_tool import (
    AgentToolRegistry,
    agent_as_tool,
    agent_tools,
    register_agent_tool,
)

# SubAgent
from hud.multi_agent.sub_agent import SimpleSubAgent, SubAgent, SubAgentConfig

# Config
from hud.multi_agent.config import (
    AgentConfig,
    AgentToolConfig,
    CodeActConfig,
    ConfigLoader,
    MultiAgentConfig,
    ReturnsConfig,
    SCHEMA_REGISTRY,
    load_config,
    register_schema,
)

# Logger
from hud.multi_agent.logger import StepLogger

# Runner
from hud.multi_agent.runner import MultiAgentRunner, RunResult

__all__ = [
    # Schemas
    "AgentResultBase",
    "Checkpoint",
    "CodeIssue",
    "CodeResult",
    "ContextEntry",
    "ContextEntryType",
    "FileChange",
    "GenericResult",
    "IssueSeverity",
    "PlannedTask",
    "PlanResult",
    "ResearchResult",
    "ReviewResult",
    "Source",
    "StepLog",
    "TaskStatus",
    # Context
    "AppendOnlyContext",
    # Memory
    "FilesystemMemory",
    "MemoryEntry",
    "SearchResult",
    # Compaction
    "ContextCompactor",
    "ContextOffloader",
    # CodeAct
    "CodeActExecutor",
    "ExecutionResult",
    "SandboxExecutor",
    # Agent-as-Tool
    "AgentToolRegistry",
    "agent_as_tool",
    "agent_tools",
    "register_agent_tool",
    # SubAgent
    "SimpleSubAgent",
    "SubAgent",
    "SubAgentConfig",
    # Config
    "AgentConfig",
    "AgentToolConfig",
    "CodeActConfig",
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
