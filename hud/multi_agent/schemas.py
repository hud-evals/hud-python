"""Pydantic schemas for structured agent returns.

These schemas define the structured output format for each sub-agent type.
Only these structured results are returned to the main agent, preventing
context pollution from intermediate conversation history.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Base Schema
# =============================================================================


class AgentResultBase(BaseModel):
    """Base class for all agent result schemas."""

    success: bool = Field(default=True, description="Whether the task completed successfully")
    error: str | None = Field(default=None, description="Error message if task failed")
    duration_ms: float | None = Field(default=None, description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When result was produced")
    
    # Tool call logging - these fields capture the tools used by the agent
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list, 
        description="List of tool calls made by the agent during execution"
    )
    tool_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of tool results received by the agent"
    )


# =============================================================================
# Research Agent Results
# =============================================================================


class Source(BaseModel):
    """A source reference from research."""

    title: str
    url: str | None = None
    path: str | None = None  # For file-based sources
    snippet: str | None = None
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)


class ResearchResult(AgentResultBase):
    """Structured result from a research sub-agent.

    Example:
        {
            "summary": "Python's GIL prevents true parallelism...",
            "sources": [{"title": "Python Docs", "url": "..."}],
            "confidence": 0.85,
            "key_findings": ["GIL limits CPU-bound tasks", ...]
        }
    """

    summary: str = Field(default="", description="Concise summary of research findings")
    sources: list[Source] = Field(default_factory=list, description="Sources referenced")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in findings")
    key_findings: list[str] = Field(default_factory=list, description="Bullet points of key findings")
    follow_up_questions: list[str] = Field(
        default_factory=list, description="Questions that need further research"
    )


# =============================================================================
# Code Agent Results
# =============================================================================


class FileChange(BaseModel):
    """A file that was created or modified."""

    path: str
    action: str = Field(default="created", description="created, modified, or deleted")
    language: str | None = None
    lines_changed: int | None = None


class CodeResult(AgentResultBase):
    """Structured result from a coding sub-agent.

    Example:
        {
            "explanation": "Created REST API with FastAPI...",
            "files_created": [{"path": "api/main.py", "action": "created"}],
            "code_snippet": "from fastapi import FastAPI...",
            "tests_passed": True
        }
    """

    explanation: str = Field(default="", description="What was done and why")
    files_created: list[FileChange] = Field(default_factory=list, description="Files created/modified")
    code_snippet: str | None = Field(default=None, description="Key code snippet for context")
    tests_passed: bool | None = Field(default=None, description="Whether tests passed (if run)")
    dependencies_added: list[str] = Field(default_factory=list, description="New dependencies added")
    commands_run: list[str] = Field(default_factory=list, description="Shell commands executed")

    @field_validator("files_created", mode="before")
    @classmethod
    def normalize_files(cls, v: Any) -> list[dict[str, Any]]:
        """Accept strings or dicts for files_created - LLMs often simplify."""
        if not isinstance(v, list):
            return []
        normalized = []
        for item in v:
            if isinstance(item, str):
                normalized.append({"path": item, "action": "created"})
            elif isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append(item)  # Let pydantic handle it
        return normalized


# =============================================================================
# Review Agent Results
# =============================================================================


class IssueSeverity(str, Enum):
    """Severity level for code review issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class CodeIssue(BaseModel):
    """An issue found during code review."""

    severity: IssueSeverity
    file: str
    line: int | None = None
    description: str
    suggestion: str | None = None
    category: str | None = None  # e.g., "security", "performance", "style"


class ReviewResult(AgentResultBase):
    """Structured result from a code review sub-agent.

    Example:
        {
            "summary": "Found 2 critical issues and 5 suggestions",
            "issues": [{"severity": "critical", "file": "auth.py", ...}],
            "approved": False,
            "score": 0.6
        }
    """

    summary: str = Field(default="", description="Overall review summary")
    issues: list[CodeIssue] = Field(default_factory=list, description="Issues found")
    approved: bool = Field(default=False, description="Whether code is approved")
    score: float = Field(default=0.5, ge=0.0, le=1.0, description="Quality score")
    suggestions: list[str] = Field(default_factory=list, description="General improvement suggestions")
    security_concerns: list[str] = Field(default_factory=list, description="Security-related concerns")

    @field_validator("issues", mode="before")
    @classmethod
    def normalize_issues(cls, v: Any) -> list[dict[str, Any]]:
        """Accept strings or dicts for issues - LLMs often simplify."""
        if not isinstance(v, list):
            return []
        normalized = []
        for item in v:
            if isinstance(item, str):
                # Simple string becomes a medium-severity issue
                normalized.append({"severity": "medium", "file": "unknown", "description": item})
            elif isinstance(item, dict):
                # Ensure severity is valid
                if "severity" in item and isinstance(item["severity"], str):
                    item["severity"] = item["severity"].lower()
                normalized.append(item)
            else:
                normalized.append(item)
        return normalized


# =============================================================================
# Plan Agent Results
# =============================================================================


class TaskStatus(str, Enum):
    """Status of a planned task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class PlannedTask(BaseModel):
    """A task in the execution plan."""

    id: str
    description: str
    agent: str = Field(description="Which agent should handle this task")
    dependencies: list[str] = Field(default_factory=list, description="IDs of tasks that must complete first")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    priority: int = Field(default=1, ge=1, le=5)
    estimated_complexity: str | None = None  # "simple", "moderate", "complex"


class PlanResult(AgentResultBase):
    """Structured result from a planning sub-agent.

    Example:
        {
            "goal": "Build REST API with authentication",
            "tasks": [{"id": "1", "description": "Set up project", "agent": "coder"}],
            "estimated_steps": 10
        }
    """

    goal: str = Field(default="", description="The high-level goal being planned")
    tasks: list[PlannedTask] = Field(default_factory=list, description="Ordered list of tasks")
    estimated_steps: int = Field(default=1, ge=1, description="Estimated total steps")
    risks: list[str] = Field(default_factory=list, description="Potential risks or blockers")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions made in planning")

    @field_validator("tasks", mode="before")
    @classmethod
    def normalize_tasks(cls, v: Any) -> list[dict[str, Any]]:
        """Accept strings or dicts for tasks - LLMs often simplify."""
        if not isinstance(v, list):
            return []
        normalized = []
        for i, item in enumerate(v):
            if isinstance(item, str):
                # Simple string becomes a task description
                normalized.append({"id": str(i + 1), "description": item, "agent": "coder"})
            elif isinstance(item, dict):
                # Ensure id exists
                if "id" not in item:
                    item["id"] = str(i + 1)
                if "agent" not in item:
                    item["agent"] = "coder"
                normalized.append(item)
            else:
                normalized.append(item)
        return normalized


# =============================================================================
# Generic/Custom Results
# =============================================================================


class GenericResult(AgentResultBase):
    """Generic result for custom sub-agents.

    Use this when creating new agent types that don't fit predefined schemas.
    """

    output: str = Field(default="", description="Main output content")
    data: dict[str, Any] = Field(default_factory=dict, description="Arbitrary structured data")
    files: list[str] = Field(default_factory=list, description="File paths involved")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Sub-Agent Minimal Result (Token Optimized)
# =============================================================================


class SubAgentResult(BaseModel):
    """Minimal result returned from sub-agent to parent.

    Following the following principles:
    - Only essential info returns to parent (saves ~90% tokens)
    - Detailed execution trace written to log_file (restorable)
    - Errors stay visible
    - Artifacts are file paths only, not content

    Example:
        {
            "output": "Created revenue chart visualization",
            "success": True,
            "error": None,
            "artifacts": ["./workspace/revenue.png", "./workspace/chart.py"],
            "summary": "Executed 5 tools: read_file, bash, str_replace_based_edit_tool",
            "log_file": ".logs/abc123/coder.yaml"
        }
    """

    output: str = Field(default="", description="Natural language summary of what was accomplished")
    success: bool = Field(default=True, description="Whether the task completed successfully")
    error: str | None = Field(default=None, description="Error message if failed (KEEP)")
    artifacts: list[str] = Field(
        default_factory=list, description="File paths created/modified (not content)"
    )
    summary: str = Field(default="", description="Brief action summary, e.g. 'Executed 5 tools'")
    log_file: str | None = Field(
        default=None, description="Path to detailed YAML execution log"
    )
    duration_ms: float | None = Field(default=None, description="Execution time in milliseconds")

    # NOT included (written to log_file instead):
    # - tool_calls: list[dict] - full execution trace
    # - tool_results: list[dict] - raw tool outputs
    # - available_tools: list[str] - tools the agent had access to


# =============================================================================
# Schema Factory Helper
# =============================================================================


def make_schema(
    name: str,
    __base__: type[BaseModel] | None = None,
    **fields: Any,
) -> type[BaseModel]:
    """Create and register a custom schema in one line.

    This is the easiest way to define a custom return schema for your sub-agents.
    The schema is automatically registered so you can reference it by name in YAML.

    Args:
        name: Schema name (used in YAML config's `returns.schema` field)
        __base__: Base class (defaults to AgentResultBase which includes
                  success, error, duration_ms, timestamp fields)
        **fields: Field definitions. Can be:
            - Just a type: `insights=list[str]` (uses smart defaults)
            - A tuple of (type, default): `score=(float, 0.5)`
            - A tuple with Field: `items=(list[str], Field(default_factory=list))`

    Returns:
        The created Pydantic model class (also registered in SCHEMA_REGISTRY)

    Example:
        ```python
        from hud.multi_agent import make_schema

        # Simple schema - one line!
        DataAnalysisResult = make_schema(
            "DataAnalysisResult",
            insights=list[str],           # List field, defaults to []
            chart_path=(str | None, None), # Optional field
            metrics=dict[str, float],      # Dict field, defaults to {}
            confidence=(float, 0.8),       # Float with default
        )

        # Now use in YAML:
        # agents/analyst.yaml
        # returns:
        #   schema: DataAnalysisResult
        ```

    Advanced example with custom base:
        ```python
        # If you don't want AgentResultBase fields (success, error, etc.)
        from pydantic import BaseModel
        
        SimpleResult = make_schema(
            "SimpleResult",
            __base__=BaseModel,  # Plain Pydantic, no extra fields
            answer=str,
            confidence=float,
        )
        ```
    """
    from typing import get_args, get_origin

    from pydantic import create_model

    # Import here to avoid circular import
    from hud.multi_agent.config import SCHEMA_REGISTRY

    # Default to AgentResultBase for common fields
    if __base__ is None:
        __base__ = AgentResultBase

    # Normalize fields to Pydantic's (type, default) format
    field_definitions: dict[str, Any] = {}

    for field_name, field_spec in fields.items():
        if isinstance(field_spec, tuple) and len(field_spec) == 2:
            # User provided (type, default)
            field_type, default = field_spec
        else:
            # User provided just a type - infer sensible default
            field_type = field_spec
            origin = get_origin(field_type)

            if origin is list:
                default = Field(default_factory=list)
            elif origin is dict:
                default = Field(default_factory=dict)
            elif origin is set:
                default = Field(default_factory=set)
            elif field_type is str:
                default = ""
            elif field_type is bool:
                default = False
            elif field_type is int:
                default = 0
            elif field_type is float:
                default = 0.0
            else:
                # Check if it's Optional (Union with None)
                args = get_args(field_type)
                if type(None) in args:
                    default = None
                else:
                    # Required field
                    default = ...

        field_definitions[field_name] = (field_type, default)

    # Create the Pydantic model dynamically
    schema_cls = create_model(name, __base__=__base__, **field_definitions)

    # Auto-register so YAML can reference by name
    SCHEMA_REGISTRY[name] = schema_cls

    return schema_cls


__all__ = [
    # Agent Results
    "AgentResultBase",
    "ResearchResult",
    "Source",
    "CodeResult",
    "FileChange",
    "ReviewResult",
    "CodeIssue",
    "IssueSeverity",
    "PlanResult",
    "PlannedTask",
    "TaskStatus",
    "GenericResult",
    "SubAgentResult",
    # Schema Factory
    "make_schema",
]

