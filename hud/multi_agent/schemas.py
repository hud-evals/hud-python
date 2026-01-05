"""Pydantic schemas for structured agent returns.

These schemas define the structured output format for each sub-agent type.
Only these structured results are returned to the main agent, preventing
context pollution from intermediate conversation history.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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

    Following Manus context engineering principles:
    - Only essential info returns to parent (saves ~90% tokens)
    - Detailed execution trace written to log_file (restorable)
    - Errors stay visible (per Manus principle #5)
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
    error: str | None = Field(default=None, description="Error message if failed (KEEP per Manus)")
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
# Context Entry Types (for AppendOnlyContext)
# =============================================================================


class ContextEntryType(str, Enum):
    """Types of entries in the append-only context."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE_CONTENT = "file_content"
    FILE_REF = "file_ref"  # Compacted reference
    ERROR = "error"
    SUMMARY = "summary"  # Irreversible summarization


class ContextEntry(BaseModel):
    """A single entry in the append-only context."""

    id: str = Field(description="Unique identifier for this entry")
    type: ContextEntryType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # For file-related entries
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    # For tool-related entries
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None

    # Metadata
    agent_id: str | None = None
    token_count: int | None = None
    compacted: bool = Field(default=False, description="Whether this entry has been compacted")

    def render(self) -> str:
        """Render this entry as a string for the context window."""
        if self.type == ContextEntryType.FILE_REF:
            return f"[File: {self.path} (lines {self.start_line}-{self.end_line})]"
        elif self.type == ContextEntryType.TOOL_CALL:
            return f"[Tool Call: {self.tool_name}({self.tool_args})]"
        elif self.type == ContextEntryType.TOOL_RESULT:
            return f"[Tool Result: {self.content[:200]}...]" if len(self.content) > 200 else f"[Tool Result: {self.content}]"
        elif self.type == ContextEntryType.ERROR:
            return f"[Error: {self.content}]"
        else:
            return self.content


# =============================================================================
# Checkpoint Schema (for resilience)
# =============================================================================


class Checkpoint(BaseModel):
    """Snapshot of system state for crash recovery."""

    run_id: str
    step_number: int
    timestamp: datetime = Field(default_factory=datetime.now)

    # Context state
    context_entries: list[dict[str, Any]]
    frozen_prefix_length: int

    # Filesystem state
    workspace_files: dict[str, str] = Field(
        default_factory=dict, description="path -> content hash"
    )
    memory_index: dict[str, Any] = Field(default_factory=dict)

    # Agent state
    current_agent: str
    pending_task: dict[str, Any] | None = None


# =============================================================================
# Step Log Schema
# =============================================================================


class StepLog(BaseModel):
    """Log entry for a single agent step."""

    # Identity
    step_id: str
    run_id: str
    agent_id: str
    parent_step_id: str | None = None

    # Input
    input_prompt: str
    input_context_size: int  # Token count

    # Output
    output_response: str | None = None
    output_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    output_tool_results: list[dict[str, Any]] = Field(default_factory=list)

    # Timing
    timestamp_start: datetime
    timestamp_end: datetime | None = None
    duration_ms: float | None = None

    # Metadata
    model: str
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None

    # Context health
    context_tokens_before: int | None = None
    context_tokens_after: int | None = None
    compactions_performed: int = 0


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
    # Context Types
    "ContextEntry",
    "ContextEntryType",
    # Checkpoint
    "Checkpoint",
    # Logging
    "StepLog",
]

