"""Tests for multi-agent schemas."""

import pytest
from datetime import datetime

from hud.multi_agent.schemas import (
    ResearchResult,
    CodeResult,
    ReviewResult,
    PlanResult,
    GenericResult,
    Source,
    FileChange,
    CodeIssue,
    IssueSeverity,
    PlannedTask,
    TaskStatus,
    ContextEntry,
    ContextEntryType,
    StepLog,
    Checkpoint,
)


class TestResearchResult:
    """Test ResearchResult schema."""

    def test_basic_result(self):
        """Test creating a basic research result."""
        result = ResearchResult(
            summary="Python is a programming language.",
            confidence=0.9,
        )
        assert result.summary == "Python is a programming language."
        assert result.confidence == 0.9
        assert result.success is True
        assert result.sources == []

    def test_with_sources(self):
        """Test research result with sources."""
        result = ResearchResult(
            summary="Found information",
            sources=[
                Source(title="Wikipedia", url="https://wikipedia.org"),
                Source(title="Local file", path="/docs/readme.md"),
            ],
            confidence=0.85,
            key_findings=["Finding 1", "Finding 2"],
        )
        assert len(result.sources) == 2
        assert result.sources[0].url == "https://wikipedia.org"
        assert len(result.key_findings) == 2


class TestCodeResult:
    """Test CodeResult schema."""

    def test_basic_result(self):
        """Test creating a basic code result."""
        result = CodeResult(
            explanation="Created a function",
            code_snippet="def hello(): pass",
        )
        assert result.explanation == "Created a function"
        assert result.files_created == []

    def test_with_files(self):
        """Test code result with file changes."""
        result = CodeResult(
            explanation="Implemented feature",
            files_created=[
                FileChange(path="src/main.py", action="created", language="python"),
                FileChange(path="tests/test_main.py", action="created"),
            ],
            tests_passed=True,
            dependencies_added=["fastapi", "uvicorn"],
        )
        assert len(result.files_created) == 2
        assert result.files_created[0].language == "python"
        assert result.tests_passed is True
        assert "fastapi" in result.dependencies_added


class TestReviewResult:
    """Test ReviewResult schema."""

    def test_basic_result(self):
        """Test creating a basic review result."""
        result = ReviewResult(
            summary="Code looks good",
            approved=True,
            score=0.95,
        )
        assert result.approved is True
        assert result.score == 0.95

    def test_with_issues(self):
        """Test review result with issues."""
        result = ReviewResult(
            summary="Found some issues",
            issues=[
                CodeIssue(
                    severity=IssueSeverity.HIGH,
                    file="auth.py",
                    line=42,
                    description="SQL injection vulnerability",
                    suggestion="Use parameterized queries",
                    category="security",
                ),
                CodeIssue(
                    severity=IssueSeverity.LOW,
                    file="utils.py",
                    description="Missing docstring",
                ),
            ],
            approved=False,
            score=0.6,
            security_concerns=["SQL injection in auth.py"],
        )
        assert len(result.issues) == 2
        assert result.issues[0].severity == IssueSeverity.HIGH
        assert not result.approved


class TestPlanResult:
    """Test PlanResult schema."""

    def test_basic_plan(self):
        """Test creating a basic plan."""
        result = PlanResult(
            goal="Build a REST API",
            estimated_steps=5,
        )
        assert result.goal == "Build a REST API"
        assert result.tasks == []

    def test_with_tasks(self):
        """Test plan with tasks."""
        result = PlanResult(
            goal="Build a REST API",
            tasks=[
                PlannedTask(
                    id="1",
                    description="Set up project structure",
                    agent="coder",
                    priority=1,
                ),
                PlannedTask(
                    id="2",
                    description="Implement endpoints",
                    agent="coder",
                    dependencies=["1"],
                    estimated_complexity="moderate",
                ),
            ],
            estimated_steps=10,
            risks=["Database schema may need revision"],
        )
        assert len(result.tasks) == 2
        assert result.tasks[1].dependencies == ["1"]


class TestGenericResult:
    """Test GenericResult schema."""

    def test_basic_result(self):
        """Test creating a generic result."""
        result = GenericResult(
            output="Some output",
            data={"key": "value"},
        )
        assert result.output == "Some output"
        assert result.data["key"] == "value"


class TestContextEntry:
    """Test ContextEntry schema."""

    def test_user_entry(self):
        """Test creating a user entry."""
        entry = ContextEntry(
            id="entry1",
            type=ContextEntryType.USER,
            content="Hello!",
        )
        assert entry.type == ContextEntryType.USER
        assert entry.content == "Hello!"
        assert entry.compacted is False

    def test_file_entry(self):
        """Test creating a file content entry."""
        entry = ContextEntry(
            id="entry2",
            type=ContextEntryType.FILE_CONTENT,
            content="def hello(): pass",
            path="/src/main.py",
            start_line=1,
            end_line=5,
        )
        assert entry.path == "/src/main.py"
        assert entry.start_line == 1

    def test_render_file_ref(self):
        """Test rendering a file reference."""
        entry = ContextEntry(
            id="ref1",
            type=ContextEntryType.FILE_REF,
            content="",
            path="/src/main.py",
            start_line=1,
            end_line=10,
        )
        rendered = entry.render()
        assert "/src/main.py" in rendered
        assert "1-10" in rendered


class TestStepLog:
    """Test StepLog schema."""

    def test_basic_log(self):
        """Test creating a step log."""
        log = StepLog(
            step_id="step1",
            run_id="run1",
            agent_id="main",
            input_prompt="Do something",
            input_context_size=1000,
            timestamp_start=datetime.now(),
            model="claude-3",
        )
        assert log.step_id == "step1"
        assert log.agent_id == "main"
        assert log.output_tool_calls == []


class TestCheckpoint:
    """Test Checkpoint schema."""

    def test_basic_checkpoint(self):
        """Test creating a checkpoint."""
        checkpoint = Checkpoint(
            run_id="run1",
            step_number=5,
            context_entries=[],
            frozen_prefix_length=2,
            current_agent="main",
        )
        assert checkpoint.run_id == "run1"
        assert checkpoint.step_number == 5

