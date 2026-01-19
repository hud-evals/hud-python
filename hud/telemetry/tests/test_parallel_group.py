"""Tests for hud.telemetry.parallel_group module."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hud.telemetry.parallel_group import (
    ParallelAgentGroup,
    ParallelAgentInfo,
    parallel_agent_group,
)


class TestParallelAgentInfo:
    """Tests for ParallelAgentInfo dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        agent = ParallelAgentInfo(id="test-id", name="Test Agent")
        assert agent.id == "test-id"
        assert agent.name == "Test Agent"
        assert agent.status == "pending"
        assert agent.trace_id is None

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        agent = ParallelAgentInfo(
            id="agent-1",
            name="Worker 1",
            status="completed",
            trace_id="trace-123",
        )
        result = agent.to_dict()
        assert result == {
            "id": "agent-1",
            "name": "Worker 1",
            "status": "completed",
            "trace_id": "trace-123",
        }

    def test_to_status_dict(self) -> None:
        """Test to_status_dict minimal serialization."""
        agent = ParallelAgentInfo(
            id="agent-1",
            name="Worker 1",
            status="running",
            trace_id="trace-123",
        )
        result = agent.to_status_dict()
        assert result == {
            "id": "agent-1",
            "status": "running",
        }


class TestParallelAgentGroup:
    """Tests for ParallelAgentGroup class."""

    def test_creation(self) -> None:
        """Test group creation with agents."""
        agents = [
            ParallelAgentInfo(id="a1", name="Agent 1"),
            ParallelAgentInfo(id="a2", name="Agent 2"),
        ]
        group = ParallelAgentGroup(
            title="Test Group",
            description="Test description",
            agents=agents,
        )
        assert group.title == "Test Group"
        assert group.description == "Test description"
        assert len(group.agents) == 2
        assert group.total_count == 2
        assert group.completed_count == 0

    def test_update_status(self) -> None:
        """Test updating agent status."""
        agents = [
            ParallelAgentInfo(id="a1", name="Agent 1"),
            ParallelAgentInfo(id="a2", name="Agent 2"),
        ]
        group = ParallelAgentGroup(
            title="Test",
            description="Test",
            agents=agents,
        )

        with patch.object(group, "_emit_update"):
            group.update_status("a1", "running")
            assert group.agents[0].status == "running"

            group.update_status("a1", "completed", trace_id="trace-123")
            assert group.agents[0].status == "completed"
            assert group.agents[0].trace_id == "trace-123"

    def test_update_status_invalid_id(self) -> None:
        """Test updating non-existent agent raises error."""
        group = ParallelAgentGroup(
            title="Test",
            description="Test",
            agents=[ParallelAgentInfo(id="a1", name="Agent 1")],
        )

        with pytest.raises(ValueError, match="Agent with id 'invalid' not found"):
            group.update_status("invalid", "running")

    def test_mark_helpers(self) -> None:
        """Test mark_running, mark_completed, mark_failed helpers."""
        agents = [
            ParallelAgentInfo(id="a1", name="Agent 1"),
            ParallelAgentInfo(id="a2", name="Agent 2"),
            ParallelAgentInfo(id="a3", name="Agent 3"),
        ]
        group = ParallelAgentGroup(
            title="Test",
            description="Test",
            agents=agents,
        )

        with patch.object(group, "_emit_update"):
            group.mark_running("a1")
            assert group.agents[0].status == "running"

            group.mark_completed("a2", trace_id="trace-2")
            assert group.agents[1].status == "completed"
            assert group.agents[1].trace_id == "trace-2"

            group.mark_failed("a3")
            assert group.agents[2].status == "failed"

    def test_count_properties(self) -> None:
        """Test count properties calculate correctly."""
        agents = [
            ParallelAgentInfo(id="a1", name="Agent 1", status="completed"),
            ParallelAgentInfo(id="a2", name="Agent 2", status="failed"),
            ParallelAgentInfo(id="a3", name="Agent 3", status="running"),
            ParallelAgentInfo(id="a4", name="Agent 4", status="pending"),
        ]
        group = ParallelAgentGroup(
            title="Test",
            description="Test",
            agents=agents,
        )

        assert group.total_count == 4
        assert group.completed_count == 2  # completed + failed
        assert group.success_count == 1
        assert group.failure_count == 1

    def test_build_span_without_trace_id(self) -> None:
        """Test _build_span returns empty dict when no trace_id."""
        group = ParallelAgentGroup(
            title="Test",
            description="Test",
            agents=[],
        )

        with patch(
            "hud.telemetry.parallel_group._get_trace_id", return_value=None
        ):
            span = group._build_span()
            assert span == {}

    def test_build_span_with_trace_id(self) -> None:
        """Test _build_span builds correct span structure."""
        agents = [
            ParallelAgentInfo(id="a1", name="Agent 1", status="completed"),
            ParallelAgentInfo(id="a2", name="Agent 2", status="running"),
        ]
        group = ParallelAgentGroup(
            title="Test Group",
            description="Test description",
            agents=agents,
            _task_run_id="test-trace-id-123456789012",
        )

        span = group._build_span(final=False)

        assert span["name"] == "parallel_agent_group"
        # Trace ID is normalized to 32 hex chars (dashes removed, padded/truncated)
        assert len(span["trace_id"]) == 32
        assert span["trace_id"].startswith("testtraceid")
        assert span["status_code"] == "OK"
        assert span["internal_type"] == "parallel-agent-group"

        attrs = span["attributes"]
        assert attrs["category"] == "parallel-agent-group"
        assert attrs["request"]["title"] == "Test Group"
        assert attrs["request"]["description"] == "Test description"
        assert len(attrs["request"]["agents"]) == 2
        assert attrs["result"]["completed"] == 1
        assert attrs["result"]["total"] == 2

    def test_build_span_final_with_failures(self) -> None:
        """Test _build_span sets ERROR status when there are failures."""
        agents = [
            ParallelAgentInfo(id="a1", name="Agent 1", status="failed"),
        ]
        group = ParallelAgentGroup(
            title="Test",
            description="Test",
            agents=agents,
            _task_run_id="test-trace-id-123456789012",
        )

        span = group._build_span(final=True)
        assert span["status_code"] == "ERROR"


class TestParallelAgentGroupContextManager:
    """Tests for parallel_agent_group context manager."""

    @pytest.mark.asyncio
    async def test_basic_usage(self) -> None:
        """Test basic context manager usage."""
        queued_spans: list[dict[str, Any]] = []

        with patch(
            "hud.telemetry.parallel_group.queue_span",
            side_effect=lambda s: queued_spans.append(s),
        ), patch(
            "hud.telemetry.parallel_group._get_trace_id",
            return_value="test-trace-12345678901234567890",
        ):
            async with parallel_agent_group(
                title="Test Group",
                description="Test description",
                agents=[{"name": "Worker 1"}, {"name": "Worker 2"}],
            ) as group:
                assert len(group.agents) == 2
                assert group.agents[0].name == "Worker 1"
                assert group.agents[1].name == "Worker 2"
                assert all(a.status == "pending" for a in group.agents)

        # Should have emitted at least 2 spans (initial + final)
        assert len(queued_spans) >= 2

    @pytest.mark.asyncio
    async def test_status_updates_emit_spans(self) -> None:
        """Test that status updates emit spans."""
        queued_spans: list[dict[str, Any]] = []

        with patch(
            "hud.telemetry.parallel_group.queue_span",
            side_effect=lambda s: queued_spans.append(s),
        ), patch(
            "hud.telemetry.parallel_group._get_trace_id",
            return_value="test-trace-12345678901234567890",
        ):
            async with parallel_agent_group(
                title="Test",
                description="Test",
                agents=[{"name": "Worker 1"}],
            ) as group:
                initial_count = len(queued_spans)
                group.mark_running(group.agents[0].id)
                assert len(queued_spans) == initial_count + 1

                group.mark_completed(group.agents[0].id)
                assert len(queued_spans) == initial_count + 2

        # Final span emitted on exit
        assert len(queued_spans) >= 3

    @pytest.mark.asyncio
    async def test_agent_name_defaults(self) -> None:
        """Test that agent names default correctly."""
        with patch(
            "hud.telemetry.parallel_group.queue_span"
        ), patch(
            "hud.telemetry.parallel_group._get_trace_id",
            return_value="test-trace-12345678901234567890",
        ):
            async with parallel_agent_group(
                title="Test",
                description="Test",
                agents=[{}, {"name": "Custom Name"}],
            ) as group:
                assert group.agents[0].name == "Agent 0"
                assert group.agents[1].name == "Custom Name"

    @pytest.mark.asyncio
    async def test_parallel_execution_pattern(self) -> None:
        """Test typical parallel execution pattern."""
        results: list[str] = []

        with patch(
            "hud.telemetry.parallel_group.queue_span"
        ), patch(
            "hud.telemetry.parallel_group._get_trace_id",
            return_value="test-trace-12345678901234567890",
        ):
            async with parallel_agent_group(
                title="Parallel Work",
                description="Do work in parallel",
                agents=[{"name": f"Worker {i}"} for i in range(3)],
            ) as group:

                async def do_work(agent_info: ParallelAgentInfo) -> str:
                    group.mark_running(agent_info.id)
                    await asyncio.sleep(0.01)  # Simulate work
                    group.mark_completed(agent_info.id)
                    return f"Result from {agent_info.name}"

                results = await asyncio.gather(*[do_work(a) for a in group.agents])

        assert len(results) == 3
        assert all("Result from Worker" in r for r in results)

    @pytest.mark.asyncio
    async def test_exception_handling(self) -> None:
        """Test that exceptions propagate and final span is still emitted."""
        queued_spans: list[dict[str, Any]] = []

        with patch(
            "hud.telemetry.parallel_group.queue_span",
            side_effect=lambda s: queued_spans.append(s),
        ), patch(
            "hud.telemetry.parallel_group._get_trace_id",
            return_value="test-trace-12345678901234567890",
        ):
            with pytest.raises(ValueError, match="Test error"):
                async with parallel_agent_group(
                    title="Test",
                    description="Test",
                    agents=[{"name": "Worker"}],
                ) as group:
                    raise ValueError("Test error")

        # Final span should still be emitted
        assert len(queued_spans) >= 2


class TestModuleExports:
    """Test that module exports are correct."""

    def test_imports(self) -> None:
        """Test that all expected symbols are importable."""
        from hud.telemetry import (
            ParallelAgentGroup,
            ParallelAgentInfo,
            parallel_agent_group,
        )

        assert ParallelAgentGroup is not None
        assert ParallelAgentInfo is not None
        assert parallel_agent_group is not None

    def test_hud_import(self) -> None:
        """Test that parallel_agent_group is importable from hud."""
        from hud import parallel_agent_group

        assert parallel_agent_group is not None
