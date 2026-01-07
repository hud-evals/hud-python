"""Tests for schemas.py including make_schema helper."""

import pytest
from pydantic import BaseModel

from hud.multi_agent.config import SCHEMA_REGISTRY
from hud.multi_agent.schemas import AgentResultBase, make_schema


class TestMakeSchema:
    """Test the make_schema() helper function.
    
    Note: Many tests use `# type: ignore[attr-defined]` because make_schema()
    creates dynamic models whose attributes can't be statically inferred.
    """

    def test_basic_schema_creation(self):
        """Test creating a simple schema with make_schema."""
        # Create a schema
        TestResult = make_schema(
            "TestResult",
            summary=str,
            score=float,
        )

        # Verify it's a Pydantic model
        assert issubclass(TestResult, BaseModel)
        assert issubclass(TestResult, AgentResultBase)

        # Verify it was registered
        assert "TestResult" in SCHEMA_REGISTRY
        assert SCHEMA_REGISTRY["TestResult"] is TestResult

    def test_list_and_dict_defaults(self):
        """Test that list and dict fields get proper defaults."""
        ListDictResult = make_schema(
            "ListDictResult",
            items=list[str],
            data=dict[str, int],
        )

        # Create instance without providing values
        instance = ListDictResult()

        # Should have empty defaults, not None
        assert instance.items == []  # type: ignore[attr-defined]
        assert instance.data == {}  # type: ignore[attr-defined]
        assert instance.success is True  # type: ignore[attr-defined]

    def test_explicit_defaults(self):
        """Test providing explicit default values."""
        DefaultsResult = make_schema(
            "DefaultsResult",
            title=(str, "default_title"),
            count=(int, 42),
            enabled=(bool, True),
        )

        instance = DefaultsResult()

        assert instance.title == "default_title"  # type: ignore[attr-defined]
        assert instance.count == 42  # type: ignore[attr-defined]
        assert instance.enabled is True  # type: ignore[attr-defined]

    def test_optional_fields(self):
        """Test that Optional types default to None."""
        OptionalResult = make_schema(
            "OptionalResult",
            maybe_str=str | None,
            maybe_int=int | None,
        )

        instance = OptionalResult()

        assert instance.maybe_str is None  # type: ignore[attr-defined]
        assert instance.maybe_int is None  # type: ignore[attr-defined]

    def test_custom_base_class(self):
        """Test using a custom base class instead of AgentResultBase."""

        class MyBase(BaseModel):
            custom_field: str = "custom"

        CustomBaseResult = make_schema(
            "CustomBaseResult",
            __base__=MyBase,
            data=str,
        )

        instance = CustomBaseResult()

        # Should have custom base field
        assert instance.custom_field == "custom"  # type: ignore[attr-defined]
        # Should NOT have AgentResultBase fields
        assert not hasattr(instance, "success")

    def test_inherits_agent_result_base_fields(self):
        """Test that schemas inherit AgentResultBase fields by default."""
        InheritResult = make_schema(
            "InheritResult",
            my_field=str,
        )

        instance = InheritResult()

        # AgentResultBase fields should be present
        assert hasattr(instance, "success")
        assert hasattr(instance, "error")
        assert hasattr(instance, "duration_ms")
        assert hasattr(instance, "timestamp")

    def test_schema_validation(self):
        """Test that created schemas validate data properly."""
        ValidatedResult = make_schema(
            "ValidatedResult",
            count=(int, 0),
            items=list[str],
        )

        # Valid data
        instance = ValidatedResult(count=5, items=["a", "b"])
        assert instance.count == 5  # type: ignore[attr-defined]
        assert instance.items == ["a", "b"]  # type: ignore[attr-defined]

        # Invalid data should raise
        with pytest.raises(Exception):  # Pydantic ValidationError
            ValidatedResult(count="not_an_int")

    def test_schema_serialization(self):
        """Test that created schemas serialize properly."""
        SerialResult = make_schema(
            "SerialResult",
            message=str,
            score=(float, 0.5),
        )

        instance = SerialResult(message="hello", score=0.9)
        data = instance.model_dump()

        assert data["message"] == "hello"
        assert data["score"] == 0.9
        assert "success" in data  # From AgentResultBase

    def test_real_world_example(self):
        """Test a realistic custom schema for data analysis."""
        DataAnalysisResult = make_schema(
            "DataAnalysisResult",
            insights=list[str],
            chart_path=(str | None, None),
            metrics=dict[str, float],
            confidence=(float, 0.8),
        )

        # Simulate what an LLM might return
        result = DataAnalysisResult(
            insights=["Revenue up 20%", "Q4 strongest quarter"],
            chart_path="/workspace/chart.png",
            metrics={"revenue": 1000000, "growth": 0.2},
            confidence=0.95,
            success=True,
        )

        assert len(result.insights) == 2  # type: ignore[attr-defined]
        assert result.chart_path == "/workspace/chart.png"  # type: ignore[attr-defined]
        assert result.metrics["growth"] == 0.2  # type: ignore[attr-defined]
        assert result.confidence == 0.95  # type: ignore[attr-defined]
