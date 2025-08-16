"""Todo app problems."""

from typing import Dict, Any
import logging
from .registry import problem

logger = logging.getLogger(__name__)


# === BASE PROBLEM CLASSES (for inheritance) ===


class BaseSetupResetProblem:
    """Base class for problems that need a clean start."""

    def get_setup(self) -> Dict[str, Any]:
        """Default setup that resets the database."""
        return {"name": "todo_reset", "arguments": {}}


class BaseSeedProblem:
    """Base class for problems that need seeded data."""

    def get_setup(self) -> Dict[str, Any]:
        """Default setup that seeds test data."""
        return {"name": "todo_seed", "arguments": {"num_items": 5}}


class BaseCompletionProblem:
    """Base class for problems that test completion functionality."""

    def get_evaluation(self) -> Dict[str, Any]:
        """Default evaluation for completion count."""
        return {"name": "todo_completed", "arguments": {"expected_count": 2}}


# === CONCRETE PROBLEM CLASSES ===


@problem(
    "todo_basic_usage",
    app="todo",
    description="Basic todo app usage test",
    difficulty="easy",
    task_type="functional",
)
class TodoBasicUsageProblem(BaseSeedProblem, BaseCompletionProblem):
    """Test basic todo app functionality with seeded data.

    Inherits setup from BaseSeedProblem and evaluation from BaseCompletionProblem.
    """

    pass


@problem(
    "todo_empty_start",
    app="todo",
    description="Test starting with empty todo list",
    difficulty="easy",
    task_type="edge_case",
)
class TodoEmptyStartProblem(BaseSetupResetProblem):
    """Test behavior with empty todo list."""

    def get_evaluation(self) -> Dict[str, Any]:
        """Evaluate that completion rate handles empty case gracefully."""
        return {"name": "todo_completion_rate", "arguments": {"min_rate": 0.0}}


@problem(
    "todo_composite_weighted",
    app="todo",
    description="Composite evaluation with weighted criteria",
    difficulty="medium",
    task_type="functional",
)
class TodoCompositeWeightedProblem:
    """Test composite evaluation with multiple weighted criteria."""

    def get_setup(self) -> Dict[str, Any]:
        """Setup with mixed completion state."""
        return {
            "function": "todo_custom_seed",
            "args": {
                "items": [
                    {"title": "Important task", "description": "High priority", "completed": True},
                    {
                        "title": "Buy groceries",
                        "description": "Weekly shopping",
                        "completed": False,
                    },
                    {"title": "Meeting prep", "description": "Prepare slides", "completed": True},
                ]
            },
        }

    def get_evaluation(self) -> Dict[str, Any]:
        """Composite evaluation with weights."""
        return {
            "name": "composite_evaluate",
            "arguments": {
                "evaluators": [
                    {"name": "todo_completed", "arguments": {"expected_count": 2}, "weight": 0.6},
                    {"name": "todo_exists", "arguments": {"text": "important"}, "weight": 0.4},
                ]
            },
        }


@problem(
    "todo_high_completion_rate",
    app="todo",
    description="Test high completion rate requirement",
    difficulty="medium",
    task_type="performance",
)
class TodoHighCompletionRateProblem(BaseSeedProblem):
    """Test that requires a high completion rate."""

    def get_evaluation(self) -> Dict[str, Any]:
        """Evaluate that completion rate is at least 60%."""
        return {"name": "todo_completion_rate", "arguments": {"min_rate": 0.6}}


@problem(
    "todo_setup_only",
    app="todo",
    description="Setup-only problem for testing",
    difficulty="easy",
    task_type="setup_test",
)
class TodoSetupOnlyProblem(BaseSeedProblem):
    """Problem that only does setup, no evaluation - useful for preparation."""

    # No get_evaluation method - this problem only does setup


@problem(
    "todo_direct_db_test",
    app="todo",
    description="Direct database access test",
    difficulty="hard",
    task_type="technical",
)
class TodoDirectDbTestProblem(BaseSetupResetProblem):
    """Test direct database access functionality."""

    def get_setup(self) -> Dict[str, Any]:
        """Setup with a specific item for testing."""
        return {
            "function": "todo_custom_seed",
            "args": {
                "items": [
                    {"title": "Test item", "description": "For direct access", "completed": True}
                ]
            },
        }

    def get_evaluation(self) -> Dict[str, Any]:
        """Evaluate specific item state via direct access."""
        return {"name": "todo_db_direct", "arguments": {"todo_id": "1", "expected_completed": True}}
