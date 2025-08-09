"""Todo app evaluators."""

import logging
from typing import Dict, Any, List
from hud.tools import BaseEvaluator, EvaluationResult
from ..evaluators import evaluator

logger = logging.getLogger(__name__)


@evaluator("todo_completed", description="Check if expected number of todos are completed")
class TodoCompletedEvaluator(BaseEvaluator):
    """Evaluator that checks if the expected number of todos are completed."""

    async def __call__(self, context, expected_count: int) -> EvaluationResult:
        """Check if at least expected_count todos are completed."""
        try:
            # Use the app-centric approach: call todo backend API
            stats = await context.call_app_api("todo", "/api/eval/stats")
            completed_count = stats.get("completed_items", 0)

            success = completed_count >= expected_count
            reward = (
                1.0
                if success
                else (completed_count / expected_count if expected_count > 0 else 0.0)
            )

            return EvaluationResult(
                reward=reward,
                done=True,
                info={
                    "success": success,
                    "completed_count": completed_count,
                    "expected_count": expected_count,
                    "message": f"Found {completed_count} completed todos (expected {expected_count})",
                },
            )
        except Exception as e:
            logger.error(f"TodoCompletedEvaluator failed: {e}")
            return EvaluationResult(
                reward=0.0,
                done=True,
                info={
                    "success": False,
                    "error": str(e),
                    "message": "Failed to evaluate completed todos",
                },
            )


@evaluator("todo_exists", description="Check if a todo with specific title exists")
class TodoExistsEvaluator(BaseEvaluator):
    """Evaluator that checks if a todo with a specific title exists."""

    async def __call__(self, context, title: str) -> EvaluationResult:
        """Check if a todo with the given title exists."""
        try:
            # Call the app's API to get all todos
            todos = await context.call_app_api("todo", "/api/eval/todos")

            # Check if any todo has the expected title
            exists = any(todo.get("title") == title for todo in todos)

            return EvaluationResult(
                reward=1.0 if exists else 0.0,
                done=True,
                info={
                    "success": exists,
                    "title": title,
                    "total_todos": len(todos),
                    "message": f"Todo '{title}' {'exists' if exists else 'does not exist'}",
                },
            )
        except Exception as e:
            logger.error(f"TodoExistsEvaluator failed: {e}")
            return EvaluationResult(
                reward=0.0,
                done=True,
                info={
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to check if todo '{title}' exists",
                },
            )


@evaluator("todo_completion_rate", description="Check if completion rate meets target")
class TodoCompletionRateEvaluator(BaseEvaluator):
    """Evaluator that checks if the todo completion rate meets a target."""

    async def __call__(self, context, target_rate: float = 0.5) -> EvaluationResult:
        """Check if the completion rate meets the target."""
        try:
            # Get stats from the app
            stats = await context.call_app_api("todo", "/api/eval/stats")
            total_count = stats.get("total_items", 0)
            completed_count = stats.get("completed_items", 0)

            if total_count == 0:
                # No todos, consider this as 0% completion
                actual_rate = 0.0
            else:
                actual_rate = completed_count / total_count

            success = actual_rate >= target_rate
            reward = min(1.0, actual_rate / target_rate) if target_rate > 0 else 1.0

            return EvaluationResult(
                reward=reward,
                done=True,
                info={
                    "success": success,
                    "actual_rate": actual_rate,
                    "target_rate": target_rate,
                    "completed_count": completed_count,
                    "total_count": total_count,
                    "message": f"Completion rate: {actual_rate:.1%} (target: {target_rate:.1%})",
                },
            )
        except Exception as e:
            logger.error(f"TodoCompletionRateEvaluator failed: {e}")
            return EvaluationResult(
                reward=0.0,
                done=True,
                info={
                    "success": False,
                    "error": str(e),
                    "message": "Failed to evaluate completion rate",
                },
            )


@evaluator("todo_total_count", description="Check if total todo count meets minimum")
class TodoTotalCountEvaluator(BaseEvaluator):
    """Evaluator that checks if the total number of todos meets a minimum."""

    async def __call__(self, context, min_count: int = 1) -> EvaluationResult:
        """Check if there are at least min_count todos."""
        try:
            # Get stats from the app
            stats = await context.call_app_api("todo", "/api/eval/stats")
            total_count = stats.get("total_items", 0)

            success = total_count >= min_count
            reward = 1.0 if success else (total_count / min_count if min_count > 0 else 0.0)

            return EvaluationResult(
                reward=reward,
                done=True,
                info={
                    "success": success,
                    "total_count": total_count,
                    "min_count": min_count,
                    "message": f"Found {total_count} todos (minimum: {min_count})",
                },
            )
        except Exception as e:
            logger.error(f"TodoTotalCountEvaluator failed: {e}")
            return EvaluationResult(
                reward=0.0,
                done=True,
                info={
                    "success": False,
                    "error": str(e),
                    "message": "Failed to evaluate total count",
                },
            )


@evaluator("todo_all_completed", description="Check if all todos are completed")
class TodoAllCompletedEvaluator(BaseEvaluator):
    """Evaluator that checks if all todos are completed."""

    async def __call__(self, context) -> EvaluationResult:
        """Check if all todos are completed."""
        try:
            # Get stats from the app
            stats = await context.call_app_api("todo", "/api/eval/stats")
            total_count = stats.get("total_items", 0)
            completed_count = stats.get("completed_items", 0)

            if total_count == 0:
                # No todos, consider this as success
                success = True
                message = "No todos exist"
            else:
                success = completed_count == total_count
                message = f"{completed_count}/{total_count} todos completed"

            return EvaluationResult(
                reward=1.0
                if success
                else (completed_count / total_count if total_count > 0 else 0.0),
                done=True,
                info={
                    "success": success,
                    "completed_count": completed_count,
                    "total_count": total_count,
                    "message": message,
                },
            )
        except Exception as e:
            logger.error(f"TodoAllCompletedEvaluator failed: {e}")
            return EvaluationResult(
                reward=0.0,
                done=True,
                info={
                    "success": False,
                    "error": str(e),
                    "message": "Failed to evaluate if all todos are completed",
                },
            )
