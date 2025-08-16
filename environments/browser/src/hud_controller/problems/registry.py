"""Problems registry for browser environment."""

import json
import logging
from typing import Dict, Any, Type

logger = logging.getLogger(__name__)

# Global problems registry - only classes
PROBLEM_REGISTRY: Dict[str, Type] = {}


def problem(name: str, app: str = "", description: str = "", **metadata):
    """Decorator to register class-based problems.

    Args:
        name: Problem name
        app: Application this problem is for
        description: Human-readable description
        **metadata: Additional metadata (difficulty, task_type, etc.)

    Usage:
        @problem("todo_basic", app="todo", description="Basic test", difficulty="easy")
        class TodoBasicProblem:
            def get_setup(self):
                return {"name": "todo_seed", "arguments": {"num_items": 5}}

            def get_evaluation(self):
                return {"name": "todo_completed", "arguments": {"expected_count": 2}}
    """

    def decorator(cls: Type) -> Type:
        # Store metadata
        cls._problem_name = name
        cls._problem_app = app
        cls._problem_description = description

        # Store additional metadata
        for key, value in metadata.items():
            setattr(cls, f"_problem_{key}", value)

        # Register
        PROBLEM_REGISTRY[name] = cls
        logger.debug(f"Registered problem: {name} -> {cls.__name__}")

        return cls

    return decorator


class ProblemRegistry:
    """Registry for class-based problems - stores and provides metadata only."""

    @classmethod
    def create_problem(cls, problem_name: str):
        """Get a problem instance by name."""
        if problem_name not in PROBLEM_REGISTRY:
            raise ValueError(f"Unknown problem: {problem_name}")

        problem_class = PROBLEM_REGISTRY[problem_name]
        return problem_class()

    @classmethod
    def get_all_problems(cls) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered problems."""
        result = {}
        for name, problem_class in PROBLEM_REGISTRY.items():
            # Extract all metadata
            metadata = {}
            for attr_name in dir(problem_class):
                if attr_name.startswith("_problem_"):
                    key = attr_name[9:]  # Remove '_problem_' prefix
                    metadata[key] = getattr(problem_class, attr_name)

            result[name] = {
                "name": name,
                "class": problem_class.__name__,
                "module": problem_class.__module__,
                **metadata,
            }
        return result

    @classmethod
    def get_problems_by_app(cls, app: str) -> Dict[str, Dict[str, Any]]:
        """Get problems for a specific app."""
        all_problems = cls.get_all_problems()
        return {name: info for name, info in all_problems.items() if info.get("app") == app}

    @classmethod
    def get_problem_schema(cls, problem_name: str) -> Dict[str, Any]:
        """Get detailed schema for a specific problem."""
        if problem_name not in PROBLEM_REGISTRY:
            return {"error": f"Problem '{problem_name}' not found"}

        problem_class = PROBLEM_REGISTRY[problem_name]
        schema = {
            "name": problem_name,
            "class": problem_class.__name__,
            "module": problem_class.__module__,
            "docstring": problem_class.__doc__,
            "has_setup": hasattr(problem_class, "get_setup"),
            "has_evaluation": hasattr(problem_class, "get_evaluation"),
        }

        # Add all metadata
        for attr_name in dir(problem_class):
            if attr_name.startswith("_problem_"):
                key = attr_name[9:]  # Remove '_problem_' prefix
                schema[key] = getattr(problem_class, attr_name)

        return schema

    @classmethod
    def to_json(cls) -> str:
        """Export all problems as JSON."""
        return json.dumps(
            {
                "problems": cls.get_all_problems(),
                "total_count": len(PROBLEM_REGISTRY),
                "registry_type": "problems",
            },
            indent=2,
        )
