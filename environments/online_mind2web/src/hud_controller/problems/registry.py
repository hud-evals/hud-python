"""Registry system for problem definitions."""

from typing import Dict, Type, Any, List
import json
import logging

logger = logging.getLogger(__name__)

# Global registry for problem classes
PROBLEM_REGISTRY: Dict[str, Type] = {}


def problem(name: str, description: str | None = None):
    """Decorator to register a problem class.

    Args:
        name: The problem identifier
        description: Optional description for the problem

    Example:
        @problem("navigate_and_click", description="Navigate to URL and click element")
        class NavigateAndClickProblem:
            def get_setup(self):
                return {"name": "navigate_to_url", "arguments": {"url": "https://example.com"}}
            def get_evaluation(self):
                return {"name": "element_clicked", "arguments": {"selector": "#submit"}}
    """

    def decorator(cls):
        # Store metadata on the class
        cls._problem_name = name
        cls._problem_description = description

        PROBLEM_REGISTRY[name] = cls
        logger.info(f"Registered problem: {name} -> {cls.__name__}")
        return cls

    return decorator


class ProblemRegistry:
    """Registry for problem definitions."""

    @staticmethod
    def create_problem(name: str):
        """Create a problem instance by name.

        Args:
            name: Problem identifier

        Returns:
            Problem instance
        """
        if name not in PROBLEM_REGISTRY:
            available = list(PROBLEM_REGISTRY.keys())
            raise ValueError(f"Unknown problem: {name}. Available: {available}")

        problem_class = PROBLEM_REGISTRY[name]
        return problem_class()

    @staticmethod
    def to_json() -> str:
        """Convert registry to JSON for MCP resource serving."""
        problems = []
        for name, cls in PROBLEM_REGISTRY.items():
            problems.append(
                {
                    "name": name,
                    "class": cls.__name__,
                    "description": getattr(cls, "_problem_description", None),
                }
            )
        return json.dumps(problems, indent=2)

    @staticmethod
    def list_problems() -> List[str]:
        """Get list of available problem names."""
        return list(PROBLEM_REGISTRY.keys())

    @staticmethod
    def get_problem_info(name: str) -> dict:
        """Get information about a specific problem."""
        if name not in PROBLEM_REGISTRY:
            raise ValueError(f"Unknown problem: {name}")

        cls = PROBLEM_REGISTRY[name]
        return {
            "name": name,
            "class": cls.__name__,
            "description": getattr(cls, "_problem_description", None),
        }
