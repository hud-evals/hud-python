"""
Inspect AI Task Loader

Loads inspect_ai Task definitions and analyzes their requirements.
Works with any inspect_ai eval (mbpp, swe_bench, etc.).
"""

from __future__ import annotations

import ast
import inspect as py_inspect
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from inspect_ai import Task


class TaskRequirements:
    """Describes what capabilities/tools an inspect Task needs."""

    def __init__(self):
        self.needs_exec = False
        self.needs_file_ops = False
        self.needs_git = False
        self.needs_browser = False
        self.needs_auto_evaluate = False
        self.sandbox_type: str | None = None
        self.custom_tools: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_exec": self.needs_exec,
            "needs_file_ops": self.needs_file_ops,
            "needs_git": self.needs_git,
            "needs_browser": self.needs_browser,
            "needs_auto_evaluate": self.needs_auto_evaluate,
            "sandbox_type": self.sandbox_type,
            "custom_tools": self.custom_tools,
        }

    def get_required_tools(self) -> list[str]:
        """Get list of MCP tool names that should be available."""
        tools = []

        if self.needs_exec:
            tools.append("exec")
            # Code evals always need file operations to write solutions
            if not self.needs_file_ops:
                self.needs_file_ops = True

        if self.needs_file_ops:
            tools.extend(["read_file", "write_file", "list_files"])

        if self.needs_git:
            tools.extend(["git_clone", "git_diff", "git_commit"])

        if self.needs_browser:
            tools.extend(["browser_navigate", "browser_click", "browser_type"])

        if self.needs_auto_evaluate:
            tools.append("auto_evaluate")

        tools.extend(self.custom_tools)

        return tools


def load_task_function(task_spec: str) -> Callable[..., Task]:
    """
    Load a task function from a module path.

    Args:
        task_spec: Can be:
            - Simple name: "mbpp" → loads from inspect_evals.mbpp
            - Module path: "inspect_evals.mbpp" → loads mbpp() function
            - With function: "inspect_evals.mbpp:mbpp" → explicit function
            - Custom: "custom_evals.my_eval:my_task"

    Returns:
        The task function (callable that returns Task)
    """
    # Parse task_spec
    if ":" in task_spec:
        module_path, function_name = task_spec.split(":", 1)
    else:
        module_path = task_spec
        function_name = None

    # Determine full module path
    if "." in module_path:
        # Custom eval with dots: "custom_evals.my_eval" or "inspect_evals.mbpp"
        full_module_path = module_path
        if not function_name:
            function_name = module_path.split(".")[-1]
    else:
        # Simple name: "mbpp" → "inspect_evals.mbpp"
        full_module_path = f"inspect_evals.{module_path}"
        if not function_name:
            function_name = module_path

    # Import and get task function
    try:
        eval_module = import_module(full_module_path)

        # Try to get the specified function
        if hasattr(eval_module, function_name):
            task_fn = getattr(eval_module, function_name)
            if callable(task_fn):
                return task_fn

        # If function not found or not callable, check __all__ for available functions
        if hasattr(eval_module, '__all__'):
            available_funcs = eval_module.__all__
            if available_funcs:
                # Use the first available function
                first_func = available_funcs[0]
                task_fn = getattr(eval_module, first_func)
                if callable(task_fn):
                    print(f"   ℹ️  Using '{first_func}' from available functions: {available_funcs}")
                    return task_fn

        # If still not found, raise a helpful error
        available = []
        if hasattr(eval_module, '__all__'):
            available = eval_module.__all__
        else:
            # List all callables that might be task functions
            import inspect as py_inspect_module
            available = [
                name for name, obj in py_inspect_module.getmembers(eval_module)
                if callable(obj) and not name.startswith('_')
            ][:10]  # Limit to first 10

        raise ValueError(
            f"Eval '{task_spec}' does not have function '{function_name}'. "
            f"Available functions: {available}. "
            f"Use format 'eval_name:function_name' to specify."
        )

    except ImportError as e:
        raise ValueError(
            f"Could not import eval '{task_spec}'. "
            f"For custom evals, ensure the module is accessible. Error: {e}"
        )


def analyze_task_requirements(task: Task, task_fn: Callable) -> TaskRequirements:
    """
    Analyze a Task to determine what sandbox capabilities it needs.

    This inspects:
    - The scorer function to see what sandbox operations it uses
    - The sandbox type specified in the task
    - The solver to see what tools it might need
    - Known eval patterns for standard evals

    Args:
        task: The Task object to analyze
        task_fn: The original task function (for source analysis)

    Returns:
        TaskRequirements describing what the task needs
    """
    reqs = TaskRequirements()

    # Check for well-known evals with known requirements
    task_name = getattr(task, 'name', '').lower()
    if task_name:
        # SWE-bench family: needs exec, file ops, and git
        if 'swe_bench' in task_name or 'swebench' in task_name:
            reqs.needs_exec = True
            reqs.needs_file_ops = True
            reqs.needs_git = True
            reqs.sandbox_type = "docker"
        # Code eval families: need exec and file ops
        elif any(name in task_name for name in ['mbpp', 'humaneval', 'apps', 'code']):
            reqs.needs_exec = True
            reqs.needs_file_ops = True
        # Math evals: need exec and file ops for verification
        elif any(name in task_name for name in ['math', 'gsm', 'theorem']):
            reqs.needs_exec = True
            reqs.needs_file_ops = True

    # Check sandbox type
    if task.sandbox:
        if isinstance(task.sandbox, str):
            reqs.sandbox_type = task.sandbox
        else:
            reqs.sandbox_type = "docker"  # Default

    # Analyze scorer if present
    if task.scorer:
        scorer_source = _get_scorer_source(task.scorer)
        if scorer_source:
            # Check for sandbox operations in scorer code
            if "sandbox().exec" in scorer_source or "sandbox.exec" in scorer_source:
                reqs.needs_exec = True

            if any(
                op in scorer_source
                for op in ["read_file", "write_file", "fs.read", "fs.write"]
            ):
                reqs.needs_file_ops = True

            if "git" in scorer_source.lower():
                reqs.needs_git = True

            if "browser" in scorer_source.lower() or "selenium" in scorer_source.lower():
                reqs.needs_browser = True

            # Check for LLM-as-judge patterns
            if any(
                pattern in scorer_source
                for pattern in [
                    "openai",
                    "anthropic",
                    "get_model(",
                    "model.generate",
                    "chat.completions.create",
                    "messages.create",
                ]
            ):
                reqs.needs_auto_evaluate = True

    # Analyze task function source for additional hints
    try:
        task_fn_source = py_inspect.getsource(task_fn)

        # Additional heuristics from task definition
        if "sandbox=" in task_fn_source:
            # Task explicitly uses sandbox
            if not reqs.needs_exec:
                reqs.needs_exec = True  # Assume exec is needed if sandbox specified

    except (TypeError, OSError):
        # Can't get source, skip analysis
        pass

    return reqs


def _get_scorer_source(scorer) -> str | None:
    """Try to extract source code from a scorer object."""
    try:
        # Scorer might be a function or a Scorer object
        if hasattr(scorer, "__wrapped__"):
            return py_inspect.getsource(scorer.__wrapped__)
        elif callable(scorer):
            return py_inspect.getsource(scorer)
        else:
            return None
    except (TypeError, OSError):
        return None


def load_inspect_task(
    task_spec: str, task_params: dict[str, Any] | None = None
) -> tuple[Task, TaskRequirements]:
    """
    Load an inspect_ai Task and analyze its requirements.

    Args:
        task_spec: Task specification (e.g., "mbpp", "inspect_evals.mbpp:mbpp")
        task_params: Optional parameters to pass to the task function

    Returns:
        Tuple of (Task object, TaskRequirements)

    Example:
        task, reqs = load_inspect_task("mbpp", {"temperature": 0.5})
        print(f"Task has {len(task.dataset)} samples")
        print(f"Required tools: {reqs.get_required_tools()}")
    """
    task_fn = load_task_function(task_spec)

    # Call task function with params
    if task_params:
        task = task_fn(**task_params)
    else:
        task = task_fn()

    # Analyze requirements
    reqs = analyze_task_requirements(task, task_fn)

    return task, reqs


def load_scorer_only(task_spec: str, task_params: dict[str, Any] | None = None):
    """
    Load only the scorer from a task, without loading the dataset.

    This is used in the container to avoid downloading the entire dataset
    when we only need to score a single sample.

    Args:
        task_spec: Task specification (e.g., "mbpp")
        task_params: Optional parameters

    Returns:
        The scorer object from the task
    """
    import inspect_ai.dataset

    # Monkeypatch dataset loading functions to return empty datasets
    # This prevents downloading datasets when we only need the scorer
    original_hf_dataset = inspect_ai.dataset.hf_dataset
    original_json_dataset = inspect_ai.dataset.json_dataset

    def mock_hf_dataset(*args, **kwargs):
        """Return empty dataset instead of loading from HuggingFace."""
        return []

    def mock_json_dataset(*args, **kwargs):
        """Return empty dataset instead of loading from file."""
        return []

    try:
        # Replace dataset loaders with mocks
        inspect_ai.dataset.hf_dataset = mock_hf_dataset
        inspect_ai.dataset.json_dataset = mock_json_dataset

        # Import the task function
        task_fn = load_task_function(task_spec)

        # Call it to get the task (dataset will be empty)
        if task_params:
            task = task_fn(**task_params)
        else:
            task = task_fn()

        return task.scorer

    finally:
        # Restore original functions
        inspect_ai.dataset.hf_dataset = original_hf_dataset
        inspect_ai.dataset.json_dataset = original_json_dataset
