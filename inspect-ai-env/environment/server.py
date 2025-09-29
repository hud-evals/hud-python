"""Minimal FastAPI environment server (HTTP-based)."""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import json
import logging
import sys
import uuid
from importlib import import_module

from inspect_ai import Task
from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser, ModelOutput

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Inspect AI Sample Processing Environment")

_count = 0
_sample_results = {}  # Store results by sample_id
_processing_status = {}  # Track processing status
_task_cache = {}  # Cache loaded eval tasks by eval_name


def load_eval_task(eval_spec: Dict[str, Any]) -> Task:
    """
    Dynamically load and instantiate an inspect_evals Task.

    Args:
        eval_spec: Dict containing:
            - eval_name: Name/path of the eval. Can be:
                * Simple name: "mbpp" → imports from inspect_evals.mbpp
                * Module path: "custom_evals.my_eval" → imports from that module path
                * Full path with function: "custom_evals.my_eval:my_task_fn"
            - task_params: Optional parameters to pass to the task function

    Returns:
        Task: The instantiated inspect_ai Task object

    Examples:
        # Official inspect_evals
        {"eval_name": "mbpp"}  → import inspect_evals.mbpp; mbpp()

        # Custom eval (auto-detect function name)
        {"eval_name": "custom_evals.my_eval"}  → import custom_evals.my_eval; my_eval()

        # Custom eval with explicit function
        {"eval_name": "custom_evals.my_eval:custom_task"}  → import custom_evals.my_eval; custom_task()
    """
    eval_name = eval_spec.get("eval_name")
    if not eval_name:
        raise ValueError("eval_spec must contain 'eval_name'")

    # Check cache first
    cache_key = f"{eval_name}:{json.dumps(eval_spec.get('task_params', {}), sort_keys=True)}"
    if cache_key in _task_cache:
        logger.info(f"Using cached task for {eval_name}")
        return _task_cache[cache_key]

    try:
        # Parse eval_name to extract module path and optional function name
        if ":" in eval_name:
            # Explicit function name: "custom_evals.my_eval:my_task_fn"
            module_path, function_name = eval_name.split(":", 1)
        else:
            module_path = eval_name
            function_name = None

        # Determine the full module path
        if "." in module_path:
            # Already a full path like "custom_evals.my_eval"
            full_module_path = module_path
            # Default function name is the last part of the module path
            if not function_name:
                function_name = module_path.split(".")[-1]
        else:
            # Simple name like "mbpp" → assume inspect_evals
            full_module_path = f"inspect_evals.{module_path}"
            if not function_name:
                function_name = module_path

        logger.info(f"Attempting to import: {full_module_path}")

        # Import the eval module
        eval_module = import_module(full_module_path)

        # Get the task function
        if not hasattr(eval_module, function_name):
            raise AttributeError(
                f"Module '{full_module_path}' does not have function '{function_name}'. "
                f"Available: {dir(eval_module)}"
            )

        task_fn = getattr(eval_module, function_name)

        # Instantiate the task with custom parameters
        task_params = eval_spec.get("task_params", {})
        logger.info(f"Loading eval: {eval_name} with params: {task_params}")
        task = task_fn(**task_params)

        # Cache the task
        _task_cache[cache_key] = task

        return task

    except ImportError as e:
        raise ValueError(
            f"Could not import eval '{eval_name}'. "
            f"For custom evals, ensure the module is in /app/custom_evals/ and accessible. "
            f"Error: {e}"
        )
    except AttributeError as e:
        raise ValueError(f"Eval loading error: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading eval '{eval_name}': {e}")


def create_task_state_from_sample(
    sample: Sample,
    solver_output: str,
    model_name: str = "custom_agent"
) -> TaskState:
    """
    Create an inspect_ai TaskState from a Sample and solver output.

    Args:
        sample: The Sample being processed
        solver_output: The output from your custom solver/agent
        model_name: Name to use for the model in the task state

    Returns:
        TaskState: Populated TaskState for scoring
    """
    from inspect_ai.solver import TaskState
    from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ModelOutput

    # Create message history
    messages = [
        ChatMessageUser(content=str(sample.input))
    ]

    # Create the model output
    output = ModelOutput(
        model=model_name,
        completion=solver_output,
        stop_reason="stop"
    )

    # Create TaskState
    state = TaskState(
        sample_id=sample.id,
        epoch=0,
        input=str(sample.input),
        messages=messages,
        output=output,
        metadata=sample.metadata or {}
    )

    return state


# Sample-related models removed - using evaluate endpoint only


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/act")
def act():
    global _count
    _count += 1
    return {"count": _count}


class SetupRequest(BaseModel):
    """Request to setup/reset environment with optional eval-specific installs"""
    eval_name: Optional[str] = None


@app.post("/setup")
async def setup(request: SetupRequest):
    """
    Setup environment with optional eval-specific installations.

    Some evals require extra dependencies (e.g., swe_bench needs swebench and docker).
    If eval_name is provided, this automatically tries to install inspect_evals[eval_name]
    using uv pip install. Uses try/except to gracefully handle evals without extra deps.
    """
    global _count
    _count = 0
    _sample_results.clear()
    _processing_status.clear()

    install_log = []

    # Try to install eval-specific extras if eval_name provided
    if request.eval_name:
        import subprocess

        try:
            logger.info(f"Attempting to install extras for eval: {request.eval_name}")
            cmd = ["uv", "pip", "install", f"inspect_evals[{request.eval_name}]"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                install_log.append(f"✅ Installed inspect_evals[{request.eval_name}]")
                logger.info(f"Successfully installed extras for {request.eval_name}")
            else:
                # Not an error - eval might not have extras
                stderr_lower = result.stderr.lower()
                if "no extras" in stderr_lower or "does not exist" in stderr_lower:
                    install_log.append(f"ℹ️  No extra dependencies needed for {request.eval_name}")
                    logger.info(f"No extra dependencies found for {request.eval_name} (this is normal)")
                else:
                    # Actual error
                    install_log.append(f"⚠️  Warning: Could not install extras for {request.eval_name}: {result.stderr[:200]}")
                    logger.warning(f"Could not install extras for {request.eval_name}: {result.stderr}")

        except subprocess.TimeoutExpired:
            install_log.append(f"⚠️  Installation timed out after 5 minutes")
            logger.warning("Installation timed out")
        except Exception as e:
            install_log.append(f"⚠️  Installation error: {str(e)[:200]}")
            logger.warning(f"Installation error: {str(e)}")

    return {"ok": True, "install_log": install_log}


@app.post("/reset")
def reset():
    """Legacy reset endpoint - redirects to setup without installs"""
    global _count
    _count = 0
    _sample_results.clear()
    _processing_status.clear()
    return {"ok": True}


@app.get("/state")
def state():
    return {
        "count": _count,
        "total_samples_processed": len(_sample_results),
        "currently_processing": len([k for k, v in _processing_status.items() if v == "processing"])
    }


class EvaluateRequest(BaseModel):
    """Request to run an inspect_ai evaluation"""
    eval_name: str
    task_params: Optional[Dict[str, Any]] = None
    sample: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None


class ModelGenerateRequest(BaseModel):
    """Request from HUD model provider to generate a response"""
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]] = []
    tool_choice: Optional[Any] = None
    config: Dict[str, Any] = {}


@app.post("/model/generate")
async def model_generate(request: ModelGenerateRequest):
    """
    Handle model generate() calls from the HUD ModelAPI provider.

    This endpoint receives generate() calls from inspect_ai running in Docker
    and forwards them to your external agent via HTTP callback.

    Set AGENT_CALLBACK_URL environment variable to your agent's endpoint.
    Example: AGENT_CALLBACK_URL=http://host.docker.internal:9000/generate
    """
    import os
    import httpx

    logger.info(f"Model generate called with {len(request.messages)} messages")

    # Get callback URL from environment
    callback_url = os.getenv("AGENT_CALLBACK_URL")

    if not callback_url:
        # No callback URL configured, return mock response
        logger.warning("No AGENT_CALLBACK_URL configured, returning mock response")
        last_message = request.messages[-1] if request.messages else {}
        user_content = last_message.get("content", "")

        return {
            "content": f"Mock response to: {user_content[:100]}...",
            "model": "hud/agent",
            "stop_reason": "stop"
        }

    try:
        # Forward to external agent
        logger.info(f"Forwarding to agent at {callback_url}")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                callback_url,
                json={
                    "messages": request.messages,
                    "tools": request.tools,
                    "config": request.config
                }
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Received response from agent: {len(result.get('content', ''))} chars")

            return result

    except Exception as e:
        logger.error(f"Error calling agent: {e}")
        return {
            "content": f"Error calling agent: {str(e)}",
            "model": "hud/agent",
            "stop_reason": "error"
        }


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """
    Run a full inspect_ai evaluation using the eval's native solver and scorer.

    This executes the eval exactly as inspect_ai would, using:
    - The eval's dataset
    - The eval's native solver (generate(), basic_agent(), etc.)
    - The eval's native scorer
    - The eval's sandbox configuration
    """
    eval_name = request.eval_name
    task_params = request.task_params or {}
    sample_data = request.sample
    limit = request.limit

    logger.info(f"Starting evaluation: {eval_name} with params: {task_params}, sample: {sample_data is not None}, limit: {limit}")

    try:
        # Import inspect_ai's eval function
        from inspect_ai import eval as inspect_eval
        from inspect_ai.log import read_eval_log

        # Import and register the HUD model provider
        from environment.hud_model import HUDAgentModel  # noqa: F401

        # Load the eval task
        eval_spec = {
            "eval_name": eval_name,
            "task_params": task_params
        }
        task = load_eval_task(eval_spec)

        # Filter dataset based on parameters
        if sample_data is not None:
            # Process single sample provided directly (for parallel processing)
            from inspect_ai.dataset import Sample

            # Convert dict to Sample object
            sample = Sample(
                id=sample_data.get("id"),
                input=sample_data.get("input"),
                target=sample_data.get("target"),
                metadata=sample_data.get("metadata", {}),
                sandbox=sample_data.get("sandbox")
            )
            task.dataset = [sample]
            logger.info(f"Processing single sample: {sample.id}")
        elif limit:
            # Limit number of samples
            task.dataset = task.dataset[:limit]
            logger.info(f"Running eval with {len(task.dataset)} samples (limited)")
        else:
            logger.info(f"Running eval with {len(task.dataset)} samples (full dataset)")

        # Run the evaluation using inspect_ai
        # Use the HUD model provider which will route calls back through MCP
        logs = await inspect_eval(
            task,
            model="hud/agent",  # Routes to your HUD agent
            log_dir="logs"
        )

        # Parse results
        log = logs[0] if logs else None
        if log:
            results = {
                "status": log.status,
                "eval_name": eval_name,
                "samples_completed": len([s for s in log.samples if s.score]),
                "total_samples": len(log.samples),
                "scores": {
                    metric: value.value
                    for metric, value in (log.results.metrics if log.results else {}).items()
                }
            }
        else:
            results = {"status": "no_log", "eval_name": eval_name}

        logger.info(f"Evaluation complete: {results}")

        return {
            "trace_id": str(uuid.uuid4()),
            "status": "completed",
            "results": results
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return {
            "trace_id": str(uuid.uuid4()),
            "status": "error",
            "error": str(e)
        }


# Note: process_sample endpoint and related functions removed
# Use the evaluate endpoint instead which runs full inspect_ai evaluations
