"""Minimal FastAPI environment server (HTTP-based)."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import logging
import sys
import uuid
import time
from datetime import datetime
from importlib import import_module

from inspect_ai import Task
from inspect_ai.solver import TaskState, Generate
from inspect_ai.scorer import Target
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant

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
            - eval_name: Name of the eval (e.g., "mbpp", "swe_bench")
            - task_params: Optional parameters to pass to the task function

    Returns:
        Task: The instantiated inspect_ai Task object
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
        # Import the eval module from inspect_evals
        eval_module = import_module(f"inspect_evals.{eval_name}")

        # Get the task function (typically named same as the module)
        task_fn = getattr(eval_module, eval_name)

        # Instantiate the task with custom parameters
        task_params = eval_spec.get("task_params", {})
        logger.info(f"Loading eval: {eval_name} with params: {task_params}")
        task = task_fn(**task_params)

        # Cache the task
        _task_cache[cache_key] = task

        return task

    except ImportError as e:
        raise ValueError(f"Could not import eval '{eval_name}': {e}")
    except AttributeError as e:
        raise ValueError(f"Eval '{eval_name}' does not have a task function named '{eval_name}': {e}")


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


class Sample(BaseModel):
    """Sample model matching inspect_ai Sample structure"""
    input: Union[str, List[Dict[str, Any]]]
    target: Union[str, List[str]] = ""
    choices: Optional[List[str]] = None
    id: Union[int, str, None] = None
    metadata: Optional[Dict[str, Any]] = None
    sandbox: Optional[Dict[str, Any]] = None
    files: Optional[Dict[str, str]] = None
    setup: Optional[str] = None


class SampleProcessRequest(BaseModel):
    """Request to process a single sample"""
    sample: Sample
    task_config: Optional[Dict[str, Any]] = None
    eval_spec: Optional[Dict[str, Any]] = None


class SampleResult(BaseModel):
    """Result of processing a single sample"""
    sample_id: Union[int, str]
    success: bool
    setup_output: Optional[str] = None
    solver_output: Optional[str] = None
    score: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    timestamp: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/act")
def act():
    global _count
    _count += 1
    return {"count": _count}


@app.post("/reset")
def reset():
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
    limit: Optional[int] = None


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
    limit = request.limit

    logger.info(f"Starting evaluation: {eval_name} with params: {task_params}, limit: {limit}")

    try:
        # Import inspect_ai's eval function
        from inspect_ai import eval as inspect_eval
        from inspect_ai.log import read_eval_log

        # Load the eval task
        eval_spec = {
            "eval_name": eval_name,
            "task_params": task_params
        }
        task = load_eval_task(eval_spec)

        # Limit dataset if requested
        if limit:
            task.dataset = task.dataset[:limit]

        logger.info(f"Running eval with {len(task.dataset)} samples")

        # Run the evaluation using inspect_ai
        # This will use the eval's native solver and scorer
        logs = await inspect_eval(
            task,
            model="openai/gpt-4o-mini",  # TODO: Make this configurable
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


@app.post("/process_sample")
async def process_sample(request: SampleProcessRequest) -> SampleResult:
    """
    Process a single sample through the setup -> solver -> scorer pipeline.
    This is the main endpoint for inspect-ai integration.
    """
    sample = request.sample
    sample_id = sample.id or str(uuid.uuid4())

    logger.info(f"Processing sample {sample_id}")
    start_time = time.time()

    # Mark as processing
    _processing_status[sample_id] = "processing"

    try:
        # Step 1: Setup phase
        setup_output = await run_sample_setup(sample, request.task_config, request.eval_spec)
        logger.info(f"Setup completed for sample {sample_id}")

        # Step 2: Solver phase (main execution)
        solver_output = await run_sample_solver(sample, setup_output, request.task_config, request.eval_spec)
        logger.info(f"Solver completed for sample {sample_id}")

        # Step 3: Scoring phase
        score = await run_sample_scorer(sample, solver_output, request.task_config, request.eval_spec)
        logger.info(f"Scoring completed for sample {sample_id}")

        processing_time = time.time() - start_time

        result = SampleResult(
            sample_id=sample_id,
            success=True,
            setup_output=setup_output,
            solver_output=solver_output,
            score=score,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

        # Store result
        _sample_results[sample_id] = result
        _processing_status[sample_id] = "completed"

        return result

    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {e}")
        processing_time = time.time() - start_time

        result = SampleResult(
            sample_id=sample_id,
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

        _sample_results[sample_id] = result
        _processing_status[sample_id] = "error"

        return result


@app.get("/sample_result/{sample_id}")
def get_sample_result(sample_id: str):
    """Get the result of a processed sample"""
    if sample_id not in _sample_results:
        raise HTTPException(status_code=404, detail="Sample result not found")
    return _sample_results[sample_id]


@app.get("/sample_status/{sample_id}")
def get_sample_status(sample_id: str):
    """Get the processing status of a sample"""
    status = _processing_status.get(sample_id, "not_found")
    return {"sample_id": sample_id, "status": status}


async def run_sample_setup(sample: Sample, task_config: Dict[str, Any] = None, eval_spec: Dict[str, Any] = None) -> str:
    """
    Custom setup logic for the sample.
    Override this method to implement your specific setup requirements.
    """
    setup_commands = []

    if eval_spec and "setup_commands" in eval_spec:
        setup_commands.extend(eval_spec["setup_commands"])

    if sample.setup:
        setup_commands.append(sample.setup)

    # For now, just simulate setup execution
    if setup_commands:
        logger.info(f"Executing setup commands: {setup_commands}")
        await asyncio.sleep(0.1)  # Simulate work
        return f"Setup completed: {'; '.join(setup_commands)}"
    else:
        return "No setup required"


async def run_sample_solver(sample: Sample, setup_output: str, task_config: Dict[str, Any] = None, eval_spec: Dict[str, Any] = None) -> str:
    """
    Custom solver logic for the sample.
    This is where your Docker container agent or custom solver runs.

    Args:
        sample: The sample to solve
        setup_output: Output from the setup phase
        task_config: Task configuration
        eval_spec: Eval specification with eval_name and task_params

    Returns:
        str: The solver output (model completion)
    """
    solver_type = eval_spec.get("solver_type", "custom_agent") if eval_spec else "custom_agent"

    logger.info(f"Running solver type: {solver_type} for sample: {sample.id}")

    # Option 1: Use your custom Docker container agent
    if solver_type == "custom_agent":
        # TODO: Integrate with your Docker container here
        # This is where you'd send the sample to your custom agent
        # and get back the solution

        # For now, using a placeholder that demonstrates the expected format
        # For MBPP, this should return Python code
        # For SWE-bench, this should return git diff or patch
        output = await run_custom_docker_agent(sample, eval_spec)

    # Option 2: Use the eval's default solver (inspect_ai's basic_agent, generate(), etc.)
    elif solver_type == "eval_default":
        # Load the eval task and use its solver
        task = load_eval_task(eval_spec)

        # The eval's solver would typically run here
        # This requires running inspect_ai's solve pipeline, which is complex
        # For now, we'll focus on custom_agent mode
        raise NotImplementedError("eval_default solver not yet implemented - use custom_agent")

    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")

    return output


async def run_custom_docker_agent(sample: Sample, eval_spec: Dict[str, Any]) -> str:
    """
    This function is called from within the Docker container's environment server.

    IMPORTANT: The actual agent that will solve this sample is running OUTSIDE
    this Docker container, in run_task.py. The agent calls the process_sample MCP tool,
    which routes here.

    Your custom solving logic should go here. This could be:
    - Running a local model
    - Calling an API
    - Executing code in a sandbox
    - Or whatever custom logic you need

    For now, this is a placeholder that returns eval-specific mock responses.
    In production, you would implement your actual solving logic here.

    Args:
        sample: The sample to solve
        eval_spec: Eval specification

    Returns:
        str: The solver output (format depends on eval type)
    """
    eval_name = eval_spec.get("eval_name", "unknown")

    logger.info(f"Custom solver for eval: {eval_name}, sample: {sample.id}")
    logger.info(f"Sample input: {str(sample.input)[:200]}...")

    # TODO: Replace this with your actual solving logic
    # For example:
    # - Use a local LLM
    # - Call an external API
    # - Run code generation model
    # - Execute multi-step reasoning

    # Simulate some processing time
    await asyncio.sleep(0.1)

    # Return eval-specific placeholder responses
    # In production, your agent would generate real solutions
    if eval_name == "mbpp":
        # For MBPP, return Python code wrapped in markdown
        # The MBPP scorer will execute this code against test cases
        return f"```python\ndef solution():\n    # TODO: Implement solution for: {sample.input[:50]}...\n    pass\n```"
    elif eval_name == "swe_bench":
        # For SWE-bench, return code changes/patches
        return f"# Modified files for issue: {sample.id}\n# TODO: Implement solution"
    else:
        # Generic response
        return f"Agent output for {eval_name}: Processing {sample.input[:100]}..."


async def run_sample_scorer(sample: Sample, solver_output: str, task_config: Dict[str, Any] = None, eval_spec: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Score the sample using the eval's native scorer.

    Args:
        sample: The sample that was processed
        solver_output: The output from the solver
        task_config: Task configuration
        eval_spec: Eval specification with eval_name and task_params

    Returns:
        Dict: Score results with value, explanation, and metadata
    """
    if not eval_spec or not eval_spec.get("eval_name"):
        logger.warning("No eval_spec provided, using simple string match scoring")
        return {
            "value": 1.0 if sample.target and str(sample.target) in solver_output else 0.0,
            "explanation": "Simple string match scoring (no eval specified)"
        }

    try:
        # Load the eval task to get its scorer
        task = load_eval_task(eval_spec)

        logger.info(f"Using native scorer for eval: {eval_spec['eval_name']}")

        # Create TaskState from the sample and solver output
        task_state = create_task_state_from_sample(
            sample,
            solver_output,
            model_name=eval_spec.get("model_name", "custom_agent")
        )

        # Create Target from the sample
        target = Target(sample.target)

        # Run the eval's scorer
        score_result = await task.scorer(task_state, target)

        # Convert Score object to dict
        score_dict = {
            "value": score_result.value,
            "explanation": score_result.explanation or "",
            "answer": score_result.answer or solver_output,
        }

        # Include metadata if present
        if score_result.metadata:
            score_dict["metadata"] = score_result.metadata

        logger.info(f"Score result: {score_dict['value']}")

        return score_dict

    except Exception as e:
        logger.error(f"Error running eval scorer: {e}", exc_info=True)
        # Fallback to simple scoring
        return {
            "value": 0.0,
            "explanation": f"Scorer error: {str(e)}",
            "error": str(e)
        }
