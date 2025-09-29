"""Controller tools that call the environment API."""

import json
import httpx
import logging
import sys

from controller import mcp, http_client
from hud.tools.types import EvaluationResult

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@mcp.tool()
async def setup() -> str:
    """Initialize or reset the environment to its starting state."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    resp = await http_client.post("/reset")
    return json.dumps({"status": "ready", "content": resp.json()})


@mcp.tool()
async def evaluate(eval_name: str, task_params: dict = {}, limit: int = None) -> EvaluationResult:
    """
    Run a full inspect_ai evaluation using the eval's native solver and scorer.

    Args:
        eval_name: Name of the eval (e.g., "mbpp", "swe_bench", "gpqa")
        task_params: Parameters to pass to the eval's task function (e.g., {"temperature": 0.5})
        limit: Optional limit on number of samples to evaluate

    This will:
    - Load the eval from inspect_evals
    - Use the eval's native solver (generate(), basic_agent(), etc.)
    - Use the eval's native scorer
    - Return results with scores and metrics
    """
    try:
        response = await http_client.post(
            "/evaluate",
            json={
                "eval_name": eval_name,
                "task_params": task_params,
                "limit": limit
            },
            timeout=600.0,  # 10 minutes for full eval runs
        )

        # Raise an exception if the API returns an error (e.g., 400, 500)
        response.raise_for_status()

        data = response.json()
        logger.info(f"Evaluation response: {data}")

        status = data.get("status", "unknown")
        results = data.get("results", {})

        if status == "completed":
            # Extract score information
            scores = results.get("scores", {})
            score_summary = ", ".join([f"{k}: {v}" for k, v in scores.items()])

            return EvaluationResult(
                reward=scores.get("accuracy", 0.0) if scores else 0.0,
                done=True,
                isError=False,
                content=f"Evaluation complete. Results: {score_summary}\n\nFull results: {json.dumps(results, indent=2)}",
            )
        elif status == "error":
            return EvaluationResult(
                reward=0.0,
                done=True,
                isError=True,
                content=f"Evaluation error: {data.get('error', 'Unknown error')}",
            )
        else:
            return EvaluationResult(
                reward=0.0,
                done=False,
                isError=False,
                content=f"Evaluation status: {status}. Trace ID: {data.get('trace_id')}",
            )

    except httpx.HTTPStatusError as e:
        # The API server responded with an error
        return EvaluationResult(
            reward=0.0,
            done=False,
            isError=True,
            content=f"API Error: {e.response.text}",
        )
    except httpx.RequestError as e:
        # A network-level error occurred (e.g., connection refused)
        return EvaluationResult(
            reward=0.0, done=False, isError=True, content=f"Connection Error: {e}"
        )


@mcp.tool()
async def get_status() -> str:
    """
    Checks and returns the status of the long-running benchmark process.
    The response will indicate if the process is 'running', 'not_running', or 'completed_or_crashed'.
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    print("Sending request to GET /status")
    resp = await http_client.get("/status")

    # Return the server's JSON response as a string
    return json.dumps(resp.json())


@mcp.tool()
async def stop() -> str:
    """
    Stops the currently running benchmark process.
    This will gracefully terminate the process and release the lock.
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    print("Sending request to POST /stop")
    resp = await http_client.post("/stop")

    # Return the server's JSON response as a string
    return json.dumps(resp.json())


@mcp.tool()
async def process_sample(
    sample_data: dict,
    task_config: dict = None,
    eval_spec: dict = None
) -> str:
    """
    Process a single Sample record through the setup -> solver -> scorer pipeline.

    Args:
        sample_data: Sample data dict with fields: input, target, choices, id, metadata, sandbox, files, setup
        task_config: Optional task configuration (timeouts, limits, etc.)
        eval_spec: Optional evaluation specification (setup_commands, solver_type, scorer_config)

    Returns:
        JSON string with processing result including success status, outputs, and score
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    request_data = {
        "sample": sample_data,
        "task_config": task_config or {},
        "eval_spec": eval_spec or {}
    }

    logger.info(f"Processing sample {sample_data.get('id', 'unknown')}")

    try:
        resp = await http_client.post("/process_sample", json=request_data, timeout=60.0)
        resp.raise_for_status()
        result = resp.json()

        logger.info(f"Sample processing completed: success={result.get('success')}")
        return json.dumps(result)

    except httpx.HTTPStatusError as e:
        error_msg = f"Sample processing failed: {e.response.text}"
        logger.error(error_msg)
        return json.dumps({"success": False, "error": error_msg})

    except httpx.RequestError as e:
        error_msg = f"Request failed: {e}"
        logger.error(error_msg)
        return json.dumps({"success": False, "error": error_msg})


@mcp.tool()
async def get_sample_result(sample_id: str) -> str:
    """
    Get the result of a previously processed sample by its ID.

    Args:
        sample_id: The ID of the sample to retrieve results for

    Returns:
        JSON string with the sample result or error message
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    try:
        resp = await http_client.get(f"/sample_result/{sample_id}")
        resp.raise_for_status()
        return json.dumps(resp.json())

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return json.dumps({"error": "Sample result not found"})
        else:
            return json.dumps({"error": f"Failed to get sample result: {e.response.text}"})

    except httpx.RequestError as e:
        return json.dumps({"error": f"Request failed: {e}"})
