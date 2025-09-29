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
async def setup(eval_name: str = None) -> str:
    """
    Initialize or reset the environment to its starting state.

    Args:
        eval_name: Optional eval name (e.g., "swe_bench", "mbpp"). If provided,
                   will attempt to install eval-specific dependencies automatically.

    Some evals require additional dependencies (e.g., swe_bench needs swebench>=3.0.15 and docker).
    When eval_name is provided, this tool automatically tries to install inspect_evals[eval_name]
    with a try/except to handle evals that don't have extra dependencies.
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    resp = await http_client.post("/setup", json={"eval_name": eval_name})
    return json.dumps({"status": "ready", "content": resp.json()})


@mcp.tool()
async def evaluate(
    eval_name: str, sample: dict, task_params: dict = {}, limit: int = None
) -> EvaluationResult:
    """
    Run a full inspect_ai evaluation using the eval's native solver and scorer.

    Args:
        eval_name: Name of the eval (e.g., "mbpp", "swe_bench", "gpqa")
        sample: Single sample dict to process.
                This is used for parallel processing where each container gets one sample.
                Sample should be in inspect_ai Sample format (id, input, target, metadata, etc.)
        task_params: Parameters to pass to the eval's task function (e.g., {"temperature": 0.5})

        limit: Optional limit on number of samples to evaluate (only used if sample is None)

    This will:
    - Load the eval from inspect_evals
    - Use the eval's native solver (generate(), basic_agent(), etc.)
    - Use the eval's native scorer
    - Return results with scores and metrics

    For parallel processing: Pass a single sample dict. The eval will be run with just that one sample.
    """
    try:
        response = await http_client.post(
            "/evaluate",
            json={
                "eval_name": eval_name,
                "task_params": task_params,
                "sample": sample,
                "limit": limit,
            },
            timeout=60.0,
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
    Checks and returns the status of the process.
    The response will indicate if the process is 'not_started', 'running', or 'completed', or 'crashed'.
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

    return json.dumps(resp.json())
