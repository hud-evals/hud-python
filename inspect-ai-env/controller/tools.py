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
async def evaluate(eval_config: dict = {}) -> EvaluationResult:
    """
    Triggers a long-running evaluation on the backend API and returns
    immediately with the trace_id for tracking.
    """
    try:
        response = await http_client.post(
            "/evaluate",
            json=eval_config,
            timeout=15.0,
        )

        # Raise an exception if the API returns an error (e.g., 400, 500)
        response.raise_for_status()

        data = response.json()
        logger.warning(f"data received by mcp: {data}")
        trace_id = data.get("content", {}).get("trace_id")
        assert trace_id is not None

        return EvaluationResult(
            reward=0.0,
            done=False,
            isError=False,
            content=f"Evaluation successfully started. Track with trace_id: {trace_id}",
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
