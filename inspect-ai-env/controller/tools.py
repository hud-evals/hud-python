"""Controller tools that call the environment API."""

import json
from controller import mcp, http_client
from hud.tools.types import EvaluationResult


@mcp.tool
async def run() -> str:
    """Perform one action step in the environment (increment the counter)."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    status = await http_client.get("/health")
    if status in ["ready", "ok"]:
        resp = await http_client.post("/run")
        data = resp.json()
        return data
    else:
        return {
            "status": status,
            "error": "Something went wrong. Call setup before run",
        }


@mcp.tool
async def setup(target_eval: str, model: str) -> str:
    """Initialize or reset the environment to its starting state."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    resp = await http_client.post(
        "/reset", json=json.dumps({"target_eval": target_eval, "model": model})
    )
    data = resp.json()
    return data


@mcp.tool
async def evaluate(eval_params: dict) -> EvaluationResult:
    """Evaluate progress toward the target count and return a reward and done flag."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    status = await http_client.get("/health")
    if status in ["ready", "ok"]:
        resp = await http_client.post("/run", json=json.dumps(eval_params))
        data = resp.json()
    else:
        return {
            "status": status,
            "error": "Something went wrong.",
        }

    return EvaluationResult(reward=data["reward"], done=data["done"], content=data)
