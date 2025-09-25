"""Controller tools that call the environment API."""

import json
from controller import mcp, http_client
from hud.tools.types import EvaluationResult


@mcp.tool()
async def setup(target_eval: str, model: str) -> str:
    """Initialize or reset the environment to its starting state."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    resp = await http_client.post(
        "/reset", json={"target_eval": target_eval, "model": model}
    )
    data = resp.json()
    return json.dumps({"status": "ready", "content": data})


@mcp.tool()
async def evaluate(eval_config: dict = {}) -> EvaluationResult:
    """Evaluate progress toward the target count and return a reward and done flag."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    resp = await http_client.get("/health")
    status = resp.json().get("content", "error")
    data = {}
    if status in ["ready", "ok"]:
        resp = await http_client.post("/evaluate", json=eval_config)
        data = resp.json()
    else:
        return EvaluationResult(
            reward=0.0,
            done=False,
            isError=True,
            content=f"{status}  {str(status.json())}",
        )

    return EvaluationResult(
        reward=data.get("reward", 0.0),
        done=str(data.get("done", False), content=str(data)),
    )
