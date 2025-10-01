"""Tools router for environment interaction."""

from hud.server import MCPRouter
from hud.tools.types import EvaluationResult
from server.shared import http_client

router = MCPRouter()


@router.tool
async def act() -> str:
    """Perform one action step in the environment (increment the counter)."""
    resp = await http_client.post("/act")
    data = resp.json()
    return f"Action #{data.get('count', 0)} performed. Current count: {data.get('count', 0)}"


@router.tool
async def setup() -> str:
    """Initialize or reset the environment to its starting state."""
    await http_client.post("/reset")
    return "Counter reset to 0"


@router.tool
async def evaluate(target: int = 10) -> EvaluationResult:
    """Evaluate progress toward the target count and return a reward and done flag."""
    resp = await http_client.get("/state")
    current_count = resp.json().get("count", 0)
    delta = target - current_count
    reward = max(0.0, 1.0 - abs(delta) / target) if target > 0 else current_count
    done = current_count >= target
    return EvaluationResult(
        reward=reward, done=done, content=f"Counter at {current_count}/{target}"
    )
