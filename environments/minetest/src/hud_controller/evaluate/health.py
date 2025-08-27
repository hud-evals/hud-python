"""Health evaluator for Minetest environment."""

from . import evaluate


@evaluate.tool()
async def health() -> dict:
    """Return simple health info: computer tool display and process status."""
    ctx = evaluate.env
    svc = ctx.get_service_manager()
    status = {
        "minetest_running": svc.is_minetest_running(),
        "display": ":1",
        "novnc_url": "http://localhost:8080/vnc.html",
    }
    return status

