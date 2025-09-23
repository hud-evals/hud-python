"""Minimal FastAPI environment server (HTTP-based)."""

import logging
import sys
import traceback
from fastapi import FastAPI
from pydantic import BaseModel


from .utils import run_uv_command

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)

app = FastAPI(title="Inspect-AI eval-wrapper API")

_model = ""
_target_eval = ""

_status = "not ready"


class ResetPayload(BaseModel):
    target_eval: str
    model: str


@app.get("/health")
def health():
    return {"status": _status}


@app.post("/reset")
def reset(payload: ResetPayload):
    """Setup and/or reset the environment.
    This is where we'd do a check for extra installation requirements
    of a specific inspect eval, and satisfy those. e.g. sweval"""

    global _target_eval, _model
    _target_eval = payload.target_eval
    _model = payload.model
    try:
        extra_stdout, _extra_stderr = ""
        stdout, stderr = run_uv_command(["sync"])
        try:
            # sorry for the nested try/except
            # some evals have extra installation needed
            extra_stdout, _extra_stderr = run_uv_command(
                ["pip", "install", f"inspect-ai[{_target_eval}]"]
            )
        except Exception as irrelevant:
            pass
        global _status
        _status = "ready"
        return {"ok": True, "stdout": stdout, "stderr": stderr}
    except Exception as e:
        global _status
        _status = "error"
        return {"ok": False, "error": e, "traceback": traceback.format_exc()}


@app.post("/run")
def run(target_eval: str):
    try:
        # uv run inspect eval inspect_evals/
        stdout, stderr = run_uv_command(
            ["run", "inspect", "eval", f"inspect_evals/{_target_eval}"]
        )
        global _status
        _status = "ok"
        return {"ok": True, "stdout": stdout, "stderr": stderr}
    except Exception as e:
        global _status
        _status = "error"
        return {"ok": False, "error": e, "trace back": traceback.format_exc()}


@app.get("/state")
def state():
    return {"model": _model, "target_eval": _target_eval, "status": _status}
