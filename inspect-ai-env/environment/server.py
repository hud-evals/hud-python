"""Minimal FastAPI environment server (HTTP-based)."""

import logging
import sys
import traceback
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel


from .utils import run_uv_command

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)

# globals for tracking state
_model = ""
_target_eval = ""
_status = "not ready"

app = FastAPI(title="Inspect-AI eval-wrapper API")


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

    global _target_eval, _model, _status
    _target_eval = payload.target_eval
    _model = payload.model
    try:
        result = subprocess.run(
            ["pip", "install", "uv"],
            capture_output=True,
            text=True,
            check=True,  # This will raise a CalledProcessError if the command fails
        )
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
        _status = "ready"
        return {"ok": True, "stdout": stdout, "stderr": stderr}
    except Exception as e:
        _status = "error"
        return {"ok": False, "error": e, "traceback": traceback.format_exc()}


@app.post("/run")
def run(target_eval: str):
    global _status
    try:
        # uv run inspect eval inspect_evals/
        stdout, stderr = run_uv_command(
            ["run", "inspect", "eval", f"inspect_evals/{_target_eval}"]
        )
        _status = "ok"
        return {"ok": True, "stdout": stdout, "stderr": stderr}
    except Exception as e:
        _status = "error"
        return {"ok": False, "error": e, "trace back": traceback.format_exc()}


@app.get("/state")
def state():
    return {"model": _model, "target_eval": _target_eval, "status": _status}
