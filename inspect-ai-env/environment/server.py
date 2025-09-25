"""Minimal FastAPI environment server (HTTP-based)."""

import os
import logging
import sys
import traceback
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel


from .utils import run_command

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
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
    return {"ok": True, "content": _status}


@app.post("/reset")
def reset(payload: ResetPayload):
    """Setup and/or reset the environment.
    This is where we'd do a check for extra installation requirements
    of a specific inspect eval, and satisfy those. e.g. sweval"""

    global _target_eval, _model, _status
    _target_eval = payload.target_eval
    _model = payload.model
    # TODO: setup local model if needed
    extra_stdout = ""
    extra_stderr = ""

    try:
        # some evals have extra installation needed
        extra_stdout, extra_stderr = run_command(
            ["uv", "pip", "install", f"inspect-ai[{_target_eval}]"]
        )
    except Exception as e:
        pass
    _status = "ready"
    return {"ok": True}


@app.post("/evaluate")
def evaluate(eval_config: dict = {}):
    global _status
    logger.warning(
        f"starting inspect-eval run. info: eval_config: {eval_config}, type {type(eval_config)}"
    )
    eval_params = []
    if eval_config != {}:
        for k, v in eval_config.items():
            eval_params.append(f"--{k}")
            eval_params.append(v)
    logger.warning(
        f"starting inspect-eval run. info: eval_config: {eval_params}, type {type(eval_params)}"
    )
    try:
        stdout, stderr = run_command(
            [
                "inspect",
                "eval",
                f"inspect_evals/{_target_eval}",
                "--model",
                _model,
            ]
            + eval_params
        )
        logger.warning(f"full commands: {["inspect","eval",f"inspect_evals/{_target_eval}","--model",_model,] + eval_params}"
        logger.warning(f"run_command result: {stdout}\n{stderr}")

        _status = "ok"
        return {"ok": True, "info": f"stdout: {stdout}, stderr: {stderr}"}
    except Exception as e:
        _status = "error"
        return {
            "ok": False,
            "content": str(eval_config),
            "info": f"{traceback.format_exc()}",
        }


@app.get("/state")
def state():
    return {"model": _model, "target_eval": _target_eval, "status": _status}
