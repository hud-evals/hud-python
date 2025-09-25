"""Minimal FastAPI environment server (HTTP-based)."""

import logging
import sys
import os
import warnings
from datetime import datetime


from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import traceback

from .utils import run_eval_and_log

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


@app.get("/health")
def health():
    return {"ok": True, "content": _status}


@app.post("/reset")
def reset():
    """Setup and/or reset the environment.
    This is where we'd do a check for extra installation requirements
    of a specific inspect eval, and satisfy those. e.g. sweval"""

    global _target_eval, _model, _status
    _target_eval = os.getenv("TARGET_EVAL", "specify_target_eval_in_the_.env")
    _model = os.getenv("MODEL", "specify_model_in_the_.env")
    logger.warning(f"Set up model and eval. Model: {_model}, Eval: {_target_eval}")
    # TODO: setup local model if needed
    extra_stdout = ""
    extra_stderr = ""

    # try:
    #     # some evals have extra installation needed
    #     extra_stdout, extra_stderr = run_command(
    #         ["uv", "pip", "install", f"inspect-ai[{_target_eval}]"]
    #     )
    # except Exception as e:
    #     pass
    _status = "ready"
    return {"ok": True}


@app.post("/evaluate")
async def evaluate(eval_config: dict):
    """
    Creates and starts a new evaluation.
    Returns immediately with a trace_id to track the evaluation.
    """
    global _status

    eval_params = []
    if eval_config != {}:
        for k, v in eval_config.items():
            eval_params.append(f"--{k}")
            eval_params.append(v)
    logger.warning(
        f"starting inspect-eval run. info: eval_config: {eval_params}, type {type(eval_params)}"
    )
    try:

        full_commands = [
            "inspect",
            "eval",
            f"/app/inspect_evals/{_target_eval}",
            "--model",
            _model,
        ] + eval_params
        full_commands = [str(x) for x in full_commands]
        logger.warning(f"full commands: {full_commands}")

        trace_id = f"inspectai_{_target_eval}_{_model.split('/')[-1]}_{datetime.now().strftime('%y%m%d_%H%M%S')}"

        # Create the background task using asyncio.create_task to get a handle to it
        task = asyncio.create_task(run_eval_and_log(trace_id, full_commands))

        # Store the task handle in our registry so we can check its status
        # evaluation_tasks[trace_id] = task
        _status = "ok"
        return {"ok": True, "content": {"trace_id": trace_id}}

    except Exception as e:
        _status = "error"
        logger.warning(
            f"Something has gone terribly wrong...\n{traceback.format_exc()}"
        )


@app.get("/state")
def state():
    return {"model": _model, "target_eval": _target_eval, "status": _status}
