"""Minimal FastAPI environment server (HTTP-based)."""

import logging
import sys
import os
import warnings
from datetime import datetime
import signal
import subprocess
import time

from fastapi import FastAPI, HTTPException
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

LOCK_FILE_PATH = "/tmp/long_running_process.lock"
LOG_FILE_PATH = "/tmp/benchmark.log"
_model = ""
_target_eval = ""

app = FastAPI(title="Inspect-AI eval-wrapper API")


def is_pid_running(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def get_pid_from_lock_file():
    try:
        with open(LOCK_FILE_PATH, "r") as f:
            return int(f.read().strip())
    except (IOError, ValueError):
        return None


def get_process_status():
    """Internal function to check process status and clean up stale locks."""
    pid = get_pid_from_lock_file()

    if pid is None:
        return {"status": "not_running"}

    if is_pid_running(pid):
        return {"status": "running", "pid": pid, "log_path": LOG_FILE_PATH}
    else:
        try:
            os.remove(LOCK_FILE_PATH)
        except OSError:
            pass

        return {
            "status": "completed_or_crashed",
            "message": f"Process with PID {pid} is no longer running. Stale lock file removed.",
        }


@app.get("/health")
def health():
    return {"ok": True, "content": {"status": get_process_status()}}


@app.post("/reset")
def reset():
    """Setup and/or reset the environment.
    This is where we'd do a check for extra installation requirements
    of a specific inspect eval, and satisfy those. e.g. sweval"""

    global _target_eval, _model
    _target_eval = os.getenv("TARGET_EVAL", "specify_target_eval_in_the_.env")
    _model = os.getenv("MODEL", "specify_model_in_the_.env")
    logger.warning(f"Set up model and eval. Model: {_model}, Eval: {_target_eval}")
    # TODO: setup local model if needed
    # TODO: extra install step
    extra_stdout = ""
    extra_stderr = ""

    # try:
    #     # some evals have extra installation needed
    #     extra_stdout, extra_stderr = run_command(
    #         ["uv", "pip", "install", f"inspect-ai[{_target_eval}]"]
    #     )
    # except Exception as e:
    #     pass

    return {"ok": True}


@app.post("/evaluate")
async def evaluate(eval_config: dict):
    """
    Creates and starts a new evaluation.
    Returns immediately with a trace_id to track the evaluation.
    """

    eval_params = []
    if eval_config != {}:
        for k, v in eval_config.items():
            eval_params.append(f"--{k}")
            eval_params.append(v)
    logger.warning(
        f"starting inspect-eval run. info: eval_config: {eval_params}, type {type(eval_params)}"
    )

    full_commands = [
        "uv",
        "run",
        "inspect",
        "eval",
        f"/app/inspect_evals/{_target_eval}",
        "--model",
        _model,
    ] + eval_params
    full_commands = [str(x) for x in full_commands]
    logger.warning(f"full commands: {full_commands}")

    trace_id = f"inspectai_{_target_eval}_{_model.split('/')[-1]}_{datetime.now().strftime('%y%m%d_%H%M%S')}"

    # --- Atomic Lock Acquisition ---
    try:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        fd = os.open(LOCK_FILE_PATH, flags)
    except FileExistsError:
        raise HTTPException(
            status_code=409,
            detail="An Inspect-ai process is already running.",  # Conflict
        )

    # --- If Lock Acquired, Launch the Process ---
    try:

        log_file = open(LOG_FILE_PATH, "w")

        process = subprocess.Popen(full_commands, stdout=log_file, stderr=log_file)

        with os.fdopen(fd, "w") as f:
            f.write(str(process.pid))

        return {
            "message": "Process launched successfully.",
            "pid": process.pid,
            "trace_id": trace_id,
        }

    except Exception as e:
        os.remove(LOCK_FILE_PATH)
        raise HTTPException(
            status_code=500,
            detail=f"Something has gone terribly wrong...\n{traceback.format_exc()}. Failed to launch process: {str(e)}",
        )


@app.get("/state")
def state():
    return {
        "model": _model,
        "target_eval": _target_eval,
        "status": get_process_status(),
    }


@app.post("/stop")
async def stop_process():
    """Stops the running process gracefully."""
    pid = get_pid_from_lock_file()

    if pid is None or not is_pid_running(pid):
        if os.path.exists(LOCK_FILE_PATH):
            os.remove(LOCK_FILE_PATH)
        raise HTTPException(status_code=404, detail="No process is currently running.")

    try:
        # 1. Graceful shutdown with SIGTERM
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            if not is_pid_running(pid):
                break
            time.sleep(0.5)

        # 2. Force kill if still alive
        if is_pid_running(pid):
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)

        # 3. Clean up
        os.remove(LOCK_FILE_PATH)

        if not is_pid_running(pid):
            return {"message": f"Process {pid} stopped successfully."}
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to stop process {pid}."
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while stopping the process: {str(e)}",
        )
