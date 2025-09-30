"""Minimal FastAPI environment server (HTTP-based)."""

import logging
import sys
import os
from datetime import datetime
import signal
import subprocess
import time
import psutil
import traceback
import json

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import uuid

# from importlib import import_module
from pathlib import Path

# Add current directory to sys.path to enable importing local inspect_evals
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser, ModelOutput

from .utils import (
    is_pid_running,
    get_lock_data,
    write_lock_data,
    get_process_status,
    LOG_FILE_PATH,
    LOCK_FILE_PATH,
)

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# globals for tracking state


_model = ""
_target_eval = ""
_process = None  # Store the subprocess.Popen object
_processing_status = {}  # Track processing status

app = FastAPI(title="Inspect-AI eval-wrapper API")


class SetupRequest(BaseModel):
    """Request to setup/reset environment and model_wrapper"""

    eval_name: str
    model_name: str


class EvaluateRequest(BaseModel):
    """Request to run an inspect_ai evaluation"""

    eval_name: str
    task_params: Optional[Dict[str, Any]] = None
    sample: Optional[Dict[str, Any]] = None


class ModelGenerateRequest(BaseModel):
    """Request from HUD model provider to generate a response"""

    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]] = []
    tool_choice: Optional[Any] = None
    config: Dict[str, Any] = {}


@app.get("/health")
def health():
    return {"ok": True, "content": {"status": get_process_status()}}


@app.get("/status")
def status():
    return {
        "model": _model,
        "target_eval": _target_eval,
        "status": get_process_status(),
    }


@app.post("/reset")
async def reset(request: SetupRequest):
    """
    Setup environment with optional eval-specific installations.

    Some evals require extra dependencies (e.g., swe_bench needs swebench and docker).
    If eval_name is provided, this automatically tries to install inspect_evals[eval_name]
    using uv pip install. Uses try/except to gracefully handle evals without extra deps.
    """
    global _model, _target_eval

    _processing_status.clear()

    # Store model and eval names
    _model = request.model_name
    _target_eval = request.eval_name

    logger.info(f"Reset: model={_model}, eval={_target_eval}")

    install_log = []

    # Try to install eval-specific extras if eval_name provided
    if request.eval_name:
        import subprocess

        try:
            logger.info(f"Attempting to install extras for eval: {request.eval_name}")
            cmd = ["uv", "pip", "install", f"inspect_evals[{request.eval_name}]"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                install_log.append(f"✅ Installed inspect_evals[{request.eval_name}]")
                logger.info(f"Successfully installed extras for {request.eval_name}")
            else:
                # Not an error - eval might not have extras
                stderr_lower = result.stderr.lower()
                if "no extras" in stderr_lower or "does not exist" in stderr_lower:
                    install_log.append(
                        f"ℹ️  No extra dependencies needed for {request.eval_name}"
                    )
                    logger.info(
                        f"No extra dependencies found for {request.eval_name} (this is normal)"
                    )
                else:
                    # Actual error
                    install_log.append(
                        f"⚠️  Warning: Could not install extras for {request.eval_name}: {result.stderr[:200]}"
                    )
                    logger.warning(
                        f"Could not install extras for {request.eval_name}: {result.stderr}"
                    )

        except subprocess.TimeoutExpired:
            install_log.append(f"⚠️  Installation timed out after 5 minutes")
            logger.warning("Installation timed out")
        except Exception as e:
            install_log.append(f"⚠️  Installation error: {str(e)[:200]}")
            logger.warning(f"Installation error: {str(e)}")

    return {"ok": True, "install_log": install_log}


@app.post("/model/generate")
async def model_generate(request: ModelGenerateRequest):
    """
    Handle model generate() calls from the HUD ModelAPI provider.

    This endpoint receives generate() calls from inspect_ai running in Docker
    and forwards them to your external agent via HTTP callback.

    Set AGENT_CALLBACK_URL environment variable to your agent's endpoint.
    Example: AGENT_CALLBACK_URL=http://host.docker.internal:9000/generate
    """
    import os
    import httpx

    logger.info(f"Model generate called with {len(request.messages)} messages")

    # Get callback URL from environment
    callback_url = os.getenv("AGENT_CALLBACK_URL")

    if not callback_url:
        # No callback URL configured, return mock response
        logger.warning("No AGENT_CALLBACK_URL configured, returning mock response")
        last_message = request.messages[-1] if request.messages else {}
        user_content = last_message.get("content", "")

        return {
            "content": f"Mock response to: {user_content[:100]}...",
            "model": "hud/agent",
            "stop_reason": "stop",
        }

    try:
        # Forward to external agent
        logger.info(f"Forwarding to agent at {callback_url}")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                callback_url,
                json={
                    "messages": request.messages,
                    "tools": request.tools,
                    "config": request.config,
                },
            )
            response.raise_for_status()

            result = response.json()
            logger.info(
                f"Received response from agent: {len(result.get('content', ''))} chars"
            )

            return result

    except Exception as e:
        logger.error(f"Error calling agent: {e}")
        return {
            "content": f"Error calling agent: {str(e)}",
            "model": "hud/agent",
            "stop_reason": "error",
        }


# @app.post("/evaluate")
# async def evaluate(request: EvaluateRequest):
#     """
#     Run a full inspect_ai evaluation using the eval's native solver and scorer.

#     This executes the eval exactly as inspect_ai would, using:
#     - The eval's dataset
#     - The eval's native solver (generate(), basic_agent(), etc.)
#     - The eval's native scorer
#     - The eval's sandbox configuration
#     """
#     eval_name = request.eval_name
#     task_params = request.task_params or {}
#     sample_data = request.sample
#     limit = request.limit

#     logger.info(
#         f"Starting evaluation: {eval_name} with params: {task_params}, sample: {sample_data is not None}, limit: {limit}"
#     )

#     try:

#         # Parse results
#         log = logs[0] if logs else None
#         if log:
#             results = {
#                 "status": log.status,
#                 "eval_name": eval_name,
#                 "samples_completed": len([s for s in log.samples if s.score]),
#                 "total_samples": len(log.samples),
#                 "scores": {
#                     metric: value.value
#                     for metric, value in (
#                         log.results.metrics if log.results else {}
#                     ).items()
#                 },
#             }
#         else:
#             results = {"status": "no_log", "eval_name": eval_name}

#         logger.info(f"Evaluation complete: {results}")

#         return {
#             "trace_id": str(uuid.uuid4()),
#             "status": "completed",
#             "results": results,
#         }

#     except Exception as e:
#         logger.error(f"Evaluation failed: {e}", exc_info=True)
#         return {"trace_id": str(uuid.uuid4()), "status": "error", "error": str(e)}


@app.post("/evaluate")
async def evaluate(eval_config: dict):
    """
    Creates and starts a new evaluation.
    Returns immediately with a trace_id to track the evaluation.
    """
    global _process

    # Check if there's already a lock (running or completed process)
    lock_data = get_lock_data()
    if lock_data is not None:
        raise HTTPException(
            status_code=409,
            detail="An Inspect-ai process is already running or has completed. Call /reset to clear.",
        )

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
        f"hud/{_model}",  # Use HUD model wrapper
        "--sandbox",
        "local",
        "--log-dir",
        "logs",
    ] + eval_params
    full_commands = [str(x) for x in full_commands]
    logger.warning(f"full commands: {full_commands}")

    trace_id = f"inspectai_{_target_eval}_{_model.split('/')[-1]}_{datetime.now().strftime('%y%m%d_%H%M%S')}"

    # --- Launch the Process ---
    try:
        log_file = open(LOG_FILE_PATH, "w")
        _process = subprocess.Popen(full_commands, stdout=log_file, stderr=log_file)

        # # Import inspect_ai's eval function
        # from inspect_ai import eval as inspect_eval
        # from inspect_ai.log import read_eval_log

        # # Import and register the HUD model provider
        # from environment.hud_model import HUDAgentModel  # noqa: F401

        # # Load the eval task
        # eval_spec = {"eval_name": eval_name, "task_params": task_params}
        # task = load_eval_task(eval_spec)

        # # Convert dict to Sample object
        # sample = Sample(
        #     id=sample_data.get("id"),
        #     input=sample_data.get("input"),
        #     target=sample_data.get("target"),
        #     metadata=sample_data.get("metadata", {}),
        #     sandbox=sample_data.get("sandbox"),
        # )
        # task.dataset = [sample]
        # logger.info(f"Processing single sample: {sample.id}")

        # Run the evaluation using inspect_ai
        # Use the HUD model provider which will route calls back through MCP
        # logs = await inspect_eval(
        #     task, model="hud/agent", log_dir="logs"  # Routes to your HUD agent
        # )

        # Write initial lock data with running status
        lock_data = {
            "status": "running",
            "pid": _process.pid,
            "trace_id": trace_id,
            "started_at": datetime.now().isoformat(),
        }
        write_lock_data(lock_data)

        return {
            "message": "Process launched successfully.",
            "pid": _process.pid,
            "trace_id": trace_id,
        }

    except Exception as e:
        # Clean up on failure
        if os.path.exists(LOCK_FILE_PATH):
            os.remove(LOCK_FILE_PATH)
        raise HTTPException(
            status_code=500,
            detail=f"Something has gone terribly wrong...\n{traceback.format_exc()}. Failed to launch process: {str(e)}",
        )


@app.post("/stop")
async def stop_process():
    """Stops the running process gracefully."""
    global _process

    lock_data = get_lock_data()
    if lock_data is None:
        raise HTTPException(status_code=404, detail="No process is currently running.")

    # If already completed or crashed, just return
    if lock_data.get("status") in ["completed", "crashed", "stopped"]:
        return {
            "message": f"Process already {lock_data['status']}. Call /reset to clear."
        }

    pid = lock_data.get("pid")
    if pid is None or not is_pid_running(pid):
        # Update status to crashed since process is gone
        status_data = {
            "status": "crashed",
            "message": "Process was no longer running when stop was called",
        }
        write_lock_data(status_data)
        raise HTTPException(status_code=404, detail="No process is currently running.")

    try:
        # Use the subprocess object if available for more reliable termination
        if _process and _process.poll() is None:  # Process is still running
            # 1. Graceful termination
            _process.terminate()

            # Wait for graceful shutdown
            try:
                _process.wait(timeout=3.0)  # Wait up to 3 seconds
                process_stopped = True
            except subprocess.TimeoutExpired:
                # 2. Force kill if still alive
                _process.kill()
                try:
                    _process.wait(timeout=2.0)  # Wait up to 2 more seconds
                    process_stopped = True
                except subprocess.TimeoutExpired:
                    process_stopped = False
        else:
            # Fallback: use PID-based killing if subprocess object not available
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                try:
                    os.kill(pid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass

            # Wait briefly for graceful shutdown
            for _ in range(15):  # 3 seconds total
                if not is_pid_running(pid):
                    process_stopped = True
                    break
                time.sleep(0.2)
            else:
                # Force kill
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass

                # Wait a bit more
                for _ in range(10):  # 2 more seconds
                    if not is_pid_running(pid):
                        process_stopped = True
                        break
                    time.sleep(0.2)
                else:
                    process_stopped = False

        # Update lock with appropriate status
        if process_stopped:
            status_data = {
                "status": "stopped",
                "message": "Process was manually stopped. It can be resumed.",
                "return_code": -1,
            }
            write_lock_data(status_data)
            return {"message": f"Eval process {pid} stopped successfully."}
        else:
            status_data = {
                "status": "stopping",
                "message": "Stop signal sent but process may still be running. Check status again.",
                "return_code": -1,
                "stop_requested_at": datetime.now().isoformat(),
            }
            write_lock_data(status_data)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to stop eval process {pid}. Process may still be running.",
            )

    except Exception as e:
        # Update the lock to indicate stop was attempted
        status_data = {
            "status": "stopping",
            "message": f"Stop attempted but encountered error: {str(e)}",
            "return_code": -1,
            "stop_requested_at": datetime.now().isoformat(),
        }
        write_lock_data(status_data)

        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while stopping the process: {str(e)}.",
        )


# TODO: add resume endpoint
