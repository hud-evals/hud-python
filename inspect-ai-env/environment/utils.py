# from typing import Dict, Any
# from pathlib import Path
import logging
import sys
import psutil
import json

# # Add current directory to sys.path to enable importing local inspect_evals
# if str(Path.cwd()) not in sys.path:
#     sys.path.insert(0, str(Path.cwd()))
# from inspect_ai import Task

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

LOCK_FILE_PATH = "/tmp/long_running_process.lock"
LOG_FILE_PATH = "/tmp/benchmark.log"


# def load_eval_task(eval_spec: Dict[str, Any]) -> Task:
#     """
#     Dynamically load and instantiate an inspect_evals Task.

#     Args:
#         eval_spec: Dict containing:
#             - eval_name: Name/path of the eval. Can be:
#                 * Simple name: "mbpp" → imports from inspect_evals.mbpp
#                 * Module path: "custom_evals.my_eval" → imports from that module path
#                 * Full path with function: "custom_evals.my_eval:my_task_fn"
#             - task_params: Optional parameters to pass to the task function

#     Returns:
#         Task: The instantiated inspect_ai Task object

#     Examples:
#         # Official inspect_evals
#         {"eval_name": "mbpp"}  → import inspect_evals.mbpp; mbpp()

#         # Custom eval (auto-detect function name)
#         {"eval_name": "custom_evals.my_eval"}  → import custom_evals.my_eval; my_eval()

#         # Custom eval with explicit function
#         {"eval_name": "custom_evals.my_eval:custom_task"}  → import custom_evals.my_eval; custom_task()
#     """
#     eval_name = eval_spec.get("eval_name")
#     if not eval_name:
#         raise ValueError("eval_spec must contain 'eval_name'")

#     # Check cache first
#     cache_key = (
#         f"{eval_name}:{json.dumps(eval_spec.get('task_params', {}), sort_keys=True)}"
#     )
#     if cache_key in _task_cache:
#         logger.info(f"Using cached task for {eval_name}")
#         return _task_cache[cache_key]

#     try:
#         # Parse eval_name to extract module path and optional function name
#         if ":" in eval_name:
#             # Explicit function name: "custom_evals.my_eval:my_task_fn"
#             module_path, function_name = eval_name.split(":", 1)
#         else:
#             module_path = eval_name
#             function_name = None

#         # Determine the full module path
#         if "." in module_path:
#             # Already a full path like "custom_evals.my_eval"
#             full_module_path = module_path
#             # Default function name is the last part of the module path
#             if not function_name:
#                 function_name = module_path.split(".")[-1]
#         else:
#             # Simple name like "mbpp" → assume inspect_evals
#             full_module_path = f"inspect_evals.{module_path}"
#             if not function_name:
#                 function_name = module_path

#         logger.info(f"Attempting to import: {full_module_path}")

#         # Import the eval module
#         eval_module = import_module(full_module_path)

#         # Get the task function
#         if not hasattr(eval_module, function_name):
#             raise AttributeError(
#                 f"Module '{full_module_path}' does not have function '{function_name}'. "
#                 f"Available: {dir(eval_module)}"
#             )

#         task_fn = getattr(eval_module, function_name)

#         # Instantiate the task with custom parameters
#         task_params = eval_spec.get("task_params", {})
#         logger.info(f"Loading eval: {eval_name} with params: {task_params}")
#         task = task_fn(**task_params)

#         # Cache the task
#         _task_cache[cache_key] = task

#         return task

#     except ImportError as e:
#         raise ValueError(
#             f"Could not import eval '{eval_name}'. "
#             f"For custom evals, ensure the module is in /app/custom_evals/ and accessible. "
#             f"Error: {e}"
#         )
#     except AttributeError as e:
#         raise ValueError(f"Eval loading error: {e}")
#     except Exception as e:
#         raise ValueError(f"Unexpected error loading eval '{eval_name}': {e}")


# def create_task_state_from_sample(
#     sample: Sample, model_name: str = "custom_agent"
# ) -> TaskState:
#     """
#     Create an inspect_ai TaskState from a Sample and solver output.

#     Args:
#         sample: The Sample being processed
#         model_name: Name to use for the model in the task state

#     Returns:
#         TaskState: Populated TaskState for scoring
#     """
#     from inspect_ai.solver import TaskState
#     from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ModelOutput

#     # Create message history
#     messages = [ChatMessageUser(content=str(sample.input))]

#     # Create the model output
#     output = ModelOutput(model=model_name, stop_reason="stop")

#     # Create TaskState
#     state = TaskState(
#         sample_id=sample.id,
#         epoch=0,
#         input=str(sample.input),
#         messages=messages,
#         output=output,
#         metadata=sample.metadata or {},
#     )

#     return state


def is_pid_running(pid):
    if pid is None:
        return False
    return psutil.pid_exists(pid)


def get_lock_data():
    """Get lock data from lock file. Returns dict with status info or None if no lock."""
    try:
        with open(LOCK_FILE_PATH, "r") as f:
            content = f.read().strip()
            # Try to parse as JSON first (new format)
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback: old format was just PID
                return {"status": "running", "pid": int(content)}
    except (IOError, ValueError):
        return None


def write_lock_data(data):
    """Write lock data to lock file."""
    with open(LOCK_FILE_PATH, "w") as f:
        json.dump(data, f)


def get_process_status():
    """Internal function to check process status and update completion status."""
    global _process

    lock_data = get_lock_data()
    if lock_data is None:
        return {"status": "not_running"}

    # If status is already completed, crashed, or stopped, return it
    if lock_data.get("status") in ["completed", "crashed", "stopped"]:
        return lock_data

    # If status is "stopping", check if process actually stopped or timed out
    if lock_data.get("status") == "stopping":
        pid = lock_data.get("pid")
        stop_requested_at = lock_data.get("stop_requested_at")

        if pid and not is_pid_running(pid):
            # Process actually stopped, update status
            status_data = {
                "status": "stopped",
                "message": "Process was manually stopped. It can be resumed.",
                "return_code": -1,
            }
            write_lock_data(status_data)
            return status_data
        elif stop_requested_at:
            # Check if stopping has timed out (15 seconds)
            try:
                from datetime import datetime

                stop_time = datetime.fromisoformat(stop_requested_at)
                elapsed = (datetime.now() - stop_time).total_seconds()

                if elapsed > 15:
                    # Stopping has timed out, mark as crashed
                    status_data = {
                        "status": "crashed",
                        "message": f"Process failed to stop after {elapsed:.1f} seconds and may be stuck.",
                        "return_code": -1,
                        "stop_timeout": True,
                    }
                    write_lock_data(status_data)
                    return status_data
            except (ValueError, TypeError):
                # Invalid timestamp, continue with stopping status
                pass

        # Still in stopping state
        return lock_data

    # Check if process is still running
    pid = lock_data.get("pid")
    if pid and is_pid_running(pid):
        return {"status": "running", "pid": pid, "log_path": LOG_FILE_PATH}

    # Process has stopped, check completion status
    if _process is not None:
        return_code = _process.poll()
        if return_code is not None:
            if return_code == 0:
                # Read completion message from log file
                completion_message = "Process completed successfully"
                try:
                    with open(LOG_FILE_PATH, "r") as f:
                        log_content = f.read()
                        # Extract last few lines or look for completion markers
                        lines = log_content.strip().split("\n")
                        if lines:
                            completion_message = (
                                lines[-1] if lines[-1] else completion_message
                            )
                except Exception:
                    pass

                status_data = {
                    "status": "completed",
                    "message": f"completed. {completion_message}",
                    "return_code": return_code,
                }
            else:
                status_data = {
                    "status": "crashed",
                    "message": f"Process crashed with return code {return_code}",
                    "return_code": return_code,
                }

            write_lock_data(status_data)
            return status_data

    # Fallback: process stopped but we don't have return code info
    status_data = {
        "status": "crashed",
        "message": f"Process with PID {pid} is no longer running but completion status unknown.",
    }
    write_lock_data(status_data)
    return status_data
