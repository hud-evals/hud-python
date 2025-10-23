"""Controller tools for Inspect AI Sandbox

Provides MCP tools that agents can use to interact with the sandbox environment.
Also handles evaluation scoring using inspect_ai scorers.
"""

import json
import httpx
import logging
import sys
import os
from typing import Any

from controller import mcp, http_client
from hud.tools.types import EvaluationResult

# Import inspect_ai components for scoring
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser, ModelOutput

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Store task information for evaluation
_current_task: Task | None = None
_eval_name: str | None = None


@mcp.tool()
async def setup(eval_name: str, sample_id: str, task_data: dict | None = None) -> str:
    """
    Initialize sandbox environment for a specific sample.

    Args:
        eval_name: Name of the eval (e.g., "mbpp")
        sample_id: ID of the sample being evaluated
        task_data: Optional serialized task data (contains scorer, etc.)
    """
    global _current_task, _eval_name

    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    # Initialize sandbox environment
    resp = await http_client.post(
        "/reset", json={"eval_name": eval_name, "sample_id": sample_id}
    )

    _eval_name = eval_name

    result = resp.json()
    return json.dumps(
        {
            "status": "ready",
            "eval_name": eval_name,
            "sample_id": sample_id,
            "sandbox_dir": result.get("sandbox_dir"),
        }
    )


@mcp.tool()
async def exec(cmd: list[str], timeout: int = 30, cwd: str | None = None) -> str:
    """
    Execute a command in the sandbox.

    Args:
        cmd: Command to execute as a list (e.g., ["python", "-c", "print('hello')"])
        timeout: Timeout in seconds (default: 30)
        cwd: Working directory relative to sandbox root (optional)

    Returns:
        JSON string with execution results (stdout, stderr, returncode, success)
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    resp = await http_client.post(
        "/exec", json={"cmd": cmd, "timeout": timeout, "cwd": cwd}
    )

    result = resp.json()

    # Format output for agent
    output_parts = []
    if result.get("stdout"):
        output_parts.append(f"STDOUT:\n{result['stdout']}")
    if result.get("stderr"):
        output_parts.append(f"STDERR:\n{result['stderr']}")

    output_parts.append(f"Exit code: {result['returncode']}")

    return "\n\n".join(output_parts)


@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """
    Write a file in the sandbox.

    Args:
        path: Path relative to sandbox root (e.g., "solution.py")
        content: File content to write

    Returns:
        Success message with file path
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    resp = await http_client.post(
        "/write_file", json={"path": path, "content": content}
    )

    result = resp.json()
    return f"File written successfully: {result.get('path')}"


@mcp.tool()
async def read_file(path: str) -> str:
    """
    Read a file from the sandbox.

    Args:
        path: Path relative to sandbox root (e.g., "output.txt")

    Returns:
        File content
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    try:
        resp = await http_client.post("/read_file", json={"path": path})
        result = resp.json()
        return result.get("content", "")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Error: File not found: {path}"
        raise


@mcp.tool()
async def list_files(path: str = ".") -> str:
    """
    List files in a directory within the sandbox.

    Args:
        path: Directory path relative to sandbox root (default: ".")

    Returns:
        Formatted list of files and directories
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    try:
        resp = await http_client.post("/list_files", json={"path": path})
        result = resp.json()

        entries = result.get("entries", [])
        if not entries:
            return f"Directory is empty: {path}"

        lines = [f"Contents of {path}:"]
        for entry in entries:
            type_str = "DIR " if entry["is_dir"] else "FILE"
            size_str = f" ({entry['size']} bytes)" if entry.get("size") else ""
            lines.append(f"  {type_str} {entry['name']}{size_str}")

        return "\n".join(lines)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Error: Directory not found: {path}"
        raise


@mcp.tool()
async def git_clone(url: str, path: str = ".") -> str:
    """
    Clone a git repository in the sandbox.

    Args:
        url: Git repository URL to clone
        path: Destination path relative to sandbox root (default: ".")

    Returns:
        Success message with cloned repository path
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    try:
        resp = await http_client.post(
            "/exec", json={"cmd": ["git", "clone", url, path], "timeout": 300}
        )
        result = resp.json()

        if result["returncode"] == 0:
            return f"Repository cloned successfully to {path}"
        else:
            return f"Error cloning repository: {result.get('stderr', 'Unknown error')}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error during git clone: {e}"


@mcp.tool()
async def git_diff(path: str = ".", staged: bool = False) -> str:
    """
    Show git diff in the sandbox.

    Args:
        path: Path relative to sandbox root (default: ".")
        staged: Show staged changes (--cached) if True, otherwise show unstaged changes

    Returns:
        Git diff output
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    cmd = ["git", "-C", path, "diff"]
    if staged:
        cmd.append("--cached")

    try:
        resp = await http_client.post("/exec", json={"cmd": cmd, "timeout": 30})
        result = resp.json()

        if result["returncode"] == 0:
            return result.get("stdout", "(no changes)")
        else:
            return f"Error running git diff: {result.get('stderr', 'Unknown error')}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error during git diff: {e}"


@mcp.tool()
async def git_commit(message: str, path: str = ".", add_all: bool = True) -> str:
    """
    Commit changes in the sandbox repository.

    Args:
        message: Commit message
        path: Path to git repository relative to sandbox root (default: ".")
        add_all: Stage all changes before committing (default: True)

    Returns:
        Success message with commit info
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    try:
        # Stage changes if requested
        if add_all:
            resp = await http_client.post(
                "/exec", json={"cmd": ["git", "-C", path, "add", "-A"], "timeout": 30}
            )
            result = resp.json()
            if result["returncode"] != 0:
                return f"Error staging changes: {result.get('stderr', 'Unknown error')}"

        # Commit
        resp = await http_client.post(
            "/exec",
            json={"cmd": ["git", "-C", path, "commit", "-m", message], "timeout": 30},
        )
        result = resp.json()

        if result["returncode"] == 0:
            return f"Changes committed successfully: {result.get('stdout', '')}"
        else:
            stderr = result.get("stderr", "")
            # Check if there's nothing to commit
            if (
                "nothing to commit" in stderr.lower()
                or "no changes added to commit" in stderr.lower()
            ):
                return "No changes to commit"
            return f"Error committing changes: {stderr}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error during git commit: {e}"


@mcp.tool()
async def evaluate(
    sample: dict, solution_file: str = "solution.py", scorer_model: str | None = None
) -> EvaluationResult:
    """
    Evaluate the agent's solution against the sample's expected target.

    This uses the inspect_ai Task's scorer to evaluate the solution.
    For code evals, the agent should write its solution to a file (default: solution.py).

    Args:
        sample: The original sample data (from task metadata)
        solution_file: Path to file containing agent's solution (default: "solution.py")
        scorer_model: Model to use for LLM-as-a-judge scoring (e.g., "openai/gpt-4o")

    Returns:
        EvaluationResult with reward and done flag
    """
    global _current_task, _eval_name

    # Log scorer model if provided
    if scorer_model:
        logger.info(f"Using scorer model: {scorer_model}")

    try:
        # Get agent's output from the solution file
        agent_output = None
        actual_file = solution_file

        try:
            resp = await http_client.post("/read_file", json={"path": solution_file})
            agent_output = resp.json().get("content", "")
        except Exception as e:
            logger.warning(f"Could not read solution file {solution_file}: {e}")

            # Try to find any .py file in the sandbox
            try:
                resp = await http_client.post("/list_files", json={"path": "."})
                files = resp.json().get("entries", [])
                py_files = [f for f in files if f["name"].endswith(".py")]

                if py_files:
                    # Try to read the first .py file
                    actual_file = py_files[0]["name"]
                    logger.info(
                        f"Found {actual_file}, using it instead of {solution_file}"
                    )
                    resp = await http_client.post(
                        "/read_file", json={"path": actual_file}
                    )
                    agent_output = resp.json().get("content", "")
                else:
                    file_list = ", ".join([f["name"] for f in files])
                    return EvaluationResult(
                        reward=0.0,
                        done=True,
                        isError=True,
                        content=f"No Python solution file found. Expected '{solution_file}'. "
                        f"Files in sandbox: {file_list}. "
                        f"Agent should write solution to {solution_file}.",
                    )
            except Exception as list_err:
                logger.error(f"Error listing files: {list_err}")
                return EvaluationResult(
                    reward=0.0,
                    done=True,
                    isError=True,
                    content=f"Could not read solution file '{solution_file}' or list sandbox files.",
                )

        if not agent_output:
            return EvaluationResult(
                reward=0.0,
                done=True,
                isError=True,
                content=f"Solution file {actual_file} is empty.",
            )

        # Load the scorer if not already loaded
        scorer = None
        if _eval_name:
            try:
                # Only load the scorer, not the entire task/dataset
                from inspect_loader import load_scorer_only

                scorer = load_scorer_only(_eval_name)
                logger.info(f"Loaded scorer for {_eval_name}")
            except Exception as e:
                logger.warning(f"Could not load scorer for {_eval_name}: {e}")

        if scorer is None:
            # No scorer available, do simple string matching
            logger.warning("No scorer available, using simple string matching")
            target = sample.get("target")
            matches = str(target).strip() in agent_output.strip()

            return EvaluationResult(
                reward=1.0 if matches else 0.0,
                done=True,
                isError=False,
                content=f"Simple match: {'PASS' if matches else 'FAIL'}. Expected: {target}",
            )

        # Create inspect_ai Sample object
        inspect_sample = Sample(
            id=sample.get("id"),
            input=sample.get("input"),
            target=sample.get("target"),
            metadata=sample.get("metadata", {}),
            sandbox=sample.get("sandbox"),
        )

        # Create TaskState with agent output
        # Note: This is a simplified TaskState - in production you'd want to
        # capture the full conversation history
        task_state = TaskState(
            model="hud/agent",
            sample_id=str(inspect_sample.id),
            epoch=1,
            input=[ChatMessageUser(content=str(inspect_sample.input))],
            messages=[
                ChatMessageUser(content=str(inspect_sample.input)),
            ],
            output=ModelOutput.from_content(
                model="hud/agent",
                content=agent_output,
            ),
            completed=True,
        )

        # Use the scorer we loaded earlier
        if isinstance(scorer, list):
            scorer = scorer[0]  # Use first scorer if multiple

        # Score the output
        score = await scorer(task_state, inspect_sample.target)

        # Convert to EvaluationResult
        reward = 1.0 if score.value == "C" else 0.0  # "C" = CORRECT

        return EvaluationResult(
            reward=reward,
            done=True,
            isError=False,
            content=f"Score: {score.value}\nExplanation: {score.explanation}",
        )

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return EvaluationResult(
            reward=0.0,
            done=True,
            isError=True,
            content=f"Evaluation error: {str(e)}",
        )


@mcp.tool()
async def auto_evaluate(
    judge_prompt: str,
    agent_output: str,
    expected_output: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> EvaluationResult:
    """
    Evaluate agent output using an LLM-as-a-judge.

    Args:
        judge_prompt: The system prompt for the judge model
        agent_output: The agent's output to evaluate
        expected_output: Optional expected/target output for comparison
        model: OpenAI model to use (default: "gpt-4o")
        temperature: Temperature for the judge model (default: 0.0)
        max_tokens: Max tokens for judge response (default: 500)

    Returns:
        EvaluationResult with reward based on judge's decision
    """
    try:
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            logger.error("OPENAI_API_KEY environment variable not set")
            return EvaluationResult(
                reward=0.0,
                done=False,
                isError=True,
                content="OPENAI_API_KEY environment variable not set",
            )

        logger.info(f"Creating OpenAI client for LLM-as-judge evaluation...")

        # Import openai here to avoid issues if not installed
        import openai

        # Create OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client created successfully")

        # Build user prompt
        user_content = f"Agent Output:\n{agent_output}"
        if expected_output:
            user_content += f"\n\nExpected Output:\n{expected_output}"

        messages = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": user_content},
        ]

        # Call judge model
        logger.info(f"Calling {model} for evaluation...")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        result_text = response.choices[0].message.content.strip()
        logger.info(f"Judge response: {result_text[:200]}...")

        # Parse result - look for common success indicators
        result_lower = result_text.lower()
        success = any(
            indicator in result_lower
            for indicator in ["success", "correct", "pass", "yes"]
        )

        return EvaluationResult(
            reward=1.0 if success else 0.0,
            done=True,
            isError=False,
            content=result_text,
        )

    except Exception as e:
        logger.error(f"LLM-as-judge evaluation failed: {e}", exc_info=True)
        return EvaluationResult(
            reward=0.0,
            done=True,
            isError=True,
            content=f"Judge evaluation error: {str(e)}",
        )
