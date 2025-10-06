"""Sandbox Environment Server for Inspect AI Evals

This server provides sandbox capabilities (file operations, command execution)
for running inspect_ai evaluations. It does NOT orchestrate the eval - that's
Hud's job. This is purely the sandbox/environment layer.
"""

import logging
import sys
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(title="Inspect AI Sandbox Environment")


# Global sandbox state
_sandbox_initialized = False
_sandbox_dir: Path | None = None
_eval_name: str | None = None
_sample_id: str | None = None


class SetupRequest(BaseModel):
    """Request to initialize sandbox for a specific sample."""

    eval_name: str
    sample_id: str


class ExecRequest(BaseModel):
    """Request to execute a command in the sandbox."""

    cmd: list[str]
    timeout: int = 30
    cwd: str | None = None


class WriteFileRequest(BaseModel):
    """Request to write a file in the sandbox."""

    path: str
    content: str


class ReadFileRequest(BaseModel):
    """Request to read a file from the sandbox."""

    path: str


class ListFilesRequest(BaseModel):
    """Request to list files in a directory."""

    path: str = "."


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "ok": True,
        "content": {
            "initialized": _sandbox_initialized,
            "eval_name": _eval_name,
            "sample_id": _sample_id,
        },
    }


@app.post("/reset")
async def reset(request: SetupRequest):
    """
    Initialize sandbox environment for a specific sample.

    This creates a clean working directory and prepares the sandbox
    for the agent to work in.
    """
    global _sandbox_initialized, _sandbox_dir, _eval_name, _sample_id

    _eval_name = request.eval_name
    _sample_id = request.sample_id

    # Create a temporary working directory for this sample
    # In production, you might want to use a more permanent location
    _sandbox_dir = Path(tempfile.mkdtemp(prefix=f"{_eval_name}_{_sample_id}_"))

    logger.info(
        f"Initialized sandbox for {_eval_name} sample {_sample_id} at {_sandbox_dir}"
    )

    _sandbox_initialized = True

    return {
        "ok": True,
        "sandbox_dir": str(_sandbox_dir),
        "eval_name": _eval_name,
        "sample_id": _sample_id,
    }


@app.post("/exec")
async def exec_command(request: ExecRequest):
    """
    Execute a command in the sandbox.

    This is the primary tool for running code, tests, etc.
    """
    if not _sandbox_initialized:
        raise HTTPException(
            status_code=400, detail="Sandbox not initialized. Call /reset first."
        )

    # Determine working directory
    if request.cwd:
        cwd = _sandbox_dir / request.cwd
    else:
        cwd = _sandbox_dir

    logger.info(f"Executing command: {' '.join(request.cmd)} in {cwd}")

    try:
        result = subprocess.run(
            request.cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=request.timeout,
        )

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {request.timeout} seconds",
        }
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }


@app.post("/write_file")
async def write_file(request: WriteFileRequest):
    """Write a file in the sandbox."""
    if not _sandbox_initialized:
        raise HTTPException(
            status_code=400, detail="Sandbox not initialized. Call /reset first."
        )

    file_path = _sandbox_dir / request.path

    try:
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        file_path.write_text(request.content)

        logger.info(f"Wrote file: {file_path}")

        return {"ok": True, "path": str(file_path)}

    except Exception as e:
        logger.error(f"Error writing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/read_file")
async def read_file(request: ReadFileRequest):
    """Read a file from the sandbox."""
    if not _sandbox_initialized:
        raise HTTPException(
            status_code=400, detail="Sandbox not initialized. Call /reset first."
        )

    file_path = _sandbox_dir / request.path

    try:
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.path}")

        content = file_path.read_text()

        return {"ok": True, "content": content, "path": str(file_path)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/list_files")
async def list_files(request: ListFilesRequest):
    """List files in a directory within the sandbox."""
    if not _sandbox_initialized:
        raise HTTPException(
            status_code=400, detail="Sandbox not initialized. Call /reset first."
        )

    dir_path = _sandbox_dir / request.path

    try:
        if not dir_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Directory not found: {request.path}"
            )

        if not dir_path.is_dir():
            raise HTTPException(
                status_code=400, detail=f"Not a directory: {request.path}"
            )

        # List files and directories
        entries = []
        for entry in dir_path.iterdir():
            entries.append(
                {
                    "name": entry.name,
                    "path": str(entry.relative_to(_sandbox_dir)),
                    "is_file": entry.is_file(),
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
            )

        return {"ok": True, "entries": entries, "path": str(dir_path)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/capabilities")
async def capabilities():
    """
    Return the capabilities of this sandbox.

    This allows Hud to understand what operations are supported.
    """
    return {
        "capabilities": ["exec", "file_ops"],
        "tools": [
            {
                "name": "exec",
                "description": "Execute commands in sandbox",
                "supported": True,
            },
            {
                "name": "write_file",
                "description": "Write files in sandbox",
                "supported": True,
            },
            {
                "name": "read_file",
                "description": "Read files from sandbox",
                "supported": True,
            },
            {
                "name": "list_files",
                "description": "List files in sandbox directory",
                "supported": True,
            },
        ],
        "sandbox_type": "docker",
    }
