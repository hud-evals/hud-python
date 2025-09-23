"""Minimal FastAPI environment server (HTTP-based)."""

from fastapi import FastAPI

import logging
import sys
import traceback

from .utils import run_uv_command

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)

app = FastAPI(title="Blank Environment API")

_count = 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    """Setup and/or reset the environment.
    This is where we'd do a check for extra installation requirements
    of a specific inspect eval, and satisfy those. e.g. sweval"""
    try:
        stdout, stderr = run_uv_command(["sync"])

        return {"ok": True, "stdout": stdout, "stderr": stderr}
    except Exception as e:
        return {"ok": False, "error": e, "traceback": traceback.format_exc()}


@app.post("/run")
def run():
    try:
        stdout, stderr = run_uv_command(["sync"])
        return {"ok": True, "stdout": stdout, "stderr": stderr}
    except Exception as e:
        return {"ok": False, "error": e, "traceback": traceback.format_exc()}


@app.get("/state")
def state():
    return {"count": _count}
