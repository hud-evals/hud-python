"""Prepare live example tasks and check their traces."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hud.eval import Taskset


def prepare(example: str) -> None:
    task = next(iter(Taskset.from_file("tasks.py")))
    Path(".hud").mkdir(exist_ok=True)
    if example == "argument-hints":
        file_id = os.environ["HUD_CI_DATA_FILE_ID"]
        if not file_id:
            raise ValueError("HUD_CI_DATA_FILE_ID must reference the shared CI attachment")
        task.args["attachments"] = [{"file_id": file_id, "path": "notes.txt"}]
        task.args["prompt"] = "Read files/notes.txt first. " + task.args["prompt"]
    Taskset(tasks=[task]).to_file(".hud/ci-tasks.json")


def check() -> None:
    trace_dir = Path(os.environ["HUD_TELEMETRY_LOCAL_DIR"])
    traces = list(trace_dir.glob("*.jsonl"))
    assert len(traces) == 1, f"Expected one rollout trace, found {len(traces)}"
    steps = [
        span["attributes"]["hud.payload"]
        for line in traces[0].read_text().splitlines()
        if (span := json.loads(line))["attributes"].get("hud.schema") == "hud.step.v1"
    ]
    errors = [
        step["error"]
        for step in steps
        if step["source"] in {"task", "agent", "system"} and step.get("error")
    ]
    assert not errors, errors
    calls = [step["task_call"] for step in steps if step["source"] == "task"]
    assert [call["phase"] for call in calls] == ["setup", "evaluate"], calls
    assert calls[0]["result"].get("prompt"), "Task setup did not return a prompt"
    grade = calls[1]["result"]
    assert not grade.get("isError"), grade
    assert isinstance(grade["score"], int | float) and 0 <= grade["score"] <= 1, grade
    task = next(iter(Taskset.from_file(".hud/ci-tasks.json")))
    if attachments := task.args.get("attachments"):
        assert calls[0]["result"]["data_files"] == [
            {"path": "files/notes.txt", "file_id": attachments[0]["file_id"]}
        ]

    # Unset local export so `hud trace` must read back from the platform.
    env = {key: value for key, value in os.environ.items() if key != "HUD_TELEMETRY_LOCAL_DIR"}
    for attempt in range(12):
        result = subprocess.run(
            [str(Path(sys.executable).with_name("hud")), "trace", traces[0].stem, "--json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        events = json.loads(result.stdout) if result.returncode == 0 else []
        if any(event["kind"] == "agent_message" for event in events):
            (trace_dir / "remote-events.json").write_text(result.stdout)
            print(f"Hosted lifecycle verified: {traces[0].stem}, score={grade['score']}")
            return
        if attempt < 11:
            time.sleep(5)
    raise AssertionError(
        f"No persisted agent events after 12 attempts: {result.stdout}\n{result.stderr}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument("example")
    commands.add_parser("check")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.example)
    elif args.command == "check":
        check()
