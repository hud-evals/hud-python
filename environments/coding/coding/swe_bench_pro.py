"""The SWE-bench Pro task flavor: dataset-row parsing and the official grading pipeline.

An instance directory (baked into the instance image by ``swe_tasks.py``)
holds the dataset row (``instance.json``) plus the official per-instance
``run_script.sh`` and ``parser.py`` from scaleapi/SWE-bench_Pro-os. Grading
replays the official evaluator's entryscript on the repo-lifecycle primitives
in :mod:`coding.repo`: reset to the pre-agent snapshot, apply the (agent or
golden) patch, check out the hidden test files, run the script, parse, and
resolve iff every ``fail_to_pass`` and ``pass_to_pass`` test passes.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

from hud.graders import EvaluationResult, SubScore

from . import repo as repo_lib


def load_instance(instance_dir: Path) -> dict[str, Any]:
    return json.loads((instance_dir / "instance.json").read_text("utf-8"))


def str_list(field: str) -> list[str]:
    """Parse the dataset's list-shaped string fields.

    The rows store lists as Python reprs (single quotes; the official
    evaluator uses ``eval``), so ``ast.literal_eval`` is the safe equivalent.
    """
    value = ast.literal_eval(field)
    if not isinstance(value, list):
        raise ValueError(f"expected a list, got {type(value).__name__}")
    return [str(item) for item in value]


def strip_binary_hunks(patch: str) -> str:
    """Drop binary diff sections, exactly like the official evaluator."""
    if not patch:
        return patch
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept = [
        section
        for section in sections
        if section.strip()
        and not re.search(r"^Binary files .* differ$", section, re.MULTILINE)
        and not re.search(r"^GIT binary patch$", section, re.MULTILINE)
    ]
    return "".join(kept)


def score(instance: dict[str, Any], tests: list[dict[str, Any]]) -> EvaluationResult:
    """The official resolution criterion: every required test passes."""
    passed = {t["name"] for t in tests if t.get("status") == "PASSED"}
    fail_to_pass = set(str_list(instance["fail_to_pass"]))
    pass_to_pass = set(str_list(instance["pass_to_pass"]))
    required = fail_to_pass | pass_to_pass
    resolved = required <= passed

    def fraction(subset: set[str]) -> float:
        return len(subset & passed) / len(subset) if subset else 1.0

    missing = sorted(required - passed)
    return EvaluationResult(
        reward=1.0 if resolved else 0.0,
        content="resolved" if resolved else f"unresolved: {len(missing)} required test(s) not passing",
        subscores=[
            SubScore(name="resolved", value=1.0 if resolved else 0.0, weight=1.0),
            SubScore(name="fail_to_pass", value=fraction(fail_to_pass), weight=0.0),
            SubScore(name="pass_to_pass", value=fraction(pass_to_pass), weight=0.0),
        ],
        info={"tests_reported": len(tests), "missing": missing[:20]},
    )


def build_prompt(instance: dict[str, Any], repo_dir: Path) -> str:
    sections = [
        f"You are working in {repo_dir}, a checkout of the {instance['repo']} repository. "
        "Resolve the issue described below by modifying the code. Hidden tests grade your "
        "work when you finish; do not modify existing tests.",
        instance["problem_statement"],
    ]
    if instance.get("requirements"):
        sections.append("## Requirements\n\n" + instance["requirements"])
    if instance.get("interface"):
        sections.append("## Interface\n\n" + instance["interface"])
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


async def grade(
    instance: dict[str, Any],
    instance_dir: Path,
    repo_dir: Path,
    vault_dir: Path,
    logs_dir: Path,
    *,
    validate_mode: str | None = None,
    tests_timeout: float = 3600.0,
) -> EvaluationResult:
    """Replay the official evaluator's entryscript in place.

    ``validate_mode="golden"`` grades the dataset's golden patch instead of
    the agent's diff (the official gold-patch sanity check). The hidden test
    files (the last line of ``before_repo_set_cmd``) are checked out *after*
    the patch, so patch edits to them never survive.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_commit = await repo_lib.restore_history(repo_dir, vault_dir)

    if validate_mode == "golden":
        diff = instance["patch"]
    else:
        diff = await repo_lib.capture_agent_diff(repo_dir, setup_commit)
    diff = strip_binary_hunks(diff)

    await repo_lib.reset_worktree(repo_dir, setup_commit)
    apply_error = await repo_lib.apply_diff(repo_dir, diff, logs_dir / "patch.diff")
    if apply_error is not None:
        return EvaluationResult(reward=0.0, content="patch failed to apply", info={"git_apply": apply_error})

    test_checkout = instance["before_repo_set_cmd"].strip().splitlines()[-1]
    await repo_lib.run("bash", "-c", test_checkout, cwd=repo_dir)

    files_csv = ",".join(str_list(instance["selected_test_files_to_run"]))
    _, out, err = await repo_lib.run(
        "bash",
        str(instance_dir / "run_script.sh"),
        files_csv,
        cwd=repo_dir,
        timeout=tests_timeout,
        check=False,
    )
    (logs_dir / "stdout.log").write_text(out, "utf-8")
    (logs_dir / "stderr.log").write_text(err, "utf-8")
    await repo_lib.run(
        sys.executable,
        str(instance_dir / "parser.py"),
        "stdout.log",
        "stderr.log",
        "output.json",
        cwd=logs_dir,
    )
    tests = json.loads((logs_dir / "output.json").read_text("utf-8"))["tests"]
    return score(instance, tests)
