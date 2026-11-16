"""How a Harbor verifier is run and read, and its artifacts staged, wherever that runs."""

from __future__ import annotations

import fnmatch
import json
import math
import re
import shlex
import shutil
from pathlib import Path, PurePath, PurePosixPath
from typing import TYPE_CHECKING, Any, TypeVar

from hud.runtime.core import EnvRef
from hud.types import EvaluationResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hud.utils.process import ProcessResult

    from .config import Artifact

VERIFIER_LOGS = PurePosixPath("/logs/verifier")
_P = TypeVar("_P", bound=PurePath)
ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def env_refs(env: Mapping[str, str]) -> dict[str, str | EnvRef]:
    """``env`` with each ``${NAME}`` or ``${NAME:-default}`` value as a reference to it."""
    return {
        key: value
        if (match := ENV_TEMPLATE.fullmatch(value)) is None
        else EnvRef(name=match.group(1), default=match.group(2))
        for key, value in env.items()
    }


def verifier_command(script: str, path: str) -> list[str]:
    """The command that runs the test script at ``path``, by its shebang or with ``sh``."""
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#!"):
            return [*shlex.split(stripped[2:]), path]
        if stripped and not stripped.startswith("#"):
            break
    return ["/bin/sh", path]


def reward(reward_json: str | None, reward_txt: str | None) -> tuple[float | None, dict[str, Any]]:
    """The reward a verifier wrote to ``reward.json`` or else ``reward.txt``, with details."""
    if reward_json is not None:
        try:
            data = json.loads(reward_json)
        except json.JSONDecodeError:
            return None, {"reward_parse_error": "reward.json is not valid JSON"}
        candidates = [data]
        if isinstance(data, dict):
            candidates.extend((data.get("reward"), data.get("score")))
        for value in candidates:
            if isinstance(value, int | float) and not isinstance(value, bool):
                score = float(value)
                if math.isfinite(score):
                    return score, {"reward_file": str(VERIFIER_LOGS / "reward.json")}
        return None, {"reward_parse_error": "reward.json has no numeric reward"}
    if reward_txt is not None:
        text = reward_txt.strip()
        try:
            score = float(text)
        except ValueError:
            score = math.nan
        if math.isfinite(score):
            return score, {"reward_file": str(VERIFIER_LOGS / "reward.txt")}
        return None, {"reward_parse_error": text}
    return None, {}


def evaluation(
    execution: ProcessResult,
    timeout_sec: float,
    reward_json: str | None,
    reward_txt: str | None,
) -> EvaluationResult:
    """The grade of one verifier run, from its process and the reward files it left."""
    info: dict[str, Any] = {
        "exit_code": execution.returncode,
        "stdout": execution.stdout.decode("utf-8", "replace"),
        "stderr": execution.stderr.decode("utf-8", "replace"),
    }
    if execution.timed_out:
        info["verifier_timeout_sec"] = timeout_sec
        return EvaluationResult(
            isError=True,
            content=f"Harbor verifier timed out after {timeout_sec:.0f}s",
            info=info,
        )
    score, reward_info = reward(reward_json, reward_txt)
    info.update(reward_info)
    if score is None:
        return EvaluationResult(
            isError=True,
            content="Harbor verifier did not write a numeric reward",
            info=info,
        )
    return EvaluationResult(reward=score, info=info)


def artifact_path(artifact: Artifact, artifacts: _P) -> _P:
    relative = artifact.destination or artifact.source.lstrip("/").rstrip("/")
    return artifacts / relative


def exclude_artifact_paths(root: Path, patterns: list[str]) -> None:
    if not patterns or not root.is_dir() or root.is_symlink():
        return
    for entry in sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        relative = entry.relative_to(root).as_posix()
        parts = Path(relative).parts
        candidates = (
            relative,
            f"./{relative}",
            *("/".join(parts[index:]) for index in range(1, len(parts))),
        )
        if not any(
            fnmatch.fnmatchcase(candidate, pattern)
            for candidate in candidates
            for pattern in patterns
        ):
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
