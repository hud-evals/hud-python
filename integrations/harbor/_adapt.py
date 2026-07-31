"""Build adapted Harbor images."""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from hud.utils.docker import docker as _docker

from ._load import HUD_ROOT, HarborTask, _load, _task_groups, hash_directory

if TYPE_CHECKING:
    from hud.eval import Taskset

LOGGER = logging.getLogger(__name__)

_DOCKER_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DOCKER_USER = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*")
SERVING_PYTHON = "3.12"

_CONTEXT_IGNORE_NAMES = (
    "__pycache__",
    "*.pyc",
    ".git",
    ".venv",
    "venv",
    "*.egg-info",
    ".pytest_cache",
)
_CONTEXT_IGNORE = shutil.ignore_patterns(*_CONTEXT_IGNORE_NAMES)

_INSTALL_SH = """\
#!/bin/sh
set -eu
export UV_INSTALL_DIR=__HUD_ROOT__/bin \
       UV_PYTHON_INSTALL_DIR=__HUD_ROOT__/python \
       UV_PYTHON_BIN_DIR=__HUD_ROOT__/bin \
       UV_NO_CACHE=1 \
       XDG_CONFIG_HOME=__HUD_ROOT__/config
export PATH="__HUD_ROOT__/bin:$PATH"
command -v uv >/dev/null 2>&1 || {
  { command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; } \
    || { apt-get update -qq && apt-get install -y -qq curl ca-certificates; } \
    || apk add --no-cache curl ca-certificates
  { command -v curl >/dev/null 2>&1 && curl -LsSf https://astral.sh/uv/install.sh | sh; } \
    || { command -v wget >/dev/null 2>&1 && wget -qO- https://astral.sh/uv/install.sh | sh; } \
    || pip install -q -U uv
}
command -v bwrap >/dev/null 2>&1 \
  || { apt-get update -qq && apt-get install -y -qq bubblewrap; } \
  || apk add --no-cache bubblewrap \
  || echo "warning: bubblewrap unavailable; tasks declaring isolation will refuse to serve"
uv python install __PYTHON__
uv venv __HUD_ROOT__/venv --python __PYTHON__
uv pip install --python __HUD_ROOT__/venv/bin/python __HUD_REQUIREMENT__

apt_pkgs=""
apk_pkgs=""
for spec in \
  "python3|python3 python3-venv|python3" \
  "pip3|python3-pip|py3-pip" \
  "git|git|git" \
  "curl|curl ca-certificates|curl ca-certificates"
do
  if command -v "${spec%%|*}" >/dev/null 2>&1; then continue; fi
  rest=${spec#*|}
  apt_pkgs="$apt_pkgs ${rest%%|*}"
  apk_pkgs="$apk_pkgs ${rest##*|}"
done
if [ -n "$apt_pkgs" ]; then
  { apt-get update -qq && apt-get install -y -qq $apt_pkgs; } \
    || apk add --no-cache $apk_pkgs \
    || echo "warning: could not provision the agent toolchain:$apt_pkgs"
fi
"""

_LAYER = """

# HUD adaptation layer
USER root
COPY _hud __HUD_ROOT__
RUN sh __HUD_ROOT__/install.sh
COPY _hud_harbor __HUD_ROOT__/venv/lib/python__PYTHON__/site-packages/harbor
ENV HUD_SKIP_VERSION_CHECK=1
EXPOSE 8765
__DECLARED__ENTRYPOINT []
CMD ["__HUD_ROOT__/venv/bin/hud", "serve", \
     "harbor:environment", "--arg", "ref=__HUD_ROOT__/tasks", \
     "--arg", "name=__ENV_NAME__", "--host", "0.0.0.0", "--port", "8765"]
"""


def _validated_user(declared: str | int | None, source_user: str | None) -> str | None:
    """Validate a phase identity before it reaches a generated directive."""
    user = str(declared) if declared is not None else source_user
    if user is None:
        return None
    if not all(_DOCKER_USER.fullmatch(part) for part in user.split(":", 1)):
        raise ValueError(f"declared user {user!r} is not a usable user[:group]")
    return user


def _declared_directives(task: HarborTask, source_user: str | None) -> str:
    """Render the task's container-wide environment directives."""
    lines: list[str] = []
    for key, value in sorted(task.config.environment.env.items()):
        if not _DOCKER_ENV_KEY.fullmatch(key):
            raise ValueError(f"environment.env key {key!r} is not a usable variable name")
        literal = json.dumps(value).replace("$", "\\$")
        lines.append(f"ENV {key}={literal}")
    if task.config.environment.workdir:
        lines.append(f"WORKDIR {json.dumps(task.config.environment.workdir)}")
    for role in ("agent", "verifier"):
        _validated_user(task.config.phase_user(role), source_user)
    return "".join(f"{line}\n" for line in lines)


async def adapt(
    path: str | Path,
    *,
    push: str | None = None,
    hud_requirement: str = "hud",
) -> Taskset:
    """Build adapted images and return the runnable Harbor taskset."""
    root = Path(path).resolve()
    out_root = root / ".hud-adapt"
    images: dict[str, str] = {}
    for env_name, group in _task_groups(root):
        context = _write_context(out_root / env_name, env_name, group, hud_requirement)
        content = hash_directory(context)
        ref = f"{push}/{env_name}:{content}" if push else f"hud-harbor-adapted:{env_name}-{content}"
        deadlines = [
            timeout for task in group if (timeout := task.config.environment.build_timeout_sec)
        ]
        await _docker(
            "build",
            "--tag",
            ref,
            str(context),
            deadline=max(deadlines) if deadlines else None,
        )
        if push:
            await _docker("push", ref)
        images[env_name] = ref
    if images:
        LOGGER.info("adapted %d image(s)", len(images))
    return _load(root, images=images)


def _write_context(
    context: Path, env_name: str, group: list[HarborTask], hud_requirement: str
) -> Path:
    """Write one group's environment and HUD adaptation layer."""
    if context.exists():
        shutil.rmtree(context)
    source = group[0]
    env_dir = source.path / "environment"
    if not (env_dir / "Dockerfile").is_file():
        raise FileNotFoundError(f"group {env_name!r} has no environment/Dockerfile")
    _copy_task_content(env_dir, context)
    dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")

    multi_step = [task.path.name for task in group if not (task.path / "instruction.md").is_file()]
    if multi_step:
        raise NotImplementedError(
            "multi-step Harbor tasks (no root instruction.md) cannot be adapted yet: "
            + ", ".join(sorted(multi_step)[:5])
        )
    for task in group:
        if reasons := task.unsupported_features():
            raise NotImplementedError(
                f"Harbor task {task.path.name!r} declares behaviour this integration "
                f"cannot reproduce: {'; '.join(reasons)}"
            )

    dockerignore = context / ".dockerignore"
    if dockerignore.is_file():
        _write(
            dockerignore,
            dockerignore.read_text("utf-8") + "\n!_hud\n!_hud/**\n!_hud_harbor\n!_hud_harbor/**\n",
        )

    hud_dir = context / "_hud"
    hud_dir.mkdir(parents=True)
    requirement = hud_requirement
    wheel = Path(hud_requirement)
    if wheel.suffix == ".whl" and wheel.is_file():
        shutil.copy2(wheel, hud_dir / wheel.name)
        requirement = str(HUD_ROOT / wheel.name)
    shutil.copytree(
        Path(__file__).parent,
        context / "_hud_harbor",
        ignore=shutil.ignore_patterns("tests", ".hud-adapt", *_CONTEXT_IGNORE_NAMES),
    )
    for task in group:
        target = hud_dir / "tasks" / task.path.name
        target.mkdir(parents=True)
        for entry in ("instruction.md", "task.toml"):
            _copy_task_content(task.path / entry, target / entry)
        _copy_task_content(task.path / "tests", target / "tests")
    _write(
        hud_dir / "install.sh",
        _INSTALL_SH.replace("__HUD_REQUIREMENT__", shlex.quote(requirement))
        .replace("__PYTHON__", SERVING_PYTHON)
        .replace("__HUD_ROOT__", str(HUD_ROOT)),
    )
    if source.final_stage.user is not None:
        _write(hud_dir / "image-user", f"{source.final_stage.user}\n")
    layer = (
        _LAYER.replace("__HUD_ROOT__", str(HUD_ROOT))
        .replace("__ENV_NAME__", env_name)
        .replace("__PYTHON__", SERVING_PYTHON)
        .replace(
            "__DECLARED__",
            _declared_directives(source, source.final_stage.user),
        )
    )
    _write(context / "Dockerfile", dockerfile + layer)
    return context


def _copy_task_content(source: Path, destination: Path) -> None:
    """Copy task content without following symlinks."""
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True, ignore=_CONTEXT_IGNORE)
    else:
        shutil.copy2(source, destination, follow_symlinks=False)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
