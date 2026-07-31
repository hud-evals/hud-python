"""Adapted Harbor images: build them, and serve from inside them.

A Harbor task's environment *is* a container, so its environment constructor
is only meaningful in there. :func:`adapt` packages it: one image per env
group whose CMD serves the HUD control channel from inside, which is the one
assumption every container placement makes — so the same rows run on local
docker, a cloud sandbox, or a platform deploy.

:func:`environment` is what those images serve, by module reference::

    CMD[
        "hud",
        "serve",
        "harbor:environment",
        "--arg",
        "ref=/media/hud/tasks",
        "--arg",
        "name=<env>",
    ]

Everything it needs is container-local: the workspace is the image's working
directory, and grading runs each task's ``tests/test.sh`` in place. The
sandbox exists to keep the graded material — the baked tasks under the
harness's own tree, and the verifier's verdict — outside the agent's
namespace, and to sever the network when a task declares no-network::

    images = await harbor.adapt("./harbor_tasks", push="registry.io/acme")
    taskset = harbor.load("./harbor_tasks", images=images)
    await taskset.run(agent, runtime=DaytonaRuntime())
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import pwd
import re
import shlex
import shutil
from collections.abc import (  # noqa: TC003 - env.template resolves at runtime
    AsyncGenerator,
    Awaitable,
    Callable,
)
from pathlib import Path
from typing import Any

from hud.environment import Environment, Mount
from hud.environment.workspace import Workspace, install_identity_map, usable_bwrap
from hud.eval import DockerRuntime
from hud.utils.docker import docker as _docker
from hud.utils.process import ProcessGroup, create_process_group_exec

from ._load import (
    DEFAULT_VERIFIER_TIMEOUT,
    HUD_ROOT,
    TaskConfig,
    final_stage,
    grouped,
    hash_directory,
    slugify,
    unsupported_features,
    workspace_policy,
)

LOGGER = logging.getLogger(__name__)

#: Harbor's absolute conventions. ``VERIFIER_LOGS`` holds the verdict, so it
#: is masked from agent sessions (see :func:`environment`) and is the only
#: place :func:`read_reward` trusts.
LOGS = Path("/logs")
VERIFIER_LOGS = LOGS / "verifier"
TESTS = Path("/tests")

#: Task config is untrusted: these bound what may reach a generated directive.
_DOCKER_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DOCKER_USER = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*")

#: The interpreter the serving venv is built on. The layer copies this
#: integration into that venv's site-packages by path, so the two must agree.
SERVING_PYTHON = "3.12"

#: Build-context entries never copied into adapted contexts.
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
# Install a self-contained hud venv under __HUD_ROOT__: bootstrap uv (which brings its
# own managed Python), never touching the image's Python or site-packages.
# Needs network and one of: uv / curl / wget / pip (+ apt-get/apk for bare
# images with no downloader).
set -eu
# Everything HUD installs goes under __HUD_ROOT__ — the uv binary, its managed
# interpreter, that interpreter's shims — because that tree is the one the
# workspace keeps out of the graded party's namespace. At uv's defaults this lands
# in the invoking user's home, which is *inside* the task's filesystem: the
# agent then finds a HUD runtime the task never declared, and shims pointing
# into the mask that resolve to nothing. Containing it is what makes the single
# single hiding of that tree sufficient, rather than state to clean up after.
# The cache is the same argument, minus the keeping: nothing reads it after this
# script, so it is never written.
export UV_INSTALL_DIR=__HUD_ROOT__/bin \\
       UV_PYTHON_INSTALL_DIR=__HUD_ROOT__/python \\
       UV_PYTHON_BIN_DIR=__HUD_ROOT__/bin \\
       UV_NO_CACHE=1 \\
       XDG_CONFIG_HOME=__HUD_ROOT__/config
export PATH="__HUD_ROOT__/bin:$PATH"
command -v uv >/dev/null 2>&1 || {
  { command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; } \\
    || { apt-get update -qq && apt-get install -y -qq curl ca-certificates; } \\
    || apk add --no-cache curl ca-certificates
  { command -v curl >/dev/null 2>&1 && curl -LsSf https://astral.sh/uv/install.sh | sh; } \\
    || { command -v wget >/dev/null 2>&1 && wget -qO- https://astral.sh/uv/install.sh | sh; } \\
    || pip install -q -U uv
}
# bubblewrap backs the workspace sandbox, which keeps the baked tasks and the
# verifier's verdict outside the agent's namespace (and severs the network for
# a no-network task). The container must also permit nested namespaces, which
# HUD's placements grant (seccomp/systempaths unconfined); where they do not,
# the env refuses to serve rather than serving forgeable rollouts.
command -v bwrap >/dev/null 2>&1 \
  || { apt-get update -qq && apt-get install -y -qq bubblewrap; } \
  || apk add --no-cache bubblewrap \
  || echo "warning: bubblewrap unavailable; tasks declaring isolation will refuse to serve"
uv python install __PYTHON__
uv venv __HUD_ROOT__/venv --python __PYTHON__
uv pip install --python __HUD_ROOT__/venv/bin/python __HUD_REQUIREMENT__

# ─── the agent's toolchain ───
# Harbor installs the agent *into* the container, and that step provisions what
# the agent needs (BaseInstalledAgent.SYSTEM_PACKAGES: python3, pip, git, curl)
# before the agent phase. HUD's agent is external and installs nothing, so
# without this an adapted image hands the agent a barer machine than Harbor
# would for the same task — a difference in the harness, scored as if it were a
# difference in the model.
#
# Build time, not per rollout: baked into the content-addressed layer once and
# reused by every rollout. Only what the image actually lacks is installed, so
# an image shipping its own Python keeps exactly that Python.
apt_pkgs=""
apk_pkgs=""
for spec in \\
  "python3|python3 python3-venv|python3" \\
  "pip3|python3-pip|py3-pip" \\
  "git|git|git" \\
  "curl|curl ca-certificates|curl ca-certificates"
do
  if command -v "${spec%%|*}" >/dev/null 2>&1; then continue; fi
  rest=${spec#*|}
  apt_pkgs="$apt_pkgs ${rest%%|*}"
  apk_pkgs="$apk_pkgs ${rest##*|}"
done
if [ -n "$apt_pkgs" ]; then
  { apt-get update -qq && apt-get install -y -qq $apt_pkgs; } \\
    || apk add --no-cache $apk_pkgs \\
    || echo "warning: could not provision the agent toolchain:$apt_pkgs"
fi
"""

_LAYER = """

# ─── HUD adaptation layer: serve the control channel from inside ───
# (generated by harbor.adapt)
# Installed as root: the base stage may end on a non-root USER that could not
# write __HUD_ROOT__. The task's own declared user, if any, is restored below.
USER root
COPY _hud __HUD_ROOT__
RUN sh __HUD_ROOT__/install.sh
# This integration ships outside the hud distribution, so it comes from the
# build context rather than the index — which also means the code serving the
# image is the revision that adapted it, not whatever the index resolves to.
COPY _hud_harbor __HUD_ROOT__/venv/lib/python__PYTHON__/site-packages/harbor
# The CLI's update check has no user to prompt here, and its cache lives in the
# invoking user's home — inside the task's filesystem. A container is fresh
# every rollout, so the cache is never warm: left on, it calls PyPI on the
# rollout's critical path and leaves HUD state in the graded filesystem.
ENV HUD_SKIP_VERSION_CHECK=1
EXPOSE 8765
__DECLARED__ENTRYPOINT []
CMD ["__HUD_ROOT__/venv/bin/hud", "serve", \
     "harbor:environment", "--arg", "ref=__HUD_ROOT__/tasks", \
     "--arg", "name=__ENV_NAME__", "--host", "0.0.0.0", "--port", "8765"]
"""


def _validated_user(declared: str | int | None, source_user: str | None) -> str | None:
    """A phase's identity: what it declared, else the image's own ``USER``."""
    user = str(declared) if declared is not None else source_user
    if user is None:
        return None
    # Untrusted input: the task config and the image's own Dockerfile.
    if not all(_DOCKER_USER.fullmatch(part) for part in user.split(":", 1)):
        raise ValueError(f"declared user {user!r} is not a usable user[:group]")
    return user


def phase_uid(task_dir: Path, role: str) -> int | None:
    """The id *role* runs as in this image, or None to run as the harness does.

    Harbor runs the agent and the verifier as the identity the task declares
    for each, and its own harness is not subject to either. Serving from
    inside the image, a ``USER`` directive would demote the harness too —
    leaving it unable to create ``/tests`` at the filesystem root, or its own
    state under ``/hud``. So the identity is applied where the phase runs.

    A name the image does not have is an error rather than a fall back to
    root: running a phase with more privilege than the task granted is the
    thing this exists to prevent.
    """
    user = _validated_user(workspace_policy(task_dir)[f"{role}_user"], _image_user(task_dir))
    if user is None:
        return None
    name = user.split(":", 1)[0]
    if name.isdigit():
        return int(name)
    try:
        return pwd.getpwnam(name).pw_uid
    except KeyError as error:
        raise ValueError(f"declared user {user!r} does not exist in this image") from error


def phase_home(uid: int | None) -> str | None:
    """*uid*'s home, for a phase that is about to drop to it.

    Dropping ids without one leaves the phase pointed at the harness's home,
    which it cannot write — and a verifier that installs its own tooling
    (Harbor's commonly do) fails on that alone.
    """
    if uid is None:
        return None
    with contextlib.suppress(KeyError):
        return pwd.getpwuid(uid).pw_dir
    return None


def _image_user(task_dir: Path) -> str | None:
    """The ``USER`` the adapted image recorded for itself, if any.

    Read from the layer rather than the Dockerfile: by serve time the build
    context is gone, and the image's own declaration still has to be honoured
    for a task whose ``task.toml`` names no user of its own.
    """
    recorded = HUD_ROOT / "image-user"
    with contextlib.suppress(OSError):
        return recorded.read_text(encoding="utf-8").strip() or None
    return None


def _declared_directives(task_dir: Path, source_user: str | None) -> str:
    """The task's declared environment, working dir and user as Dockerfile
    directives.

    Docker applies these to every process in the container — the agent's
    shells, the verifier, the serving process — which is the semantics Harbor
    describes. Expressing them here rather than in the serving code keeps one
    implementation (the container runtime's) instead of a second, partial one.

    Values are JSON-quoted, which is how Dockerfile reads a quoted operand
    (shell quoting would bake the quote characters into the value), and the
    task config is untrusted input, so anything that could open a new
    directive is refused rather than interpolated.

    *source_user* is the env Dockerfile's own final ``USER``: the layer
    installs as root, so the image's declared identity must be restored or
    the adaptation would silently grant root where Harbor withheld it.
    """
    policy = workspace_policy(task_dir)
    lines = []
    for key, value in sorted(policy["env"].items()):
        if not _DOCKER_ENV_KEY.fullmatch(key):
            raise ValueError(f"environment.env key {key!r} is not a usable variable name")
        # Docker substitutes ``$VAR`` in an ENV operand; the task declared a
        # literal, so the dollar is escaped rather than expanded at build.
        literal = json.dumps(value).replace("$", "\\$")
        lines.append(f"ENV {key}={literal}")
    if policy["workdir"]:
        lines.append(f"WORKDIR {json.dumps(policy['workdir'])}")
    # Validated at build, where a bad task config should fail, but applied per
    # phase at runtime rather than as a USER directive — see :func:`phase_uid`.
    for role in ("agent", "verifier"):
        _validated_user(policy[f"{role}_user"], source_user)
    return "".join(f"{line}\n" for line in lines)


async def adapt(
    path: str | Path,
    *,
    push: str | None = None,
    build: bool = True,
    hud_requirement: str = "hud",
) -> dict[str, str]:
    """Write adapted build contexts for every env group; build and push them.

    Returns ``{env_name: image_ref}`` — pass it to
    :func:`~harbor.load` as ``images=`` so the rows carry
    their image. *push* is a registry prefix (``registry.io/acme``); without it
    images stay local, which serves ``DockerRuntime`` but not cloud
    placements. ``build=False`` writes the contexts under ``.hud-adapt/`` and
    stops. *hud_requirement* pins the hud installed in-image — a PyPI
    requirement, or a path to a local wheel (baked into the context) for
    unreleased SDKs; it must speak the same control-channel protocol as the
    SDK driving the run.
    """
    root = Path(path).resolve()
    out_root = root / ".hud-adapt"
    images: dict[str, str] = {}
    for env_name, group_dirs in grouped(root):
        context = _write_context(out_root / env_name, env_name, group_dirs, hud_requirement)
        if not build:
            continue
        content = hash_directory(context)
        ref = f"{push}/{env_name}:{content}" if push else f"hud-harbor-adapted:{env_name}-{content}"
        deadlines = [
            t for t in (TaskConfig.read(d).environment.build_timeout_sec for d in group_dirs) if t
        ]
        await _docker(
            "build", "--tag", ref, str(context), deadline=max(deadlines) if deadlines else None
        )
        if push:
            await _docker("push", ref)
        images[env_name] = ref

    if images:
        LOGGER.info("adapted %d image(s)", len(images))
    return images


def _write_context(
    context: Path, env_name: str, group_dirs: list[Path], hud_requirement: str
) -> Path:
    """One group's adapted build context: env build context + the /hud layer."""
    if context.exists():
        shutil.rmtree(context)
    env_dir = group_dirs[0] / "environment"
    if not (env_dir / "Dockerfile").is_file():
        raise FileNotFoundError(f"group {env_name!r} has no environment/Dockerfile")
    _copy_task_content(env_dir, context)
    dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")

    multi_step = [d.name for d in group_dirs if not (d / "instruction.md").is_file()]
    if multi_step:
        raise NotImplementedError(
            "multi-step Harbor tasks (no root instruction.md) cannot be adapted yet: "
            + ", ".join(sorted(multi_step)[:5])
        )
    for task_dir in group_dirs:
        if reasons := unsupported_features(task_dir):
            raise NotImplementedError(
                f"Harbor task {task_dir.name!r} declares behaviour this integration "
                f"cannot reproduce: {'; '.join(reasons)}"
            )

    dockerignore = context / ".dockerignore"
    if dockerignore.is_file():
        # The task's ignore rules were written for its own build; they must
        # not exclude the adaptation layer from this one.
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
    for task_dir in group_dirs:
        target = hud_dir / "tasks" / task_dir.name
        target.mkdir(parents=True)
        for entry in ("instruction.md", "task.toml"):
            _copy_task_content(task_dir / entry, target / entry)
        _copy_task_content(task_dir / "tests", target / "tests")
    _write(
        hud_dir / "install.sh",
        _INSTALL_SH.replace("__HUD_REQUIREMENT__", shlex.quote(requirement))
        .replace("__PYTHON__", SERVING_PYTHON)
        .replace("__HUD_ROOT__", str(HUD_ROOT)),
    )
    # The image's own USER, kept for serve time: the build context is gone by
    # then, and a task that declares no user of its own still runs its phases
    # as whatever the image said.
    if (image_user := final_stage(dockerfile).user) is not None:
        _write(hud_dir / "image-user", f"{image_user}\n")
    layer = (
        _LAYER.replace("__HUD_ROOT__", str(HUD_ROOT))
        .replace("__ENV_NAME__", env_name)
        .replace("__PYTHON__", SERVING_PYTHON)
        .replace(
            # One env serves one policy, so the group's tasks agree on these.
            "__DECLARED__",
            _declared_directives(group_dirs[0], final_stage(dockerfile).user),
        )
    )
    _write(context / "Dockerfile", dockerfile + layer)
    return context


def _copy_task_content(source: Path, destination: Path) -> None:
    """Copy a task's own files into *destination*.

    A dataset is untrusted input: links are copied as links rather than
    followed, so a task cannot pull host content in through a symlink — into
    a build context, or into the ``/tests`` a rollout serves. Callers own
    what *destination* is; this owns what "copy a task's files" means.
    """
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True, ignore=_CONTEXT_IGNORE)
    else:
        shutil.copy2(source, destination, follow_symlinks=False)


def _write(path: Path, text: str) -> None:
    """LF endings: these files run in Linux containers, where ``\\r\\n``
    breaks shebangs and shell scripts."""
    path.write_text(text, encoding="utf-8", newline="\n")


def docker_runtime(**kwargs: Any) -> DockerRuntime:
    """A local placement for adapted images.

    An adapted image sandboxes inside itself — that is what keeps the baked
    tests and the verdict away from the agent — so its container needs the
    nested-namespace relaxation that plain images should not be given.
    """
    kwargs.setdefault("nested_sandbox", True)
    return DockerRuntime(**kwargs)


# ─── what an adapted image serves, from inside the container ────────────


def _harness_out_of_view() -> tuple[Mount, ...]:
    """Mounts that leave the harness's tree absent from a session's namespace.

    A tmpfs over the tree itself would hide what is in it and still leave the
    directory — and a directory named after the harness, at a path the task's
    own image never had, is the one thing left telling the graded party what
    is running it. Rebuilding the parent removes the entry instead: a fresh
    tmpfs, then every sibling bound back as it was. Only ``--tmpfs`` and
    ``--bind``, so it holds on the bubblewrap that ships with bullseye.

    What the sandbox loses is entries created *directly* in that parent during
    the rollout, which is why it is one nothing writes to: writes anywhere
    else, including new directories at the filesystem root, still land in the
    real filesystem for the verifier to grade.
    """
    parent = HUD_ROOT.parent
    mounts = [Mount("tmpfs", dst=str(parent))]
    if parent.is_dir():
        mounts += [
            Mount("rw", src=str(sibling), dst=str(sibling))
            for sibling in sorted(parent.iterdir())
            if sibling != HUD_ROOT
        ]
    return tuple(mounts)


def environment(ref: str | Path = HUD_ROOT / "tasks", *, name: str | None = None) -> Environment:
    """The live env serving the task dirs under *ref* — the contract verb.

    Harbor environments are container filesystems, so this constructor is
    meaningful only where the tasks' world is the current filesystem: inside
    an image :func:`adapt` built, whose CMD serves exactly this. The
    workspace is wherever the serving process starts (the image's
    ``WORKDIR`` — the adaptation layer preserves it); grading runs each
    task's ``tests/test.sh`` there, writing the Harbor reward under
    ``/logs``.
    """
    root = Path(ref)
    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not task_dirs:
        raise ValueError(f"no Harbor tasks under {root}")

    # The image already carries the task's declared environment, working dir
    # and user as Dockerfile directives, so this process starts inside them
    # and sessions inherit them.
    workdir = Path.cwd()
    # An image with no WORKDIR starts at the filesystem root. The agent still
    # gets the whole container (the bind below), but tracking every file in
    # it is not a meaningful diff — and walking it stalls startup — so file
    # tracking is off unless the task works somewhere specific.
    rooted_at_filesystem = workdir == Path("/")
    if rooted_at_filesystem:
        LOGGER.warning(
            "%s has no WORKDIR and declares no [environment] workdir; serving from / "
            "with file tracking disabled",
            root,
        )
    policy = workspace_policy(task_dirs[0])

    env = Environment(name or slugify(root.name))
    # The task's world is the whole container filesystem, exposed writable at
    # its real paths — the sandbox is here to control the network namespace
    # and to keep the graded material (baked tests, the serving venv) out of
    # the graded party's reach, not to narrow the filesystem.
    workspace = env.workspace(
        workdir,
        guest_path=workdir.as_posix(),
        system_mounts=(
            Mount("rw", src="/", dst="/"),
            Mount("proc", dst="/proc"),
            Mount("dev", dst="/dev"),
        ),
        # Masks go in ``mounts``, which bwrap applies *after* the workspace
        # bind — as system mounts they would be re-covered when the guest
        # path is ``/`` (an image with no WORKDIR). ``/tests`` and the verdict
        # are handled by not existing during the agent phase at all
        # (:func:`_hide_grading_dirs`), which is what Harbor does; the
        # harness's own tree has to exist throughout, being what serves, so it
        # is hidden by rebuilding its parent instead.
        mounts=_harness_out_of_view(),
        credentials_dir=HUD_ROOT / "session-keys",
        # The agent's declared identity, not the harness's: the serving
        # process stays root so it can place and remove the grading
        # directories and keep its own state under /hud.
        shell_uid=(agent_uid := phase_uid(task_dirs[0], "agent")),
        # Whose the workspace is, is the image's statement — Harbor does not
        # re-own it for the agent, so neither does this.
        hand_over_root=False,
        track_files=False if rooted_at_filesystem else None,
        # The agent phase's own variables, scoped to its sessions. A dropped
        # session would otherwise be pointed at the harness's home.
        env={**({"HOME": home} if (home := phase_home(agent_uid)) else {}), **policy["agent_env"]},
        network=policy["network"],
        # What the task declared its agent may reach. Given a set, the
        # workspace takes a network of its own whose only way out applies it —
        # which is also what puts the substrate, and the channel grading this
        # rollout, out of the agent's reach.
        allowed_hosts=policy["allowed_hosts"],
        # Always: the sandbox is what keeps the baked tests and the serving
        # venv out of the graded party's reach — an unsandboxed fallback
        # would hand the agent the verifier's answers.
        require_isolation=True,
    )

    for task_dir in task_dirs:
        _register(env, task_dir, workdir, workspace)
    return env


def _register(env: Environment, task_dir: Path, workdir: Path, workspace: Workspace) -> None:
    config = TaskConfig.read(task_dir)

    @env.template(
        id=task_dir.name,
        description=config.task.description or f"Harbor task {task_dir.name}",
    )
    async def _run_harbor_task() -> AsyncGenerator[Any, Any]:
        # Harbor uploads /tests when it runs the verifier, so its agent phase
        # never contains the assertions it is graded on — nor the verdict dir.
        # Adapted images ship both (empty), and an earlier rollout in this
        # container leaves them behind, so the agent phase starts by removing
        # them and ends the same way: :func:`_grade` lays them down in between,
        # for as long as the verifier needs them.
        _hide_grading_dirs()
        try:
            answer = yield (task_dir / "instruction.md").read_text(encoding="utf-8")
            yield await _grade(task_dir, workdir, answer, workspace)
        finally:
            _hide_grading_dirs()
            # Harbor's agent phase is one continuous session, so a service the
            # agent starts is still running when the verifier looks for it —
            # and is gone before the next rollout, which this container may
            # well serve too.
            await workspace.discard_sandbox()


def _sync_tests(task_dir: Path) -> None:
    """Leave ``/tests`` holding exactly *task_dir*'s tests."""
    _reset_dir(TESTS)
    for child in (task_dir / "tests").iterdir():
        _copy_task_content(child, TESTS / child.name)


def _hide_grading_dirs() -> None:
    """Leave no ``/tests`` or verdict directory for the agent phase to find.

    Removed rather than emptied: an empty directory the base image does not
    have is itself a signal, and these two are the harness's whole visible
    footprint outside its own tree. Where the serve process cannot remove them
    — not root, and they sit at the filesystem root — emptying is the fallback,
    which is what the mask used to achieve.
    """
    for path in (TESTS, VERIFIER_LOGS):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            continue
        except OSError:
            LOGGER.debug("could not remove %s; emptying it instead", path)
            _reset_dir(path)


def _reset_dir(path: Path) -> None:
    """Leave *path* an existing, empty directory, whatever it was before.

    Contents are cleared in place rather than removing and recreating the
    directory: a non-root serve process can empty ``/tests`` but cannot
    recreate it at the filesystem root.
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
    if not path.is_dir():
        path.mkdir(parents=True)
        return
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


async def _grade(
    task_dir: Path, workdir: Path, answer: Any, workspace: Workspace | None = None
) -> dict[str, Any]:
    logs = LOGS
    # The agent shares the container, so restore the verifier from the baked
    # (masked) copy before running it.
    _sync_tests(task_dir)

    # Run the script itself: its shebang picks the interpreter Harbor's task
    # intends, and a minimal image may have no bash at all.
    test_sh = TESTS / "test.sh"
    test_sh.chmod(test_sh.stat().st_mode | 0o111)
    argv = [str(test_sh)]

    config = TaskConfig.read(task_dir)
    verifier_env: dict[str, str] = {}
    verifier_uid = phase_uid(task_dir, "verifier")
    setpriv = shutil.which("setpriv")
    if verifier_uid is not None and setpriv is not None and os.geteuid() == 0:
        # Root laid the grading directories down, so hand them over before
        # dropping: a verifier that cannot write /logs/verifier cannot produce
        # the verdict it is being run to produce. The verdict directory is made
        # here rather than by the grading helper, since it must exist first.
        _reset_dir(VERIFIER_LOGS)
        for root_path in (TESTS, VERIFIER_LOGS):
            for target in (root_path, *root_path.rglob("*")):
                os.lchown(target, verifier_uid, verifier_uid)
        if home := phase_home(verifier_uid):
            verifier_env["HOME"] = home
        argv = [
            setpriv,
            "--reuid",
            str(verifier_uid),
            "--regid",
            str(verifier_uid),
            "--clear-groups",
            *argv,
        ]
    joins_workspace_net = (
        workspace is not None and workspace.owns_netns and config.network("verifier")
    )
    if joins_workspace_net and (sandbox := await workspace.sandbox_pid()) is not None:  # type: ignore[union-attr]
        # Harbor grades a service by talking to it — telnet to the VM the agent
        # booted, git clone from the server it configured. Those listen in the
        # workspace's network now, so the verifier is run there: its user and
        # network namespaces, and nothing else. Keeping this mount namespace is
        # the point — /tests and the verdict are here, where the workspace
        # cannot reach them — and the ids map through, so it stays able to
        # chown what it untars.
        # Behind the workspace's boundary now, so it takes the workspace's way
        # out: Harbor's verifiers commonly install their own tooling first.
        verifier_env.update(workspace.egress_environment())  # type: ignore[union-attr]
        nsenter = shutil.which("nsenter") or "/usr/bin/nsenter"
        argv = [
            nsenter,
            "--target",
            str(sandbox),
            "--user",
            "--net",
            "--preserve-credentials",
            "--",
            *argv,
        ]

    severed = not config.network("verifier")
    if severed and (bwrap := usable_bwrap()) is None:
        raise RuntimeError(
            "the verifier declares no-network but bwrap cannot sandbox here; "
            "refusing to grade with network access the task ruled out"
        )

    async def run_tests() -> ProcessGroup:
        spawn = argv
        environ = {**os.environ, **verifier_env, **config.verifier.env}
        if not severed:
            return await create_process_group_exec(
                *spawn,
                cwd=workdir,
                env=environ,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        # A network namespace of its own is the only reason the verifier is
        # sandboxed at all, and bwrap can only make one inside a user
        # namespace. That namespace maps a single id unless this side says
        # otherwise — and a verifier that cannot chown is one that cannot
        # untar, or run apt, which is most of what Harbor's verifiers do
        # before they assert anything. So it is held at creation and mapped,
        # exactly as an agent session's sandbox is.
        info_read, info_write = os.pipe()
        block_read, block_write = os.pipe()
        try:
            os.set_inheritable(info_write, True)
            os.set_inheritable(block_read, True)
            # /dev and /proc are fresh: binding the real root inside a user
            # namespace leaves device nodes unwritable, and a test.sh
            # redirecting to /dev/null would fail on that alone.
            group = await create_process_group_exec(
                bwrap,
                "--unshare-user",
                "--info-fd",
                str(info_write),
                "--userns-block-fd",
                str(block_read),
                "--bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--unshare-net",
                "--",
                *spawn,
                cwd=workdir,
                env=environ,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=(info_write, block_read),
            )
            os.close(info_write)
            os.close(block_read)
            info_write = block_read = -1
            await install_identity_map(info_read, block_write)
            return group
        finally:
            os.close(info_read)
            os.close(block_write)
            for stray in (info_write, block_read):
                if stray != -1:
                    os.close(stray)

    return await _grade_with_verifier(config, logs, answer, run_tests)


# ─── verifier grading and docker plumbing ───────────────────────────────


async def _grade_with_verifier(
    config: TaskConfig,
    logs: Path,
    answer: Any,
    run_tests: Callable[[], Awaitable[ProcessGroup]],
) -> dict[str, Any]:
    """Run the Harbor verifier and shape its output into a HUD grade.

    *run_tests* starts the ``tests/test.sh`` process group wherever it must
    run; this owns the answer file, the ``[verifier] timeout_sec`` bound, and
    parsing ``reward.json``/``reward.txt``.
    """
    timeout = config.verifier.timeout_sec or DEFAULT_VERIFIER_TIMEOUT
    # Harbor's harness guarantees the verifier output dir exists. The agent's
    # namespace masks it, but this process shares the container with whatever
    # the *setup* left behind, so the verdict dir is recreated and the answer
    # file is written without following anything into it.
    _reset_dir(logs / "verifier")
    _write_no_follow(logs / "agent_answer.txt", "" if answer is None else str(answer))
    group = await run_tests()
    proc = group.process
    # Drain while waiting, and bound on the script's own exit. Draining only
    # after exit deadlocks a chatty verifier once the pipe buffer fills;
    # waiting for pipe EOF instead reports a timeout for a script that
    # finished but left a daemon holding its stdout. Reading concurrently and
    # timing the process avoids both.
    assert proc.stdout is not None and proc.stderr is not None
    reading = (
        asyncio.create_task(proc.stdout.read()),
        asyncio.create_task(proc.stderr.read()),
    )
    try:
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            timed_out = False
        except TimeoutError:
            timed_out = True
        # The group is the boundary however test.sh ends: descendants release
        # the inherited pipes here (so the reads complete), and none survive
        # to write /logs during the next rollout's grading.
        await group.terminate()
        out_bytes, err_bytes = await asyncio.gather(*reading)
    except BaseException:
        # A cancelled rollout unwinds through here; the group and the readers
        # are this function's to release however it exits.
        for reader in reading:
            reader.cancel()
        await group.terminate()
        raise
    if timed_out:
        return {
            "score": 0.0,
            "isError": True,
            "content": f"Harbor verifier timed out after {timeout:.0f}s",
            "info": {
                "verifier_timeout_sec": timeout,
                "stdout": out_bytes.decode("utf-8", "replace")[-4000:],
                "stderr": err_bytes.decode("utf-8", "replace")[-4000:],
            },
        }
    # The reward file is Harbor's verdict, not the exit code: verifiers
    # commonly end with `exit "$status"`, carrying the test suite's code after
    # writing the score it implies. A verifier that exits nonzero *without*
    # writing a reward is caught below, as a grading error.
    reward, info = _read_reward(logs / "verifier")
    info.update(
        {
            "stdout": out_bytes.decode("utf-8", "replace")[-4000:],
            "stderr": err_bytes.decode("utf-8", "replace")[-4000:],
        }
    )
    if reward is None:
        return {
            "score": 0.0,
            "isError": True,
            "content": "Harbor verifier did not write reward.json or reward.txt",
            "info": info,
        }
    return {"score": reward, "info": info}


def _read_reward(verifier_logs: Path) -> tuple[float | None, dict[str, Any]]:
    """Parse Harbor's verifier output: ``reward.json`` first, then ``reward.txt``.

    A reward is a finite number; booleans (an ``int`` subclass) and
    ``nan``/``inf`` are parse failures, not scores.
    """
    reward_json = verifier_logs / "reward.json"
    if reward_json.is_file():
        try:
            data = json.loads(reward_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None, {"parse_error": "reward.json is not valid JSON"}
        if (score := _as_score(data)) is not None:
            return score, {"reward_file": str(reward_json)}
        if isinstance(data, dict):
            for key in ("reward", "score"):
                if (score := _as_score(data.get(key))) is not None:
                    return score, {"reward_file": str(reward_json), "reward_json": data}
        return None, {"reward_file": str(reward_json), "reward_parse_error": "no numeric reward"}

    reward_txt = verifier_logs / "reward.txt"
    if reward_txt.is_file():
        text = reward_txt.read_text(encoding="utf-8").strip()
        try:
            value = float(text)
        except ValueError:
            return None, {"reward_file": str(reward_txt), "reward_parse_error": text}
        if not math.isfinite(value):
            return None, {"reward_file": str(reward_txt), "reward_parse_error": text}
        return value, {"reward_file": str(reward_txt)}

    return None, {}


def _write_no_follow(path: Path, text: str) -> None:
    """Write *text* to *path* itself — never through a symlink planted there."""
    if path.is_symlink():
        path.unlink()
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def _as_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(float(value)) else None
