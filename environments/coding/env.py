"""Coding environment (HUD v6): a repo workspace with diff-based grading.

The env publishes one ``ssh`` workspace over a git repo; the agent's harness
brings its own bash/file tools. Task setup vaults the repo's real history
(:mod:`coding.repo`) so the agent gets a fresh single-commit repo with working
git but no answer-key refs; grading captures the agent's changes as a diff,
resets to the pre-agent snapshot, re-applies, brings in the hidden tests, and
runs them.

Three task templates:

- ``coding-task`` — hidden tests live as refs in the repo's vaulted history
  (the ``{task}_baseline`` / ``{task}_test`` / ``{task}_golden`` branch
  convention) and a shell command scores by exit code. Runs locally (clones
  ``REPO_URL`` per process) or from an image that bakes the repo.
- ``sdlc-task`` — adds workflow: a bare mock-GitHub remote the agent pushes
  to, ``github_*`` issue/PR tools (:mod:`coding.github`), and grading over
  the pushed pull-request branch instead of the worktree.
- one SWE-bench Pro instance template, registered only when an instance dir
  is baked in (see ``swe_tasks.py``): grading replays the official evaluator
  (:mod:`coding.swe_bench_pro`).

Images set ``REPO_DIR``; locally everything lives in per-process temp dirs.
The vault and instance assets sit outside the workspace — under ``/hud``
(root, mode 700) in images — and agent shells drop to a non-root uid, so only
the env process can read them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from hud.environment import Environment, Workspace
from hud.graders import BashGrader, EvaluationResult, LLMJudgeGrader, SubScore, combine
from hud.settings import settings

from coding import repo as repo_lib
from coding import swe_bench_pro as swe
from coding.github import MockGitHub
from coding.github import serve as serve_github

logger = logging.getLogger(__name__)

# ─── substrate layout ────────────────────────────────────────────────────
# Images set REPO_DIR (and ship the repo); locally everything lives in
# per-process temp dirs and the repo is cloned from REPO_URL at startup.

_explicit_repo = os.environ.get("REPO_DIR")
if _explicit_repo:
    REPO_DIR = Path(_explicit_repo)
    VAULT_DIR = Path(os.environ.get("VAULT_DIR", "/hud/vault"))
    LOGS_DIR = Path(os.environ.get("GRADING_LOGS_DIR", "/hud/logs"))
    _LOCAL_SUBSTRATE = False
else:
    _LOCAL_ROOT = Path(tempfile.gettempdir()) / "hud-coding"
    REPO_DIR = _LOCAL_ROOT / f"repo-{os.getpid()}"
    VAULT_DIR = _LOCAL_ROOT / f"vault-{os.getpid()}"
    LOGS_DIR = _LOCAL_ROOT / f"logs-{os.getpid()}"
    _LOCAL_SUBSTRATE = True

# The SDLC flavor's mock-GitHub remote: a bare repo the agent pushes to. It
# must be writable by the agent, so it lives outside /hud.
REMOTE_DIR = (
    Path(os.environ.get("REMOTE_DIR", "/srv/git/project.git"))
    if _explicit_repo
    else _LOCAL_ROOT / f"remote-{os.getpid()}" / "project.git"
)

REPO_URL = os.environ.get("REPO_URL", "https://github.com/hud-evals/coding-template-sample")
TESTS_TIMEOUT = float(os.environ.get("GRADING_TIMEOUT", "3600"))

INSTANCE_DIR = Path(os.environ.get("INSTANCE_DIR", "/hud/instance"))

AGENT_UID = 1000
AGENT_HOME = Path("/tmp/agent-home")  # noqa: S108 - container-local, created at setup

env = Environment(name="coding")

# shell_uid is the privilege wall: the env process (root, in images) keeps the
# vault and instance assets under /hud at mode 700; the uid-dropped agent can
# edit the repo but never read the answer key. No-op off root (local).
_ws = Workspace(
    REPO_DIR,
    guest_path=str(REPO_DIR),
    network=True,
    env={"HOME": str(AGENT_HOME)},
    track_files=settings.file_tracking_enabled,
    shell_uid=AGENT_UID,
)

_github = MockGitHub()
_github_server: asyncio.Task[None] | None = None


@env.initialize
async def _up() -> None:
    global _github_server
    if _LOCAL_SUBSTRATE and not (REPO_DIR / ".git").exists():
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        logger.info("cloning %s into %s", REPO_URL, REPO_DIR)
        await repo_lib.run("git", "clone", "-q", REPO_URL, str(REPO_DIR), cwd=REPO_DIR.parent)
    await _ws.start()
    env.add_capability(_ws.capability("shell"))
    if _ws.tracks_files:
        env.add_capability(_ws.file_tracking_capability())
    _github_server, github_capability = serve_github(_github)
    env.add_capability(github_capability)


@env.shutdown
async def _down() -> None:
    if _github_server is not None:
        _github_server.cancel()
    await _ws.stop()
    if _LOCAL_SUBSTRATE:
        for path in (REPO_DIR, VAULT_DIR, LOGS_DIR, REMOTE_DIR.parent):
            shutil.rmtree(path, ignore_errors=True)


# ─── shared task lifecycle ───────────────────────────────────────────────


async def _setup(base_ref: str | None = None, *, remote: bool = False) -> None:
    """Check out the task's baseline, vault the history, hand the repo to the agent.

    With ``remote=True`` (the SDLC flavor) a bare "GitHub" remote is created
    from the baseline first, and the agent's repo is attached to it as a
    normal clone would be — realistic history, pushes, and branches, but no
    reachable answer-key refs.
    """
    if base_ref:
        await repo_lib.git(REPO_DIR, "checkout", "-qf", base_ref)
    if remote:
        await repo_lib.create_remote(REPO_DIR, REMOTE_DIR, base_ref or "HEAD")
    await repo_lib.vault_history(REPO_DIR, VAULT_DIR)
    if remote:
        await repo_lib.attach_to_remote(REPO_DIR, REMOTE_DIR)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        AGENT_HOME.mkdir(parents=True, exist_ok=True)
        uid = f"{AGENT_UID}:{AGENT_UID}"
        paths = [str(REPO_DIR), str(AGENT_HOME)] + ([str(REMOTE_DIR)] if remote else [])
        await repo_lib.run("chown", "-R", uid, *paths, cwd=Path("/"))
        if remote:
            # The grader (root) pushes to and clones from the agent-owned remote.
            await repo_lib.run("git", "config", "--global", "--add", "safe.directory", str(REMOTE_DIR), cwd=Path("/"))


# ─── generic flavor ──────────────────────────────────────────────────────


async def _grade_generic(
    test_command: str,
    base_ref: str | None,
    test_ref: str | None,
    test_files: list[str] | None,
    golden_ref: str | None,
    validate_mode: str | None,
) -> EvaluationResult:
    setup_commit = await repo_lib.restore_history(REPO_DIR, VAULT_DIR)
    if validate_mode == "golden":
        if not (base_ref and golden_ref):
            raise ValueError('validate_mode="golden" needs base_ref and golden_ref')
        _, diff, _ = await repo_lib.git(REPO_DIR, "diff", base_ref, golden_ref)
    else:
        diff = await repo_lib.capture_agent_diff(REPO_DIR, setup_commit)

    await repo_lib.reset_worktree(REPO_DIR, setup_commit)
    apply_error = await repo_lib.apply_diff(REPO_DIR, diff, LOGS_DIR / "patch.diff")
    if apply_error is not None:
        return EvaluationResult(reward=0.0, content="patch failed to apply", info={"git_apply": apply_error})
    if test_ref:
        # Hidden tests come from the vaulted history, after the agent's diff —
        # so agent edits to them never survive into grading.
        await repo_lib.git(REPO_DIR, "checkout", test_ref, "--", *(test_files or ["."]))

    command = test_command.format(test_files=" ".join(test_files or []))
    return await combine(
        BashGrader.grade(weight=1.0, command=command, cwd=str(REPO_DIR), timeout_seconds=int(TESTS_TIMEOUT))
    )


@env.template(id="coding-task", description="Fix a bug in the repo; hidden tests grade the diff.")
async def coding_task(
    description: str,
    test_command: str,
    base_ref: str | None = None,
    test_ref: str | None = None,
    test_files: list[str] | None = None,
    golden_ref: str | None = None,
    validate_mode: str | None = None,
):
    """Generic coding task.

    ``base_ref`` is the agent's starting state; ``test_ref`` holds the hidden
    tests (``test_files`` are checked out from it at grade time);
    ``test_command`` scores by exit code, with ``{test_files}`` expanded.
    ``validate_mode="golden"`` grades ``base_ref..golden_ref`` instead of
    agent edits.
    """
    if validate_mode not in (None, "golden"):
        raise ValueError(f"unknown validate_mode: {validate_mode!r}")
    await _setup(base_ref)
    _ = yield (
        f"You are working in a coding repository located at {REPO_DIR}.\n\n"
        "Use the tools provided to complete the following task. Hidden tests grade your "
        f"work when you finish; do not modify existing tests.\n\n{description}"
    )
    yield await _grade_generic(test_command, base_ref, test_ref, test_files, golden_ref, validate_mode)


# ─── SDLC flavor: issue-driven fix, graded on the pushed PR branch ───────


async def _fetch_hidden_ref(dest: Path, ref: str) -> None:
    """Fetch an answer-key ref from the vaulted history into *dest*."""
    full = f"refs/remotes/{ref}" if ref.startswith("origin/") else ref
    await repo_lib.git(dest, "fetch", "-q", str(VAULT_DIR / "git"), full)


async def _grade_sdlc(
    test_command: str,
    test_ref: str,
    test_files: list[str],
    golden_ref: str | None,
    pr_rubric: str | None,
    validate_mode: str | None,
) -> EvaluationResult:
    if validate_mode == "golden":
        # Simulate the workflow the agent is graded on: push the reference
        # fix as a branch and open a pull request for it.
        if not golden_ref:
            raise ValueError('validate_mode="golden" needs golden_ref')
        await repo_lib.git(VAULT_DIR / "git", "push", "-q", str(REMOTE_DIR), f"{golden_ref}:refs/heads/golden-fix")
        _github.create_pull_request(
            "Fix the reported bug", "Applies the reference fix.", head="golden-fix", base="main"
        )

    pr = _github.latest_pull_request()
    if pr is None:
        return EvaluationResult(reward=0.0, content="no pull request was opened")

    # Hermetic checkout of the PR head from the remote (never the worktree).
    grading_dir = LOGS_DIR / "pr-checkout"
    shutil.rmtree(grading_dir, ignore_errors=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    await repo_lib.run("git", "clone", "-q", str(REMOTE_DIR), str(grading_dir), cwd=LOGS_DIR)
    code, _, err = await repo_lib.git(grading_dir, "checkout", "-q", pr.head, check=False)
    if code != 0:
        return EvaluationResult(
            reward=0.0,
            content=f"pull request head branch {pr.head!r} was never pushed",
            info={"git_checkout": err.strip()[-2000:]},
        )

    await _fetch_hidden_ref(grading_dir, test_ref)
    await repo_lib.git(grading_dir, "checkout", "FETCH_HEAD", "--", *test_files)

    command = test_command.format(test_files=" ".join(test_files))
    quality: Any
    if pr_rubric:
        quality = LLMJudgeGrader.grade(weight=0.2, answer=_github.transcript(), criteria=[pr_rubric])
    else:
        opened_well = bool(pr.title.strip() and pr.body.strip())
        quality = SubScore(name="pull_request", value=1.0 if opened_well else 0.0, weight=0.2)
    return await combine(
        BashGrader.grade(weight=0.8, command=command, cwd=str(grading_dir), timeout_seconds=int(TESTS_TIMEOUT)),
        quality,
    )


@env.template(
    id="sdlc-task",
    description="Issue-driven bug fix: fix the code, push a branch, open a pull request.",
)
async def sdlc_task(
    description: str,
    test_command: str,
    base_ref: str,
    test_ref: str,
    test_files: list[str],
    issues: list[dict[str, Any]] | None = None,
    golden_ref: str | None = None,
    pr_rubric: str | None = None,
    validate_mode: str | None = None,
):
    """The generic flavor plus workflow: the repo has an ``origin`` remote (a
    bare mock-GitHub repo) and ``github_*`` tools seeded with *issues*; the
    deliverable is a pushed branch with a pull request. Grading checks the PR
    head out of the remote, brings in the hidden tests, runs ``test_command``
    (weight 0.8), and scores the PR itself (weight 0.2) — structurally, or
    against ``pr_rubric`` with an LLM judge when provided.
    """
    if validate_mode not in (None, "golden"):
        raise ValueError(f"unknown validate_mode: {validate_mode!r}")
    _github.seed(issues or [])
    await _setup(base_ref, remote=True)
    _ = yield (
        f"You are working in a coding repository located at {REPO_DIR}; its `origin` remote "
        "is the team's shared repository, and the `github_*` tools give you the team's "
        "issue tracker.\n\n"
        "Complete the following task. When you are done, push your work to `origin` as a "
        "new branch and open a pull request for it with `github_create_pull_request`. "
        "Hidden tests grade your work; do not modify existing tests.\n\n"
        f"{description}"
    )
    yield await _grade_sdlc(test_command, test_ref, test_files, golden_ref, pr_rubric, validate_mode)


# ─── SWE-bench Pro flavor (present only in baked instance images) ────────

if (INSTANCE_DIR / "instance.json").is_file():
    INSTANCE = swe.load_instance(INSTANCE_DIR)

    @env.template(
        id=INSTANCE["instance_id"],
        description=f"SWE-bench Pro instance on {INSTANCE['repo']}",
    )
    async def swe_bench_pro_task(validate_mode: str | None = None):
        """One SWE-bench Pro rollout. ``validate_mode="golden"`` grades the
        dataset's golden patch instead of agent edits (gold-patch validation)."""
        if validate_mode not in (None, "golden"):
            raise ValueError(f"unknown validate_mode: {validate_mode!r}")
        await _setup()
        _ = yield swe.build_prompt(INSTANCE, REPO_DIR)
        yield await swe.grade(
            INSTANCE,
            INSTANCE_DIR,
            REPO_DIR,
            VAULT_DIR,
            LOGS_DIR,
            validate_mode=validate_mode,
            tests_timeout=TESTS_TIMEOUT,
        )
