"""Git-history vaulting and diff-based grading primitives.

The repo's real ``.git`` may contain the answer key — solution branches, the
fix commit, the hidden tests. Every task flavor shares this lifecycle:

- setup (:func:`vault_history`): snapshot the pre-agent worktree into the
  real history, move ``.git`` into a vault outside the workspace, and leave a
  fresh single-commit repo — git works normally for the agent, but there is
  no history, no refs, and no remotes.
- grading (:func:`restore_history` → :func:`capture_agent_diff` →
  :func:`reset_worktree` → :func:`apply_diff`): discard the agent's ``.git``
  (hooks and history included), restore the vaulted history, capture the
  agent's changes as a diff against the setup snapshot, reset the worktree to
  that snapshot, and re-apply the diff. Hidden tests come from vaulted refs
  and land after the agent's diff.

The snapshot covers the whole worktree, so installed dependencies and build
artifacts survive into grading.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

_GIT_ID = ("-c", "user.name=hud-grader", "-c", "user.email=grader@hud.ai")
_SETUP_COMMIT = "setup_commit"


async def run(
    *argv: str,
    cwd: Path,
    timeout: float = 600.0,
    check: bool = True,
) -> tuple[int, str, str]:
    """Run a command, returning ``(returncode, stdout, stderr)``. Kills on timeout."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out_bytes, err_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"{' '.join(argv[:4])} timed out after {timeout:.0f}s") from None
    out = out_bytes.decode("utf-8", "replace")
    err = err_bytes.decode("utf-8", "replace")
    if check and proc.returncode != 0:
        detail = (err.strip() or out.strip())[:2000]
        raise RuntimeError(f"{' '.join(argv)} failed ({proc.returncode}): {detail}")
    return proc.returncode if proc.returncode is not None else 1, out, err


async def git(repo: Path, *args: str, timeout: float = 600.0, check: bool = True) -> tuple[int, str, str]:
    return await run("git", *_GIT_ID, *args, cwd=repo, timeout=timeout, check=check)


def _append_git_exclude(git_dir: Path) -> None:
    """Keep the workspace's SSH credential dir out of snapshots and diffs."""
    exclude = git_dir / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as fh:
        fh.write("\n.hud/\n")


async def vault_history(repo: Path, vault: Path) -> None:
    """Snapshot the pre-agent state, then hide the repo's history in *vault*."""
    if (vault / "git").exists():
        raise RuntimeError(f"vault {vault} already holds a history: one task per substrate")

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        # Grade-time git runs as root on the (chowned) agent-owned repo; git
        # honors safe.directory only from system/global config, never from -c.
        await run("git", "config", "--global", "--add", "safe.directory", str(repo), cwd=Path("/"))

    _append_git_exclude(repo / ".git")
    await git(repo, "add", "-A")
    await git(repo, "commit", "-q", "--allow-empty", "-m", "hud: pre-agent snapshot")
    _, sha, _ = await git(repo, "rev-parse", "HEAD")

    vault.mkdir(parents=True, exist_ok=True)
    (vault / _SETUP_COMMIT).write_text(sha.strip(), "utf-8")
    shutil.move(str(repo / ".git"), str(vault / "git"))

    await git(repo, "init", "-q", ".")
    _append_git_exclude(repo / ".git")
    await git(repo, "add", "-A")
    await git(repo, "commit", "-q", "--allow-empty", "-m", "baseline")


async def create_remote(repo: Path, remote: Path, ref: str, *, branch: str = "main") -> None:
    """A bare "GitHub" remote holding *ref*'s history as *branch*.

    Pushing a single ref publishes only its ancestry — sibling answer-key
    branches (hidden tests, golden fix) stay unreachable, while the agent
    still gets realistic past history to work with.
    """
    remote.parent.mkdir(parents=True, exist_ok=True)
    await git(remote.parent, "init", "-q", "--bare", "-b", branch, str(remote))
    await git(repo, "push", "-q", str(remote), f"{ref}:refs/heads/{branch}")


async def attach_to_remote(repo: Path, remote: Path, *, branch: str = "main") -> None:
    """Point the agent's fresh repo at the remote, sharing its (sanitized) history.

    Called after :func:`vault_history`: the throwaway single-commit repo is
    replaced by the remote's clone state — same worktree, but with ``origin``
    configured and history the agent can branch from and push back to.
    """
    shutil.rmtree(repo / ".git")
    await git(repo, "init", "-q", "-b", branch, ".")
    _append_git_exclude(repo / ".git")
    await git(repo, "remote", "add", "origin", str(remote))
    await git(repo, "fetch", "-q", "origin")
    # Adopt the remote's history without a checkout: the worktree already
    # matches (plus untracked build artifacts a checkout would refuse to
    # cross), so point the branch at origin and sync only the index.
    await git(repo, "update-ref", f"refs/heads/{branch}", f"refs/remotes/origin/{branch}")
    await git(repo, "reset", "-q")
    await git(repo, "branch", "-q", f"--set-upstream-to=origin/{branch}", branch)


async def restore_history(repo: Path, vault: Path) -> str:
    """Swap the agent's throwaway ``.git`` for the vaulted real history.

    Returns the setup-snapshot commit sha recorded by :func:`vault_history`.
    """
    agent_git = repo / ".git"
    if agent_git.exists():
        shutil.rmtree(agent_git)
    shutil.move(str(vault / "git"), str(agent_git))
    return (vault / _SETUP_COMMIT).read_text("utf-8").strip()


async def capture_agent_diff(repo: Path, setup_commit: str) -> str:
    """The agent's changes: setup snapshot -> final worktree state.

    Runs on the restored real history: the final worktree (including files the
    agent added) is committed and diffed against the snapshot.
    """
    await git(repo, "add", "-A")
    await git(repo, "commit", "-q", "--allow-empty", "-m", "hud: final snapshot")
    _, final, _ = await git(repo, "rev-parse", "HEAD")
    _, diff, _ = await git(repo, "diff", setup_commit, final.strip(), timeout=1200.0)
    return diff


async def reset_worktree(repo: Path, setup_commit: str) -> None:
    await git(repo, "reset", "--hard", setup_commit)
    await git(repo, "clean", "-fd")


async def apply_diff(repo: Path, diff: str, patch_path: Path) -> str | None:
    """Apply *diff* to the worktree; returns git's error output on failure."""
    if not diff.strip():
        return None
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(diff, "utf-8")
    code, out, err = await git(repo, "apply", "-v", str(patch_path), check=False)
    if code != 0:
        return (err.strip() or out.strip())[-4000:]
    return None
