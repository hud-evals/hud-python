"""Git-history vaulting and diff-based grading primitives.

The repo's real ``.git`` may contain answer keys such as solution branches and
fix commits. The environment uses this lifecycle:

- setup (:func:`vault_history`): snapshot the pre-agent worktree into the
  real history, move ``.git`` into a vault outside the workspace, and leave a
  fresh single-commit repo — git works normally for the agent, but there is
  no history, no refs, and no remotes.
- grading (:func:`restore_history` → :func:`capture_agent_diff` →
  :func:`reset_worktree` → :func:`apply_diff`): discard the agent's ``.git``
  (hooks and history included), restore the vaulted history, capture the
  agent's changes as a diff against the setup snapshot, reset the worktree to
  that snapshot, and re-apply the diff. Task-authored tests land after the
  agent's diff.

The snapshot covers the whole worktree, so installed dependencies and build
artifacts survive into grading.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

_GIT_ID = ("-c", "user.name=hud-grader", "-c", "user.email=grader@hud.ai")
_SETUP_COMMIT = "setup_commit"


async def run(
    *argv: str,
    cwd: Path,
    timeout: float = 600.0,
    check: bool = True,
) -> tuple[int, bytes, bytes]:
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
    if check and proc.returncode != 0:
        detail = (err_bytes.strip() or out_bytes.strip())[:2000].decode("utf-8", "replace")
        raise RuntimeError(f"{' '.join(argv)} failed ({proc.returncode}): {detail}")
    return proc.returncode if proc.returncode is not None else 1, out_bytes, err_bytes


async def git(
    repo: Path,
    *args: str,
    timeout: float = 600.0,
    check: bool = True,
) -> tuple[int, bytes, bytes]:
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

    _append_git_exclude(repo / ".git")
    await git(repo, "add", "-A")
    await git(repo, "commit", "-q", "--allow-empty", "-m", "hud: pre-agent snapshot")
    _, sha, _ = await git(repo, "rev-parse", "HEAD")

    vault.mkdir(parents=True, exist_ok=True)
    (vault / _SETUP_COMMIT).write_text(sha.decode().strip(), "utf-8")
    shutil.move(str(repo / ".git"), str(vault / "git"))

    await git(repo, "init", "-q", ".")
    _append_git_exclude(repo / ".git")
    await git(repo, "add", "-A")
    await git(repo, "commit", "-q", "--allow-empty", "-m", "baseline")


async def restore_history(repo: Path, vault: Path) -> str:
    """Swap the agent's throwaway ``.git`` for the vaulted real history.

    Returns the setup-snapshot commit sha recorded by :func:`vault_history`.
    """
    agent_git = repo / ".git"
    if agent_git.exists():
        shutil.rmtree(agent_git)
    shutil.move(str(vault / "git"), str(agent_git))
    return (vault / _SETUP_COMMIT).read_text("utf-8").strip()


async def capture_agent_diff(repo: Path, setup_commit: str) -> bytes:
    """The agent's changes: setup snapshot -> final worktree state.

    Runs on the restored real history: the final worktree (including files the
    agent added) is committed and diffed against the snapshot.
    """
    await git(repo, "add", "-A")
    await git(repo, "commit", "-q", "--allow-empty", "-m", "hud: final snapshot")
    _, final, _ = await git(repo, "rev-parse", "HEAD")
    return await diff_refs(repo, setup_commit, final.decode().strip())


async def diff_refs(repo: Path, before: str, after: str) -> bytes:
    """Return an applyable patch between two refs, including binary changes."""
    _, diff, _ = await git(repo, "diff", "--binary", before, after, timeout=1200.0)
    return diff


async def reset_worktree(repo: Path, setup_commit: str) -> None:
    await git(repo, "reset", "--hard", setup_commit)
    await git(repo, "clean", "-fd")


async def restore_paths(repo: Path, ref: str, paths: list[str]) -> None:
    """Restore authored test paths from *ref*, removing paths absent there."""
    for value in dict.fromkeys(paths):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"test path must stay inside the repository: {value!r}")

        target = repo / path
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)

        code, _, _ = await git(repo, "cat-file", "-e", f"{ref}:{path.as_posix()}", check=False)
        if code == 0:
            await git(repo, "checkout", ref, "--", path.as_posix())


async def apply_diff(repo: Path, diff: str | bytes, patch_path: Path) -> str | None:
    """Apply *diff* to the worktree; returns git's error output on failure."""
    data = diff.encode() if isinstance(diff, str) else diff
    if not data.strip():
        return None
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(data)
    code, out, err = await git(repo, "apply", "-v", str(patch_path), check=False)
    if code != 0:
        return (err.strip() or out.strip())[-4000:].decode("utf-8", "replace")
    return None
