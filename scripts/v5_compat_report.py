"""Empirical v5-on-v6 compat report over a corpus of environment repos.

Walks a directory of environment repos (default: ``environments/``), finds the
files that define a HUD environment, imports each one in an isolated subprocess
against the *current* SDK, and reports what happened:

- ``ok``            imported; Environments/tasks/capabilities counted
- ``hud-gap``       import died on a missing/changed ``hud`` symbol (a real
                    compat-surface gap)
- ``third-party``   import died on a non-hud dependency missing from this venv
                    (not a compat signal; the env's image would provide it)
- ``error``         anything else (syntax, package-context, runtime at import)

It also aggregates every ``DeprecationWarning`` the shims emitted, split into
redirects (working compat) and no-op/marker hits (symbols that resolve to
stand-ins — the candidates for proper capability routing).

Usage:
    uv run python scripts/v5_compat_report.py [corpus_dir]
    uv run python scripts/v5_compat_report.py --probe path/to/env.py  # internal
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROBE_TIMEOUT_S = 60

# Directories that are never env definitions: vendored SDKs, venvs, docs, tests.
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "docs",
    "tests",
    "test",
    "hud",  # vendored copies of the hud SDK inside env repos
}

ENV_DEF_RE = re.compile(r"=\s*(?:hud\.)?(?:Environment|MCPServer)\(")


# ─── probe (runs in a subprocess, prints one JSON object) ──────────────────


def probe(path: str) -> dict[str, object]:
    import warnings

    file = Path(path).resolve()
    # Help src-layout repos resolve their own packages.
    repo = _repo_root(file)
    for extra in (repo, repo / "src", file.parent):
        if extra.is_dir() and str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    from hud.utils.modules import load_module

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            module = load_module(file)
        except ModuleNotFoundError as exc:
            kind = "hud-gap" if (exc.name or "").split(".")[0] == "hud" else "third-party"
            return {"status": kind, "error": str(exc), "warnings": _messages(caught)}
        except ImportError as exc:
            kind = "hud-gap" if "hud" in str(exc) else "error"
            return {"status": kind, "error": str(exc), "warnings": _messages(caught)}
        except BaseException as exc:  # report, don't crash the harness
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "warnings": _messages(caught),
            }

    from hud.environment import Environment

    envs = [
        {
            "name": value.name,
            "tasks": len(value.tasks),
            "capabilities": [type(c).__name__ for c in value.capabilities],
            "legacy_tools": len(getattr(value, "_legacy_tools", [])),
        }
        for value in vars(module).values()
        if isinstance(value, Environment)
    ]
    return {"status": "ok", "envs": envs, "warnings": _messages(caught)}


def _messages(caught: list[object]) -> list[str]:
    return [
        str(w.message)  # type: ignore[attr-defined]
        for w in caught
        if issubclass(w.category, DeprecationWarning)  # type: ignore[attr-defined]
    ]


def _repo_root(file: Path) -> Path:
    for parent in file.parents:
        if (parent / "pyproject.toml").exists() or (parent / "Dockerfile.hud").exists():
            return parent
    return file.parent


# ─── discovery + report (parent process) ────────────────────────────────────


def find_candidates(corpus: Path) -> dict[str, list[Path]]:
    """Repo name -> files that define an Environment/MCPServer."""
    by_repo: dict[str, list[Path]] = {}
    for repo in sorted(p for p in corpus.iterdir() if p.is_dir()):
        files = []
        for py in repo.rglob("*.py"):
            if EXCLUDED_DIR_NAMES & set(py.relative_to(repo).parts[:-1]):
                continue
            try:
                text = py.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if ENV_DEF_RE.search(text):
                files.append(py)
        if files:
            by_repo[repo.name] = sorted(files)
    return by_repo


def run_probe(file: Path) -> dict[str, object]:
    cmd = [sys.executable, __file__, "--probe", str(file)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"import exceeded {PROBE_TIMEOUT_S}s"}
    if proc.returncode != 0 or not proc.stdout.strip():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return {"status": "crash", "error": " | ".join(tail)}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def classify_warning(msg: str) -> str:
    if "no-op" in msg:
        return "no-op"
    if "marker" in msg:
        return "computer-marker"
    if "moved to" in msg:
        return "redirect"
    return "other"


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--probe":
        print(json.dumps(probe(sys.argv[2])))
        return 0

    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("environments")
    if not corpus.is_dir():
        print(f"corpus dir not found: {corpus}", file=sys.stderr)
        return 1

    candidates = find_candidates(corpus)
    status_counts: Counter[str] = Counter()
    warning_kinds: Counter[str] = Counter()
    noop_messages: Counter[str] = Counter()
    gaps: Counter[str] = Counter()

    for repo, files in candidates.items():
        print(f"\n=== {repo} ({len(files)} candidate file(s))")
        for file in files:
            result = run_probe(file)
            status = str(result.get("status"))
            status_counts[status] += 1
            rel = file.relative_to(corpus)
            if status == "ok":
                envs = result.get("envs") or []
                desc = "; ".join(
                    f"{e['name']}: {e['tasks']} task(s), caps={e['capabilities']},"
                    f" legacy_tools={e['legacy_tools']}"
                    for e in envs  # type: ignore[union-attr]
                )
                print(f"  [ok]          {rel} -> {desc or 'no Environment instance'}")
            else:
                print(f"  [{status:<11}] {rel} -> {result.get('error')}")
                if status == "hud-gap":
                    gaps[str(result.get("error"))] += 1
            for msg in result.get("warnings") or []:  # type: ignore[union-attr]
                kind = classify_warning(str(msg))
                warning_kinds[kind] += 1
                if kind in ("no-op", "computer-marker"):
                    noop_messages[str(msg).split(" (")[0]] += 1

    print("\n=== Summary")
    for status, count in status_counts.most_common():
        print(f"  {status:<12} {count}")
    print("\n=== Shim warnings by kind")
    for kind, count in warning_kinds.most_common():
        print(f"  {kind:<16} {count}")
    if noop_messages:
        print("\n=== No-op / marker hits (capability-routing candidates)")
        for msg, count in noop_messages.most_common():
            print(f"  {count:>3}x {msg}")
    if gaps:
        print("\n=== hud import gaps (real compat-surface breaks)")
        for msg, count in gaps.most_common():
            print(f"  {count:>3}x {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
