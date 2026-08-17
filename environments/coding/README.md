# Coding Environment

This environment gives the agent a git repository through an `ssh` workspace and grades its
changes with hidden tests. `hud init` copies this directory as a complete environment project.

## Run the example task

The included `flask-4992` task is the `pallets__flask-4992` instance from
[SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite). Its repository
baseline, hidden tests, and reference fix are included in `flask-4992.bundle`.

```bash
uv sync
hud set HUD_API_KEY=your-key-here
hud eval tasks.py claude --task-ids flask-4992 -y --runtime local
```

## How grading works

1. Setup checks out `base_ref`, snapshots the prepared repository, and moves its original
   `.git` directory outside the workspace.
2. The agent receives a fresh repository containing one baseline commit. The original history,
   hidden-test refs, and reference fix are not reachable from the workspace.
3. Grading discards the agent-controlled Git metadata, restores the original history, and captures
   the final worktree against the setup snapshot.
4. The submitted diff is reapplied, then `test_files` are checked out from `test_ref`.
5. `test_script` writes JUnit XML to `{junit_path}`. Only the configured fail-to-pass and
   pass-to-pass test IDs contribute to the score.

The default score is the fraction of selected tests that pass. Set `use_binary_score=True` to
require every selected test. Missing selected tests count as failures.

```python
from env import coding_task

fix_parser = coding_task(
    description="Fix the parser without breaking existing inputs.",
    test_script="python -m pytest -q {test_files} --junitxml={junit_path}",
    base_ref="origin/parser_baseline",
    test_ref="origin/parser_test",
    golden_ref="origin/parser_golden",
    test_files=["test_parser.py"],
    f2p_test_nodeids=["test_parser.TestParser.test_new_input"],
    p2p_test_nodeids=["test_parser.TestParser.test_existing_input"],
)
fix_parser.slug = "fix-parser"
```

JUnit IDs are `classname.name`, matching the values written in the report. For pytest, inspect a
generated report instead of copying pytest's slash-and-`::` collection syntax.

## Adapt the environment

- Set `REPO_URL` to use another repository for local runs.
- Build the bundled repository with `docker build -f Dockerfile.hud .`.
- Replace the bundle and task refs when adapting the environment to another repository.
- Install the repository's dependencies in `Dockerfile.hud` so grading does not depend on runtime
  downloads.
- Define task rows in `tasks.py`. Keep hidden tests and reference fixes outside the baseline
  history exposed to the agent.

The image runs agent shells as UID 1000. `/hud` contains the environment code, repository vault,
and grading logs and is readable only by the environment process. The workspace requests network
isolation when bubblewrap is available. The provided image does not install bubblewrap, so container
runtimes must enforce any required egress policy. Non-root local runs require usable bubblewrap and
fail closed without it.

## Tests

```bash
uv run pytest tests/ -q
```

The suite covers repository isolation, trusted diff capture, prepared dependencies, hidden-test
restoration, reference-fix validation, and partial and binary JUnit scoring.
