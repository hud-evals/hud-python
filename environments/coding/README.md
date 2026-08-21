# Coding Environment

This environment gives the agent a git repository through an `ssh` workspace and grades its
changes with hidden tests. `hud init` copies this directory as a complete environment project.

## Run the example tasks

The included `flask-4992` and `flask-5063` tasks are instances from
[SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite). Their repository
baselines and reference fixes are included in `flask.bundle`; `tasks.py` contains their prompts,
test patches, test commands, and expected results.

```bash
uv sync
hud set HUD_API_KEY=your-key-here
hud eval tasks.py claude --task-ids flask-4992,flask-5063 -y --runtime local
```

## How grading works

1. Setup clones `base_ref` into an environment-owned baseline outside the workspace.
2. The clean worktree is copied into the workspace and initialized as a one-commit repository, so
   the source history and reference fixes are never exposed to the agent.
3. Grading terminates the agent's isolated session namespace and discards its Git metadata.
4. `test_path` is restored to the baseline, then `test_patch` is applied.
5. The custom `JUnitGrader` runs `test_command` through `BashGrader` under the workspace's isolated
   UID, then scores the selected fail-to-pass and pass-to-pass test IDs.

```python
from env import coding_task

fix_parser = coding_task(
    description="Fix the parser without breaking existing inputs.",
    test_command="python -m pytest -q test_parser.py --junitxml={junit_path}",
    test_patch="""diff --git a/test_parser.py b/test_parser.py
--- a/test_parser.py
+++ b/test_parser.py
@@ -1,3 +1,6 @@
 def test_existing_input():
     assert parse("old")
+
+def test_new_input():
+    assert parse("new")
""",
    base_ref="origin/parser_baseline",
    test_path="test_parser.py",
    fail_to_pass=["test_parser.TestParser.test_new_input"],
    pass_to_pass=["test_parser.TestParser.test_existing_input"],
)
fix_parser.slug = "fix-parser"
```

The default reward is the fraction of selected tests that pass. Set `binary=True` to require every
selected test. Missing selected tests count as failures.

## Adapt the environment

- Set `REPO_URL` to use another repository for local runs.
- Build the bundled repository with `docker build -f Dockerfile.hud .`.
- Replace the bundle and task rows when adapting the environment to another repository or version.
- Install the repository's dependencies in `Dockerfile.hud` so grading does not depend on runtime
  downloads.
- Define task rows in `tasks.py`. Keep hidden tests and reference fixes outside the baseline exposed
  to the agent.

The image installs bubblewrap and runs agent shells as UID 1000. `/hud` contains the environment
code and trusted baseline and is readable only by the environment process. The workspace fails
closed if it cannot create its filesystem, process, and network isolation. Direct Docker runs use
HUD's packaged seccomp profile and system-path settings on a host that allows unprivileged user
namespaces.

## Tests

```bash
uv run pytest tests/ -q
```

The suite covers repository and session isolation, task reset, hidden-test application, selected
JUnit scoring, and both bundled reference fixes.
