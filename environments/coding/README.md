# Coding Environment

This environment gives the agent a git repository through an `ssh` workspace and grades its
changes with hidden tests. `hud init` copies this directory as a complete environment project.

## Run the example tasks

Start with `sentry-fix`: fix `get_user_profile` so missing profiles fall back to the
user's name while preserving complete profiles. Its six behavior cases run in
both orders, checking the result and that inputs stay unchanged (12 checks).
This is the existing HUD profile-crash example, served through the same repository
workspace and hidden-test lifecycle as the advanced tasks.

The included advanced `flask-4992` and `flask-5063` tasks are instances from
[SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite). Their repository
baselines and reference fixes are included in `repository.bundle`; `tasks.json` contains their prompts,
test patches, test commands, and expected results. These same task definitions are used by
`hud eval` and taskset sync; no separate featured copy is maintained.

```bash
uv sync
hud set HUD_API_KEY=your-key-here
hud eval tasks.py claude --task-ids sentry-fix -y --runtime local
```

## Adapt the environment

- Set `REPO_URL` or replace `repository.bundle` with the repository to grade.
- Define task rows in `tasks.json`: the prompt, baseline ref, hidden test patch, test path and command,
  and selected fail-to-pass and pass-to-pass JUnit IDs.
- Install repository dependencies in `Dockerfile.hud` so grading does not depend on runtime
  downloads.
- Keep hidden tests and reference fixes outside the baseline exposed to the agent.

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
JUnit scoring, and all three bundled reference fixes.

See the [coding-agent cookbook](https://docs.hud.ai/v6/cookbooks/coding-agent) for the grading
lifecycle and a task-authoring walkthrough.
