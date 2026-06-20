# HUD Python Agent Guide

This repository is the Python SDK and CLI for HUD: environments, capabilities,
tasks, agents, the rollout engine, telemetry, and command-line workflows for
building and running agent evaluations.

Priorities: solve the requested problem, keep scope tight, preserve public SDK
behavior where it is actually shipped, and improve code quality rather than
adding local workarounds.

## Where To Look First

- `README.md` for the protocol, product concepts, and common CLI workflows.
- `docs/v6/` for the live SDK docs: quickstart, reference (environment, tasks,
  capabilities, agents, graders, types, cli), run guides, and cookbooks.
- `CONTRIBUTING.md` for setup, test, lint, and type-check commands.
- `pyproject.toml` for supported Python versions, dependencies, optional extras,
  ruff, pyright, pytest, and coverage configuration.
- Source files and colocated tests for exact behavior. Trust code and tests over
  stale prose.
- `cookbooks/` for runnable end-to-end examples (each is its own uv project).

Keep this file stable. Do not turn it into a release runbook, command matrix, or
inventory of current incidents.

## Repository Map

- Core flow: `hud/environment/` (spec: capabilities, tasks, serving) →
  `hud/eval/` (engine: rollout, runtimes, jobs) → `hud/agents/` (harnesses),
  connected by `hud/capabilities/` and `hud/clients/`.
- `hud/cli/` is the Typer surface over the same modules.
- `hud/_legacy.py` and `hud/patches/` quarantine v5 compatibility.
- `cookbooks/` and `integrations/` live outside the `hud` package.

## Working Style

- Run commands from the repository root unless a tool explicitly requires a
  subdirectory.
- Use `uv` for Python commands. Do not rely on an activated virtualenv.
- Read files before editing them and follow nearby patterns.
- Keep edits focused on the requested behavior. Do not clean up unrelated code.
- Prefer editing existing docs over creating new docs unless the user asks for a
  new document.
- Do not introduce hacks, monkey patches, or partial workarounds. If a robust
  solution needs missing support, add that support cleanly or report the blocker.
- Report any part of a change that is uncertain, fragile, or intentionally left
  unverified.

## Setup And Checks

Use the commands in `CONTRIBUTING.md` as the source of truth. Common commands:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff format . --check
uv run ruff check .
uv run pyright
```

The shared pre-push hook lives in `.githooks/pre-push`, but agents should not
change local git config unless explicitly asked.

Tests run on Python 3.11 and 3.12 in CI. `pyproject.toml` currently supports
Python `>=3.11, <3.13`.

## Code Quality Bar

- Prefer direct, typed, maintainable code over clever or magical abstractions.
- Be ambitious about simplification. Look for ways to delete whole branches,
  helper layers, modes, and special cases while preserving behavior.
- Fail fast and loudly. Avoid silent fallbacks, broad exception swallowing, and
  defensive branches that hide broken invariants.
- Minimize branching. Every new `if`, `try`, compatibility path, or nullable mode
  should earn its keep.
- Preserve documented public API and persisted behavior unless the task is an
  intentional migration. Do not add compatibility layers for unshipped branch
  work; replace the design cleanly.
- Reuse canonical helpers and local abstractions before adding new ones.
- Keep feature logic in the layer that owns the concept. Treat scattered
  feature checks in shared paths as a design problem.
- Prefer explicit contracts over optional, loosely shaped, or cast-heavy data.
- Delete dead code. Do not keep obsolete paths around "just in case."
- Keep comments rare and useful. Explain non-obvious intent, not what the next
  line mechanically does.
- Remove AI-generated slop before finishing: unnecessary comments, abnormal
  defensive checks, broad `try` blocks, type bypasses, deep nesting, and thin
  wrappers that do not reduce real complexity.
- Be suspicious of files pushed past 1000 lines. Decompose when there is a clear
  focused module to extract.
- Avoid new core dependencies. If a dependency is only needed for optional
  provider, tool, or integration behavior, put it behind the relevant extra.

## Typing And Imports

- Type public APIs and cross-module contracts. Prefer explicit Pydantic models or
  typed structures over ad-hoc dictionaries at boundaries.
- `cast(...)` and `assert ...` are acceptable for real type narrowing. Broad
  `# type: ignore` comments are not.
- Keep `Any` contained to genuinely dynamic payloads such as provider JSON,
  metadata, or third-party integration blobs.
- Keep imports at the top of the module. Use inline imports only for an existing
  lazy optional-dependency pattern or a documented circular-import constraint.
- Use `TYPE_CHECKING` imports for type-only imports that would otherwise add
  runtime dependency cost or cycles.

## Testing Expectations

- Add or update focused tests for behavior changes. Put tests near the module
  they cover, following the existing `*/tests/` layout.
- Test behavior and contracts, not private implementation details.
- Regression tests should fail on the old behavior through the normal lifecycle
  or public boundary. Do not manually seed private state such as internal maps,
  caches, cursors, or prepared containers just to prove a changed line.
- If a bug involves internal state, reach it through real setup and execution:
  construction, configuration, preparation, run loop, provider response, tool
  execution, or public API call.
- Do not add hooks, helper methods, or abstraction layers only to make tests
  easier. If a test needs that, reconsider the behavior boundary instead.
- Test names should describe the observable behavior or contract, not the
  private mechanism.
- Mock external services, provider APIs, network, Docker, browser, and filesystem
  boundaries as needed. Do not mock core logic just to make a test easy.
- Mark tests that require `HUD_API_KEY`, network access, or deployed services as
  integration tests.
- Run the narrowest relevant tests first, then broader checks when the blast
  radius is shared or user-facing.

## CLI Inspection Commands

`hud trace <trace_id>` — inspect a rollout. Reads local JSONL first
(`HUD_TELEMETRY_LOCAL_DIR/<trace_id>.jsonl`), falls back to
`GET /v2/trace/{id}/events`. Code: `hud/cli/trace.py`.

`hud jobs` — list recent jobs (`GET /v2/jobs`). `hud jobs <id>` — list traces
for a job (`GET /v2/jobs/{id}/traces`). Both responses return `{items: [...]}`;
the list is extracted with `data if isinstance(data, list) else (data.get("items") or [])`.
Code: `hud/cli/jobs.py`.

## Operational Debugging

- Follow the execution path instead of guessing from abstractions.
- For CLI issues, start with the command module, then config/settings, then the
  SDK module being exercised.
- For agent/provider issues, inspect gateway resolution, provider adapter code,
  capability-backed tool wiring, and recorded request/response shapes.
- For environment/task issues, inspect the task lifecycle (start/grade), the
  control-channel server and client, and capability routing/tunneling.
- For execution issues, inspect the rollout engine: runtime provider
  acquisition, `connect`, the `Run` lifecycle, and job/trace reporting.
- For telemetry issues, inspect instrumentation boundaries and exporter behavior
  before changing call sites.
- Report what was verified, what remains inferred, and which file, test, trace,
  or command output supports the conclusion.

## Decision Protocol

Ask first when scope, public API compatibility, or ownership is unclear.

Choose and flag when naming, test boundaries, or local structure are ambiguous
but the direction is straightforward.

Just do it when fixing formatting, applying an obvious bug fix with clear root
cause, tightening types, or removing slop that does not change behavior.
