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
  ruff, ty, pytest, and coverage configuration.
- Source files and their tests for exact behavior. Trust code and tests over
  stale prose.
- `cookbooks/` for runnable end-to-end examples (each is its own uv project).

Keep this file stable. Do not turn it into a release runbook, command matrix, or
inventory of current incidents.

## Repository Map

- Core flow: `hud/environment/` (spec: capabilities, tasks, serving) →
  `hud/eval/` (engine: rollout, runtimes, jobs) → `hud/agents/` (harnesses),
  connected by `hud/capabilities/` and `hud/clients/`.
- `hud/cli/` is the Typer surface over the same modules.
- `hud/integrations/` contains packaged adapters for external task formats.
- `hud/_legacy.py` and `hud/patches/` quarantine v5 compatibility.
- `tests/` mirrors the `hud` package layout: tests for `hud/cli/` live in
  `tests/cli/`, tests for top-level modules in `tests/` itself.
- `cookbooks/` contains standalone runnable examples outside the `hud` package.

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
uv run ruff format . --check
uv run ruff check .
uv run --extra dev --extra train ty check
uv run python scripts/check_tests.py
uv run pytest -n auto
```

Locally, ruff and ty are the loop; CI runs the test suite. Run the tests you
touched, and leave the full suite and the `e2e` lanes to CI.

Tests run on Python 3.11 and 3.12 in CI. `pyproject.toml` currently supports
Python `>=3.11, <3.13`.

## Code Quality Bar

- Prefer direct, typed, maintainable code over clever or magical abstractions.
- Fail fast and loudly. Avoid silent fallbacks, broad exception swallowing, and
  defensive branches that hide broken invariants.
- Minimize branching. Every new `if`, `try`, compatibility path, or nullable mode
  should earn its keep.
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

## Architecture And Simplification

- Before changing cross-module behavior, identify the layer that owns the concept.
  Keep its policy and invariants there; other layers should translate or delegate.
- Before introducing an abstraction, find the existing abstraction that owns the
  same semantics. Reuse or extend it instead of building a partial copy elsewhere.
- Reuse abstractions by meaning, not code similarity. Duplication is a reason to
  inspect ownership, not automatic justification for a shared helper.
- Prefer one canonical contract, data model, and execution path for each concept.
  Express legitimate variation through owned configuration or policy rather than
  parallel implementations.
- Keep adapters thin: translate external representations into native SDK concepts
  at the boundary, then use the normal implementation.
- Add a helper, wrapper, or module only when it reduces total conceptual
  complexity, centralizes an invariant, or provides meaningful reuse. Inline
  ceremonial indirection.
- Public APIs should expose stable user operations and domain concepts, not
  internal construction stages or compatibility scaffolding.
- Preserve observable behavior, persisted contracts, and security invariants, not
  accidental internal structure. Prefer explicit migrations when architecture
  requires a shipped contract to change.

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

## Testing

Tests are black-box: they drive the SDK through a public boundary and fake only
what lies outside it. The harness in `tests/harness` holds those fakes: the HUD
services, scripted model providers, a `docker` executable, a VNC screen, and the
real `hud` CLI run in a subprocess. `CONTRIBUTING.md` shows how to use it.

- Enter through a public boundary: the `hud` CLI, an exported name, the control
  channel or another wire protocol, a file format, or a process effect.
- Fake only the outside world: HUD services, model providers, Docker, cloud SDKs,
  the clock. Never patch a `hud` name, import a private one, or touch a private
  attribute; `scripts/check_tests.py` enforces this in CI. Configure the SDK the
  way a user does, through environment variables (`hud_env.set(...)`).
- Never hand-build what a producer emits. Run the producer, so the test covers
  the seam between it and its consumer.
- Assert what a user or a peer observes: the reward, run and trace fields,
  output and exit codes, files, request bodies, wire frames. Prefer fixture
  environments whose reward proves the path ran.
- Write a scenario once per contract, as a table; a new case is a new row. Use
  inline snapshots only where a whole output is the contract: request bodies the
  platform parses, `--json` documents, docker and cloud-SDK transcripts.
- Do not add tests to check your own change. Check it with a temporary script,
  and extend a scenario only when the change adds or fixes a contract; a bug fix
  lands with the scenario row that failed before it.
- Tests needing something outside the process are marked `e2e` plus `docker`,
  `sandbox`, `live` (a real HUD account) or `hosted`, and run only when selected.
- Test names describe the observable behavior, not the mechanism.

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
