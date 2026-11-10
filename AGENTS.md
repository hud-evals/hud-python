# HUD Python SDK

This repository is the Python SDK and CLI for HUD: environments, capabilities,
tasks, agents, the rollout engine, telemetry, and the command-line workflows
for building and running agent evaluations. Customers build environments on it
and pin released versions, so its public API and CLI are what they program
against.

The bar is production-grade code: one clear design per problem, no hacks, no
monkey patches, no partial solutions. When the only way to ship a request is a
hack, add the missing support cleanly, or stop and say what the SDK lacks.

## Contracts

These hold absolutely. If a change cannot satisfy one, stop and say so.

- **Public repository.** Tracked files, commits, issues, and PR descriptions
  carry no internal hostnames, account or team IDs, customer names, runbooks,
  or operational details.
- **Shipped behavior.** Preserve observable behavior, persisted contracts, and
  security invariants of released versions. When the design requires a shipped
  contract to change, migrate it explicitly. Accidental internal structure is
  not a contract.
- **Compatibility is for real consumers.** Keep an old interface only for a
  consumer outside the change, such as a released API or CLI flag. Private and
  unreleased code cuts over in one step, with no shims or aliases left behind.

## Agent protocol

- **Verify, don't guess.** Follow the execution path instead of guessing from
  abstractions (see **Debugging**). In reports, label what you observed and
  what you inferred, and cite the file, test, trace, or command output.
- **Ask, flag, or do.** Ask first when scope, public API compatibility, or
  ownership is unclear. Choose and flag when naming, test boundaries, or local
  structure are ambiguous but the direction is clear. Just do formatting,
  obvious bug fixes with a clear root cause, tightened types, and slop removal
  that does not change behavior.
- **Scope.** Keep edits to the requested behavior and follow nearby patterns.
  Do not clean up unrelated code. Prefer editing existing docs to creating new
  ones unless asked.
- **Final report.** End each task with what was verified and how, anything you
  are not confident in or that could be fragile, and a `Friction:` line.

### Friction

Agents are among HUD's heaviest users, so what they work around is product
feedback. In the `Friction:` line, list each place the SDK, CLI, or platform
made you work around it: what you tried, what you expected, what happened, and
the workaround. Examples are reading source to learn a flag or behavior the
docs and `--help` don't state, retrying past an error that didn't name its
cause, calling an API by hand because the CLI can't do it, and a recorded value
that doesn't match what ran. Write `Friction: none` when there was none. Offer
to file each item as a GitHub issue, following **Public repository**, and file
only when the user agrees.

## Principles

- **Question the need.** A new module, option, stage, or dependency earns its
  place: name the simpler design and why it fails. A flag for behavior that
  should always happen is a design bug. Optional provider, tool, or integration
  dependencies go behind their extra.
- **One way.** Each concept has one contract, one data model, one execution
  path, one name. Find the layer that owns a concept and keep its policy
  there; adapters translate external formats into native SDK concepts at the
  boundary and then use the normal path. Reuse by meaning, not code
  similarity: a helper earns its place by centralizing an invariant, not by
  sharing lines.
- **Fail loud.** Surface failures with their cause. No silent fallbacks, broad
  exception swallowing, or defensive branches that hide broken invariants.
  Errors you write name the failing input and the remedy.
- **Root cause first.** Find the exact cause from evidence before changing
  code, and fix it at the boundary where the fault is. Retries, longer
  timeouts, and special-case paths are workarounds unless the state they
  handle is legitimate.
- **Not narrow.** A defect or review comment is one instance of a rule. Fix
  every path the same rule covers.
- **Does it run.** Prove a change on the real path, smallest real surface
  first. A build, a type check, or a mocked test is not proof.
- **No fake tests.** Write a test only when it would catch a break a user
  would notice. Drive the SDK as a user does (CLI, public API) and assert what
  they see: reward, status, errors, files, exit codes, and what the platform
  records, not prompt text, rendered output, or argv. Fake only what CI cannot
  run (HUD services, model providers, Docker), and never touch `hud`
  internals; ruff's SLF001 enforces this. A change needs no test by default;
  a regression test must fail with the fix reverted.

Before calling a change done, check its diff:

- Does it add a second way to do something that already has one?
- Does it add a public knob, wrapper, or layer that hides nothing?
- Does any failure path return a normal-looking result?
- Does a compatibility path survive without a named external consumer?
- Did the real changed path run, and which test fails if it breaks?

## Code

- Type public APIs and cross-module contracts with Pydantic models or typed
  structures, not ad-hoc dictionaries. Keep `Any` to genuinely dynamic
  payloads such as provider JSON. `cast(...)` and `assert` are fine for real
  narrowing; broad `# type: ignore` is not.
- Imports at the top of the module, except an existing lazy
  optional-dependency pattern or a documented import cycle. Use
  `TYPE_CHECKING` for type-only imports that would add runtime cost or cycles.
- Public APIs expose user operations and domain concepts, not construction
  stages or compatibility scaffolding.
- Every new `if`, `try`, nullable mode, or compatibility path earns its keep.
  Before finishing, remove slop: obvious comments, abnormal defensive checks,
  broad `try` blocks, type bypasses, deep nesting, and thin wrappers.
- Comments are rare and explain non-obvious intent. Delete dead code. Decompose
  a file past 1000 lines when a focused module is there to extract.
- Use `uv` for Python commands; do not rely on an activated virtualenv. Run
  commands from the repository root.

## Checks

`CONTRIBUTING.md` holds the commands. Locally, ruff and ty are the loop; CI
runs the test suite and the `e2e` lanes. Run the tests you touched.

```bash
uv sync --all-extras
uv run ruff format . --check
uv run ruff check .
uv run ty check
```

## Repository map

- Core flow: `hud/environment/` (spec: capabilities, tasks, serving) →
  `hud/eval/` (engine: rollout, runtimes, jobs) → `hud/agents/` (harnesses),
  connected by `hud/capabilities/` and `hud/clients/`.
- `hud/cli/` is the Typer surface over the same modules.
- `hud/integrations/` holds adapters for external task formats.
- `tests/` mirrors the SDK's subsystems; `tests/fixtures` holds fixture
  environments and fakes for what CI cannot reach.
- `cookbooks/` holds standalone runnable examples, each its own uv project.
- `README.md` covers the protocol and product concepts, `docs/v6/` the live SDK
  docs, and `pyproject.toml` supported Pythons, extras, and tool config.
  Source and tests beat stale prose.

## Debugging

- CLI: the command module, then config and settings, then the SDK module it
  exercises.
- Agents and providers: gateway resolution, the provider adapter,
  capability-backed tool wiring, and recorded request and response shapes.
- Environments and tasks: the task lifecycle (start, grade), the
  control-channel server and client, and capability routing and tunneling.
- Execution: the rollout engine, from runtime acquisition and `connect` through
  the `Run` lifecycle to job and trace reporting.
- Telemetry: instrumentation boundaries and exporter behavior before call
  sites.

Keep this file stable: no release runbooks, command matrices, or current
incidents.
