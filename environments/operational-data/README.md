# Operational data

One environment, three task templates, five portable examples. Agents work with
fictional Slack, Jira and calendar exports through MCP tools, or review attached
documents in an isolated shell. The templates share their workspace, source
adapters and grading lifecycle.

| Template | Inputs | Bundled tasks |
| --- | --- | --- |
| `assist` | Slack, Jira, calendars and attachments | `board-review-scheduling`, `march-close-status`, `ap-payment-run` |
| `review` | Contracts, invoices, payments and credits | `supplier-settlement-review` |
| `plan` | Inventory, policy, briefing and emails | `launch-dispatch-plan` |

## Run

```bash
uv sync --upgrade-package hud
uv run hud eval tasks.py claude --runtime local -y
uv run hud eval tasks.py claude --runtime local --full -y
```

The first command runs the first task; `--full` runs all five. Select a case with
`--task-ids supplier-settlement-review`. The defaults are Claude Sonnet 5.5 and
60 steps. `tasks.json` is the canonical task data; `tasks.py` binds every row to
this environment, including when `hud init` gives it a new name.

All five examples use bundled synthetic learner inputs. Loading and staging them
requires no uploaded file IDs or account credentials. Set `HUD_API_KEY` for
model calls and grading. Uploaded custom inputs also use the runtime credential;
the hidden `hud_api_key` argument enables its injection in hosted rollouts.

Local shell execution requires Linux, bubblewrap, a writable `/workspace` and
permission to run the shell as UID 1000. Use the Docker image on a Linux runtime
when those requirements are unavailable locally.

## SDK source

`pyproject.toml` installs the HUD SDK from the latest `main`. Update an existing
lock with `uv sync --upgrade-package hud`. To test an exact checkout wheel,
build or copy it into `.hud-sdk/` and replace the source before syncing:

```bash
uv add --no-sync .hud-sdk/hud-*.whl
uv sync
```

Docker refreshes the `main` source when building. If `.hud-sdk/` contains exactly
one `hud-*.whl`, it installs that wheel instead; multiple SDK wheels fail the
build. No SDK wheel is bundled with the example.

## Author a task

The three templates keep the prompt and weighted rubric editable. Their typed
upload groups expose file pickers suited to each workflow: supplier and dispatch
authors do not have to fill empty Slack or Jira slots. `assist` accepts Slack
ZIP/JSON, Jira CSV/Excel/JSON and calendar ICS exports; these rebuild simulated
systems in memory. Tools modify those simulations and record an action log.

Use `attachments` on `assist`, or the business-specific file groups on `review`
and `plan`, for uploaded HUD data-file references. The optional `bundled`
argument selects paths under `samples/`; only its `attachments` enter the
workspace. Source exports stay behind the tools. Bundled and uploaded data can
be combined; filename collisions fail explicitly.

`as_of` fixes the scenario date. `deliverables` names files to save under
`/workspace/outputs`; attachments appear under `/workspace/files`. Every task
starts with an empty workspace, and one task can occupy it at a time. The agent
shell runs as an unprivileged user without the server's credentials or network.
Its system Python includes openpyxl, python-docx, python-pptx and pypdf.

## Grading

The grader receives the final answer, recorded tool changes, and requested saved
files. Original attachments are snapshotted as supporting evidence before the
agent starts; they can verify claims but cannot supply missing work. Saved
supplier and dispatch workbooks also undergo deterministic checks from the
approved task criteria. Narrative criteria use Sonnet 5.5 with a 16,384-token
response budget. Scores retain the rubric's weights and negative penalties.

Deliverables are read through private snapshots without following symlinks.
Limits are 16 MiB per file, 64 MiB of expanded Office XML, 256 KiB of rendered
text and one million workbook cells. Missing or unreadable deliverables remain
visible to grading. Judge failures raise errors.

Only learner inputs are bundled. The approved task prompts, criteria and source
files are preserved; no answer workbooks or authoring scripts are included.
The connector dataset was adapted from
[hud-evals/operational-data-template](https://github.com/hud-evals/operational-data-template/tree/fa88fac90bdd1538dc6626629de7f61317374f47).

## Tests

```bash
uv run pytest -q
```

Tests exercise export parsers, MCP tools, upload staging, document reads, task
binding and the shared lifecycle. Provider requests and model grading are mocked.
