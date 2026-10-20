# File analysis: operational-data tasks

Start with a concrete investigation, then adapt its files, prompt, and rubric.
This template gives an agent a shell with network access disabled and asks it
to write a report. The console exposes `prompt` as text, `attachments` as an
upload picker, and `criteria` as a weighted grading table.

| Task | Inputs | Saved deliverable |
|---|---|---|
| Reconcile supplier payments | Invoices, bank ledger, receiving log, signed terms | `reconciliation-report.md` |
| Prioritize a support queue | Tickets, accounts, verified incident notes, SLA policy | `support-priorities.md` |
| Synthesize onboarding feedback | Usability sessions, participant outcomes, event log, research brief | `onboarding-feedback.md` |

## Run and deploy

```bash
hud init my-file-analysis --template file-analysis
cd my-file-analysis
uv sync
# Configure HUD_API_KEY for the agent and the rubric judge.
uv run hud eval tasks.py claude --runtime docker --all --max-steps 40
uv run hud deploy
uv run hud sync tasks operational-data tasks.py
```

The Docker run needs Docker running. On Linux with usable bubblewrap namespaces,
`--runtime local` also works. A native macOS shell cannot satisfy the required
isolation.

The bundled inputs need no upload, data-file IDs, or network credentials to read.
The agent and `LLMJudgeGrader` use the configured HUD inference credential.
Each positive criterion is judged as met/unmet by `claude-sonnet-5-5`; its
weight controls its share of the final score. Missing or empty reports score
zero. These are teaching/demo tasks, not a calibrated difficulty benchmark.

## Customize one task

1. Put source documents in its `fixtures/` subdirectory. These are the only
   bundled files copied into `/workspace/files`; never put expected answers,
   rubric files, or author notes there.
2. Edit the corresponding row in `tasks.json`: the author prompt, evidence-based
   criteria and weights, stable slug, and agent step budget. `tasks.py` binds
   those rows to the local environment; it is the source for evaluation and sync.
   The JSON and Python entry point contain one shared set of task definitions.
3. Keep each requested deliverable aligned with the rubric. Include facts the
   judge needs to verify calculations and source citations. Rubrics live outside
   the solver workspace and are never included in the agent prompt.
4. Run the task, inspect its report and subscores, then deploy and sync again.
   The generic `review_files` template accepts your own uploaded files and a
   root-level `report_path`. Optional uploads can supplement bundled inputs;
   they cannot overwrite their filenames.

The legacy `reconcile_supplier_payments` template ID and
`reconcile-supplier-payment-batch` task slug are retained for existing tasksets.
`argument-hints` is an init alias for this template rather than a second copy.

## Isolation and limitations

Hosted Linux containers use a UID-1000 solver, a protected `/hud` source tree,
and a separate workspace. The Dockerfile includes bubblewrap for isolation.
Workspace startup requires usable bubblewrap isolation and fails closed when
it is unavailable. Use the container runtime on hosts without Linux namespace
support; a native macOS run cannot provide this sandbox. Saved reports must be regular UTF-8 files up to 256 KiB;
symlinks are rejected. Grading uses the report, not the agent's final message.

Tests mock the inference boundary; they check staging, report selection,
weighted-score plumbing and source/rubric alignment without paid model calls.
They do not measure the judge's accuracy or establish score spread.

## Data provenance

All bundled companies, participants, invoices, events, policies, amounts and
quotations are fictional examples authored for HUD. The supplier fixtures are
retained from HUD's existing featured example; the support and usability packs
are newly authored. No customer exports, recordings, proprietary task packs,
or third-party benchmark questions are included.
