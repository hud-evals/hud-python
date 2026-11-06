## Summary

**No, nothing equivalent exists.** In 0.6.22 a taskset is an in-memory `dict[str, Task]`; every loader (`.py`, JSON/JSONL, platform export) materializes all rows before anything runs, `Taskset.run` expands every row × group into coroutines up front, `hud sync tasks` ships every changed row in one `POST /tasks/upload`, and the platform's own batch launcher (`POST /rollouts/run_list`) takes lists of already-stored task ids. There is no dataset reference anywhere in the SDK or the REST API (the only "datasets" are a LeRobot *output* writer and the Harbor importer, which also materializes rows), no lazy/streaming task source, no pagination in the task fetch path, and no "parameter space" beyond the JSON schema `@env.template` publishes for its arguments. Three existing primitives are, however, the right building blocks: `@env.template` already separates the *definition* (a `_TaskFactory` with an args schema) from the *row* (`Task(env, id, args, slug)`), `POST /rollouts/submit` runs a task from its inline `(env, task, args)` with no stored row at all, and `Shared(provider, width=N)` already multiplexes N rollouts over one warm environment instance (the control channel keeps one suspended task per session, so a live env serves many rows). The proposal below (`hud.Batch`) composes those three into a platform object, adds the two things the measured pilot says matter most — per-row environment-startup amortization (20–56 s of a ~72 s rollout) and a one-shot non-agentic mode (one model call instead of ~6) — and is honest about what it cannot fix: at the 3.6 s/file target, 500M files is still ~57 years of serial model time, so the full corpus is a bulk-inference problem in which HUD's role is evaluation, calibration and hard-case routing, not the per-document loop.

## 1. Findings (evidence)

### 1.1 What exists, where, and its limits

| Looked for | What exists in 0.6.22 | Evidence | Limit / consequence |
|---|---|---|---|
| Dataset-backed or generator-backed taskset | None. `Taskset.__init__` builds `self.tasks: dict[str, Task]` from `list(tasks)`. | `eval/taskset.py:64-73`, `_index_by_slug` `:169-184` | Whole set in RAM; measured **2.34 KB per `Task`** (200k rows = 457 MB RSS). |
| Lazy / streaming task loading in `hud eval` | None. `Taskset.from_file` reads the whole JSON/JSONL (`_load_tasks_json`) or scans a module; `from_api` fetches the whole export. `hud eval` loads everything, then filters `--task-ids`, and runs **1 task unless `--all`/`--full`**. | `eval/taskset.py:76-107,145-166`; `cli/eval.py:384-412` | No cursor, no limit/offset in the SDK task path. |
| Iterator-friendly scheduler | `Taskset.run` expands `task × group` into a list, then `asyncio.gather(*(_one(t, gid) ...))`; `max_concurrent` is only a semaphore inside each coroutine. | `eval/taskset.py:326-329,384-385` | One coroutine per rollout exists before any runs; `job.runs` keeps every `Run`+`Trace` in memory. |
| Template parameter space / "task space" | `@env.template` registers a `_TaskFactory` holding the function `sig`; `manifest_entry()` publishes `args` as a JSON schema (`_args_json_schema`) so the platform validates stored args and renders forms. Calling the factory mints `Task(env=self.env.name, id=self.id, args=bound)`. | `environment/env.py:60-81,84-135,198-239` | Schema only; no enumeration/sampling/binding concept. |
| Task row schema | `Task(env, id, args, slug, validation, agent_config, columns, runtime_config, verifier)`. Default `slug = id + "-" + sha1(args)[:8]`. Wire identity on upload: `name`(slug), `env:{name}`, `task_id`. | `eval/task.py:30-72,45-59`; `eval/sync.py:185-197` | **32-bit default slug**: expected collisions ≈ n²/2³³ → 4.7 at 200k rows (reproduced: `Taskset()` raised `duplicate task slugs`), ≈415 at 1.888M, ≈2.9×10⁷ at 500M. Explicit slugs mandatory at scale. |
| `hud sync tasks` limits | Loads local set, fetches **entire** remote export for the diff (`--force` skips), then one `upload_taskset` call = one `POST /tasks/upload` with all `to_apply` rows. HTTP timeout 600 s. No chunking, no resume. | `cli/sync.py:209,232-239,282-288`; `eval/sync.py:165-182`; `utils/requests.py:35` | Measured 302 B/row upload JSON → 1.888M rows = **570 MB in one request**; 500M = 151 GB. |
| How hosted runs fetch tasks | SDK: `GET /tasksets/{id}/export` (unpaginated, every task's latest version). Platform: `GET /tasksets/{id}/tasks` is paginated (`limit` default 50, `offset`) but the SDK never uses it. Hosted rollout: `HostedRuntime._submit_and_await` posts `{trace_id, job_id, env, task, slug, args, agent, runtime_config?, verifier?}` to `POST /rollouts/submit` **per task** and polls `GET /trace/{id}` every `poll_interval=5 s`. | `eval/sync.py:148-159`; `eval/runtime/hosted.py:54,152-176,208-218`; REST API "Tasksets", "Rollouts" | A hosted rollout needs **no stored task row** ("remote rollouts carry the scenario inline", `eval/job.py:96`). Polling costs 12 requests/min per in-flight rollout against the 1,000 req/min member limit. |
| Job / batch APIs | `Job` (`id, name, runs, group, taskset_id`), `Job.start()` accumulates several `run()` calls under one id; reporting is `POST /trace/job/{id}/enter`, `/trace/{id}/enter`, `/trace/{id}/exit`. Platform: `POST /rollouts/run_list` "launch a batch of attempts as one job" takes `task_ids`/`task_version_ids`/`taskset_id`, `model_ids`, `group_size`, `max_concurrent`, `max_steps`. | `eval/job.py:33-157`; REST API "Rollouts → Launch" | Batch = list of **existing** task ids. Platform UI: "HUD creates the job and all of its traces at once"; "each attempt … gets its own environment instance" (platform/evaluations). |
| Pagination | `Page[T] = {items, total}`; used for registry, projects, models, jobs, traces. | `utils/platform.py:23-27`; `cli/sync.py:354-358` | Not used for tasks. |
| `group` / `--all` semantics | `group` = GRPO repeats sharing a `group_id`; `--all` runs every loaded row; `--full` = `--all --auto-respond --max-steps 100`. | `eval/taskset.py:280-283`; `cli/eval.py:148-153` | Both operate on an already-materialized set. |
| Any dataset integration (`hud.datasets`, HF, Parquet) | Only `hud.agents.robot.dataset.DatasetWriter` (writes LeRobot v3 Parquet, optional HF push) and `hud.integrations.harbor.adapt` (Harbor *dataset directory* → materialized `Taskset(dataset.name, rows)`). Platform "Data" is a file store (250 MiB/file, `DataFileArg` picker), not a row source. | `agents/robot/dataset.py:1-15`; `integrations/harbor/adapt.py:870-874`; platform/data | No input-dataset concept. |
| RL rollout data loaders | `hud.train` is forward/backward/optim over `Run`s produced by `Taskset.run(group=...)`; no loader. | `train/base.py`, docs `guides/training-agents` | Same materialized path. |
| Warm environment reuse | `Shared(provider, width)` leases one substrate per `(env, runtime_config)` and hands the same `Runtime` to up to `width` concurrent rollouts for the enclosing scope; the env's `_ControlChannel` keeps one `TaskRunner` **per session id**, so one live env runs many `tasks.start`/`tasks.grade` cycles. `HUDRuntime` leases a hosted instance by `POST {runtime}/runtime/sessions {"environment": name}` **per rollout**. | `eval/runtime/core.py:195-282`; `environment/server.py:235-328`; `eval/runtime/hud.py:153-170` | Amortization exists only for client-driven placements; `HostedRuntime` (and platform jobs) boot one instance per trace. |
| Platform limits (docs.hud.ai/platform/limits) | API 1,000 req/min/member (burst 500); gateway model calls 60/min PAYG, 300/min Pro, unlimited Enterprise — *only a hosted run's own model calls are exempt; BYO-key calls count*; one environment runtime 4 h (PAYG/Pro) / 24 h (Enterprise); trace ≤500 MB, step ≤25 MB; Harbor import ≤200k entries; imported job runs 1,000/call; repetitions 1–100; max steps ≤1,000; job "Max concurrent" unlimited by default; per-model hosted capacity (`capacity.effective_max_rollouts`). **No documented maximum taskset size.** | platform/limits, evaluations, models, rest-api | The caps that bite a batch are rate limits, the 4 h instance lifetime, and model capacity — not row count. |
| Control-channel frame size | `CONTROL_FRAME_LIMIT_BYTES = 16 MiB` per JSON-RPC frame (prompt / grade frames). | `environment/utils.py:14` | Bounds how many documents one prompt can carry inline. |

### 1.2 Measured cost of the status quo (pilot, claude-haiku-5-5, concurrency 2)

36 s/file throughput at concurrency 2 ⇒ ~72 s wall per rollout: 20–56 s sandbox/environment start (platform EC2 profile boot + `hud` build serving + HF footer/row-group fetch), 16.7 s post-start task runtime (~6 model calls, ~5 MCP tool calls, 94 % model latency), and ~10–20 s of submit/poll/grade/teardown latency (5 s poll granularity). Startup alone is **38 s × rows**: 10.6 instance-hours for 1,000 docs, ~19,900 instance-hours (2.3 instance-years) for 1.888M, ~5.3M instance-hours for 500M — paid before any model token. That is the cost "batches" exists to remove; model latency is the cost it cannot.

## 2. Workaround with the current SDK (no platform changes)

Four approaches compose; C is the one that actually attacks startup.

**A. Window rows (one `Task` = N documents).** Template `scan_window(dataset: str, revision: str, file: str, row_group: int, offset: int, count: int)`; the env iterates `count` rows internally, exposes `next_document()/submit_spans(record_id, ...)` over its MCP capability, and yields one `EvaluationResult(reward=mean, subscores=[SubScore(name=record_id, weight=1/N, value=score_i, info={...})])`. Per-document detail survives only inside `subscores`/`info` (filterable columns, leaderboard and QA all see one window), one agent context spans N documents (so N must stay small for an agentic model: 5–20), the 16 MiB frame / 25 MB step limits cap inline content, and `EvaluationResult` warns when positive weights do not sum to 1 (`graders/results.py:108-114`). Rows needed: 1.888M/20 = 94k; 500M/20 = 25M — still unsyncable (§4). Good for the local/`--runtime hud` path, poor as the system of record.

**B. Streaming client-side batch runner (no taskset, no sync).** Because `/rollouts/submit` carries the scenario inline, a runner can stream the existing index (`index.parquet`, 45.6 MB), mint `Task(env="pii-detection", id="detect_secrets", args=..., slug=record_id, columns=...)` on the fly, and call `HostedRuntime().run(task, agent, job_id=job.id)` (or `rollout(...)` for client-driven placements) under a bounded semaphore, writing each `Run` to a local ledger (SQLite/JSONL keyed by slug) instead of `job.runs`. One `Job.start("code-pii-full", taskset_id=None)` spans every chunk; resume = skip slugs present in the ledger (or in `GET /jobs/{id}/traces`). Limits: 12 poll requests/min per in-flight rollout → `poll_interval=30` to hold ~400 concurrent under 1,000 req/min; the platform still creates one trace **and one environment instance per row**, so B fixes materialization, not startup.

**C. Warm-env row streaming with `Shared`.** `async with Shared(provider, width=W) as rt:` then `rollout(task, agent, runtime=rt, job_id=...)` per row: one substrate per `(env, runtime_config)`, W concurrent control sessions, instance rotated when `limits.run_timeout_s` expires. With `provider=HUDRuntime()` this leases **one hosted instance per batch** (today it leases one per rollout) with the agent loop local; with `DockerRuntime`/`ModalRuntime`/`Runtime(url)` the substrate is yours and HUD only receives traces (`trace_enter/exit` + steps still report with `HUD_API_KEY`). The env must be session-safe (Josh's MCP tools already key on `record_id`/locator). Caveats: local agent loops count against the gateway limit (60/300 calls/min; ~6 calls/doc ⇒ ≤50 docs/min on Pro ≈ 1.2 s/file ceiling), and whether the hosted runtime accepts W concurrent control sessions on one lease, and for how long, is unverified (open question §6). This is the only today-path that can reach ~3.6 s/file agentically (18 s/row at W=5).

**D. Generated per-batch task files.** Josh's `catalog.write_batches` already emits 1,889 × 1,000-row JSONL batches with a manifest; `hud sync tasks code-pii-b00017 batch-00017.jsonl --force --yes` then `hud eval code-pii-b00017 <model> --full --remote` per batch keeps every request small (≈300 KB) and resumable at batch granularity. It still creates 1.888M platform tasks/versions and 1,889 tasksets (500M → 500k tasksets): fine for the 1.9M preview as a stopgap, not a design.

Minimal runner (B + C), using only public 0.6.22 types:

```python
import asyncio, json, os, pyarrow.parquet as pq
from hud import HUDRuntime, Job, Task
from hud.agents import create_agent
from hud.eval import Shared, rollout

async def main(index="index.parquet", width=5, ledger="ledger.jsonl"):
    done = {json.loads(l)["slug"] for l in open(ledger)} if os.path.exists(ledger) else set()
    agent = create_agent("claude-haiku-5-5")
    job = await Job.start("code-pii-v3 full scan")             # one platform job for every chunk
    sem = asyncio.Semaphore(width)
    async with Shared(HUDRuntime(), width=width) as rt:         # one warm hosted env per batch
        async def one(row):
            task = Task(env="pii-detection", id="detect_secrets", slug=row["id"],
                        args={"record_id": row["id"], "locator": {"backend": "hf", "file": row["file"],
                              "row_group": row["row_group"], "row": row["row"]}},
                        columns={"language": row["language"]})
            async with sem:
                run = await rollout(task, agent, runtime=rt, job_id=job.id, rollout_timeout=600)
            with open(ledger, "a") as f:                        # resumable cursor = the ledger
                f.write(json.dumps({"slug": task.slug, "reward": run.reward, "trace": run.trace_id,
                                    "error": run.trace.error}) + "\n")
        for batch in pq.ParquetFile(index).iter_batches(batch_size=2000):   # storage order, never all rows
            rows = [r for r in batch.to_pylist() if r["eligible"] and r["id"] not in done]
            await asyncio.gather(*(one(r) for r in rows))
```

## 3. Feature proposal: `hud.Batch`

A **Batch** is a *task template bound to a dataset by a binding rule*, stored once on the platform; rows become `Task`s only inside a running job, on the scheduler, in bounded prefetch windows. Design rule: reuse every existing entity (`Task`, `Job`, `Trace`, `EvaluationResult`, `Shared`-style leasing, `/rollouts/submit` payload) and add exactly two new ones: `Batch` (definition, versioned like a task version) and `RowResult` (the light per-row record).

### 3.1 Python API sketch

```python
from hud import Environment, Batch
from hud.batch import Dataset, Binding, Materialization, Execution

env = Environment("pii-detection")

@env.template(returns=Submission)                      # unchanged template
async def detect_secrets(record_id: str, locator: Locator): ...

batch = Batch(
    name="code-pii-v3-full",
    template=detect_secrets,                           # _TaskFactory -> (env name, task id, args schema)
    dataset=Dataset.hf("josh-hud/code-pii-preview", revision="2421297cdcbe808e5728e0ffe248338a3cfd3acf",
                       path="data/*.parquet", columns=["id", "n_chars", "language"]),
                       # also: Dataset.s3("s3://bucket/prefix/*.parquet"), Dataset.gcs(...), Dataset.file(...)
    binding=Binding(
        row_id="id",                                   # per-row id -> Task.slug = f"{batch_id}:{row_id}"
        args={"record_id": "$.id",                     # row -> template args (JSONPath or callable)
              "locator": {"backend": "hf", "file": "$._file", "row_group": "$._row_group", "row": "$._row"}},
        where="n_chars <= 100000",                     # eligibility predicate (pushed down to the reader)
        columns={"language": "$.language"},            # Task.columns facets
    ),
    materialization=Materialization(
        mode="lazy", prefetch_rows=2000, parallel_readers=4, cache_ttl_s=3600,
        order="storage",                               # file/row_group order: one row-group fetch serves 10k rows
        shards=64,                                     # deterministic hash sharding; shard = (i, 64) selects one
        resume_cursor=None,                            # opaque (shard, file, row_group, row) cursor
    ),
    execution=Execution(
        mode="one_shot",                               # or "agentic"
        rows_per_instance=2000, sessions_per_instance=8, instance_ttl_s=3*3600,   # < 4 h platform cap
        trace_sampling=0.01, trace_on_error=True,      # full Trace for 1 % of rows + every failure
        retries=2, max_rows=None, budget_usd=5_000, stop_on_error_rate=0.05,
    ),
)

job = await batch.run(create_agent("claude-haiku-5-5"), runtime=HostedRuntime(),
                      max_concurrent_instances=50)      # -> Job with job.batch_id, job.cursor
async for result in job.results():                      # RowResult stream, resumable by cursor
    ...
sample = batch.sample(1_000, seed=1337).materialize()  # -> Taskset: today's sync/eval/QA/leaderboard path
```

Types: `Dataset(uri, revision, path, columns, format="parquet"|"jsonl", credentials="env:HF_TOKEN")`; `Binding(row_id, args, where, columns, agent_config=None, runtime_config=None)`, validated at `hud sync batch` against `manifest_entry()["args"]` (the same JSON schema the platform already validates stored args against); `RowResult(batch_id, row_id, slug, reward, evaluation: dict, status, duration_s, model_calls, cost_usd, trace_id: str | None, attempt)`. `Job` gains `batch_id`, `cursor`, `rows_done/rows_failed/rows_total_estimate`; `Task` is unchanged (minted rows carry `slug = f"{batch_id}:{row_id}"`, the idempotency key).

### 3.2 CLI sketch

```
hud sync batch batch.py                 # register/version the Batch (template+dataset+binding hash); validates binding vs args schema
hud eval batch.py claude-haiku-5-5 --remote --sample 100   # materialize 100 rows -> ordinary hosted job (today's path)
hud eval batch.py claude-haiku-5-5 --remote --full          # batch job: lazy rows, warm instances, RowResults
hud batch status <job>     # rows done/failed/eta, cursor, instances, spend, error rate
hud batch results <job> --out results.parquet [--since-cursor X]
hud batch resume <job> | hud batch cancel <job> | hud batch shard <job> --shard 3/64
```

### 3.3 Platform semantics

- **Job over a batch**: `POST /batches/{id}/jobs {agent, execution overrides, shards}` returns a job id immediately; nothing is pre-created per row (today "HUD creates the job and all of its traces at once"). The scheduler reads rows through a dataset reader with the stored credentials, applies `where`, mints `Task`s in `prefetch_rows` windows, and dispatches them as today's `/rollouts/submit` payload *internally* to a **warm instance pool**: one environment instance per `rows_per_instance`, `sessions_per_instance` concurrent control sessions each (exactly the `Shared` lease model; the env sees ordinary `tasks.start`/`tasks.grade` per row), rotated at `instance_ttl_s` or the 4 h cap.
- **One-shot mode**: the platform runs a `OneShotAgent` (new `AgentType.ONE_SHOT`, serializable like the others so `HostedRuntime` accepts it): one model call, no tools, `response_format` = the template's `returns` JSON schema (already in the manifest), answer → `tasks.grade`. Localization/grading happens in the env's grade code server-side (offset resolution from the model's quoted text, as the current `find_text` tool does, but in `scoring.py`). Per-row trace = 3 steps (`task/setup`, `agent`, `task/evaluate`), the shape `Run.__aenter__/__aexit__` already produces.
- **Per-row results and traces**: every row writes a `RowResult`; a full `Trace` is recorded for sampled rows (`trace_sampling`), failures, and any row an operator pins. Rewards, subscores (`EvaluationResult.subscores`) and `info` live on the `RowResult`, so leaderboard/QA aggregate over rows while trace storage stays O(sampled).
- **Progress, cursor, partial results**: `GET /batches/{id}/jobs/{job}` → `{rows_done, rows_failed, rows_total_estimate, cursor, rate_rows_per_s, eta, instances, spend}`; `GET .../results?cursor=` streams RowResults (Parquet/JSONL). `cursor` is deterministic given `order`/`shards`, so `resume` continues after a crash and two shards never overlap.
- **Retries / idempotency**: slug `batch_id:row_id` is unique per attempt set; a retry writes `attempt+1` and supersedes; a duplicate submit is a no-op. Row failures never fail the job (today's "failed Run inside the job" rule); `stop_on_error_rate` pauses it.
- **Cost controls**: `budget_usd`, `max_rows`, `max_concurrent_instances`, `instance_ttl_s`, model `capacity` awareness (pending until headroom, as today), `--sample` for dry runs.
- **Reproducibility**: a batch version hashes template id + env build + dataset `(uri, revision/etag)` + binding + ordering; jobs on the same version are comparable (the "task version" rule extended to dataset revision).

### 3.4 Data flow

```mermaid
flowchart LR
  subgraph client
    B[batch.py: Batch(template, dataset, binding, policies)] -->|hud sync batch| REG[(Batch version)]
    CLI[hud eval batch.py model --full] -->|POST /batches/id/jobs| SCH
    CLI -->|GET status / results?cursor| RES
  end
  subgraph platform
    REG --> SCH[Batch scheduler]
    DS[(HF / S3 / GCS Parquet)] -->|reader: where, order=storage, prefetch N, cache TTL| SCH
    SCH -->|mint Task slug=batch:row| POOL[Warm instance pool\n1 env instance / rows_per_instance\nK sessions each, TTL < 4 h]
    POOL -->|tasks.start| ENV[Environment\nTaskRunner per session]
    ENV -->|prompt| AG{mode}
    AG -->|one_shot: 1 structured call| M[(Model / gateway)]
    AG -->|agentic: tool loop| M
    M -->|answer| ENV
    ENV -->|tasks.grade -> EvaluationResult| RES[(RowResult table\nreward, subscores, info, cost)]
    RES -.->|sampled / failed rows| TR[(Trace store)]
    SCH -->|cursor, progress| JOB[(Job: batch_id, cursor)]
  end
```

### 3.5 Mapping onto existing entities and backward compatibility

| Existing | Role in a batch | Change |
|---|---|---|
| `@env.template` / `_TaskFactory` | the `template`; its args schema validates `Binding.args` | none |
| `Task` | ephemeral minted row, `slug=batch_id:row_id`; `columns` from binding | none (slug convention) |
| `Taskset` | `batch.sample(n).materialize()` / `--sample` yields one; unchanged flows for QA, review, leaderboard samples | `Taskset.from_batch(batch, limit)` helper |
| `Job` | the receipt; `Job.start`-style accumulation already exists | add `batch_id`, `cursor`, row counters |
| `Trace` / `Step` | sampled or on-error rows; one-shot = 3 steps | none |
| `EvaluationResult` / `SubScore` | per-row grade carried on `RowResult` | none |
| `HostedRuntime` / `/rollouts/submit` | internal dispatch unit per row (payload identical) | server-side pooling instead of one instance per submit |
| `Shared` | the lease model the pool implements | expose `sessions_per_instance` semantics server-side |
| `hud eval`, `hud sync` | accept `Batch` objects found in source (`Taskset._scan_tasks` extension) | new subcommands only |

A `Batch` with `mode="agentic"`, `rows_per_instance=1`, `trace_sampling=1.0` behaves exactly like today's hosted taskset job, so nothing existing changes meaning; `taskset_id` and `batch_id` are mutually exclusive on a job.

## 4. Scale analysis

Row-shape numbers measured on the pii-detection row (`record_id` + locator + one column): 287 B/row JSONL, 302 B/row upload JSON, 2.34 KB/row in-memory `Task`. Time numbers use the pilot (38 s mean startup, 16.7 s agentic, ~4 s assumed for a one-shot structured call on a 7.2k-char mean / 2.7k median document — estimate, to be measured).

| | 1,000 docs (today's POC) | 1.888M docs (v3 preview) | 500M docs |
|---|---|---|---|
| Task rows needed (status quo) | 1,000 | 1,888,450 | 5×10⁸ |
| JSONL / sync POST size | 0.29 MB / 0.30 MB — syncs in seconds (done) | 542 MB / **570 MB single POST**; diff fetches a 570 MB export too | 144 GB / 151 GB — impossible |
| `Taskset.from_file` RAM | 2.3 MB | ~4.4 GB (+ remote copy for diff ≈ 9 GB) | ~1.1 TB |
| Default-slug collisions | 0 | ≈415 → `ValueError`, explicit slugs required | ≈2.9×10⁷ |
| Platform job (status quo) | 1,000 traces, 1,000 instances; 10.6 instance-h of startup | 1.888M traces/instances created at once; **~19,900 instance-h (2.3 yr) of startup**; `hud eval --full` would `gather` 1.888M coroutines | 5×10⁸ traces; 5.3M instance-h of startup; nothing in the path survives |
| Serial model time, agentic (16.7 s) | 4.6 h | 365 days | 265 years |
| Serial model time, one-shot (~4 s) | 1.1 h | 87 days | 63 years |
| Batches (lazy rows, warm pool) | same result, ~1 instance instead of 1,000 | 0 stored rows; 1.888M RowResults; ~950 instance-lifetimes at 2,000 rows each | 0 stored rows; 5×10⁸ RowResults (~50–100 GB); traces only for the sampled 1 % (5M) |
| What still breaks with batches | nothing | gateway/model capacity if the agent loop is client-side (300 calls/min Pro) | total compute and tokens (§5) |

Ordering matters as much as row count: the HF backend reads a whole 10,000-row row group (~20 MB) to serve one record, so random-order sampling costs ~20 MB/row (500M rows → ~10 PB egress) while `order="storage"` costs one pass over the 4.28 GB file set per shard. This is why `Materialization.order` is part of the proposal, not a tuning knob.

## 5. Required concurrency, and when HUD is the wrong loop

Concurrency needed is `C = N × t_row / T_wall` (rows in flight continuously). Even at Josh's 3.6 s/file target, 500M files = 1.8×10⁹ s ≈ **57 years** of serial time; at 1.888M files it is 79 days.

| Target wall time | 1.888M, agentic warm (18 s/row) | 1.888M, one-shot (4 s/row) | 500M, agentic warm (18 s) | 500M, one-shot (4 s) |
|---|---|---|---|---|
| 24 h | 393 rows in flight (≈50 instances × 8) | 87 | 104,000 | 23,000 |
| 7 days | 56 | 13 | 14,900 | 3,300 |
| 30 days | 13 | 3 | 3,500 | 770 |
| Model calls total | 11.3M | 1.9M | 3×10⁹ | 5×10⁸ |
| Tokens (est.: ~4k/doc one-shot, ~30k/doc agentic) | ~57 B | ~7.5 B | ~15 T | ~2 T |

Reading the table: batches make the 1.9M preview a one-to-several-day job at modest concurrency (and amortize ~2 instance-years of startup), but 500M files needs thousands of continuously busy rows for a month, ~5×10⁸ model calls at minimum (≈190 calls/s sustained; the Pro gateway allows 5/s for non-hosted callers and hosted capacity is per-model and shared across teams), and on the order of 2 T tokens one-shot or ~15 T agentic — at a nominal $1 per million input tokens that is roughly $2M vs $15M before output tokens (illustrative; substitute current haiku/gateway rates). None of that is a scheduling problem HUD can remove; it is the cost of running a frontier model over every file.

Use a non-HUD bulk pipeline for the corpus when the per-row work is `f(document) → structured output` with no environment state and no tool use — which is exactly the one-shot mode: provider batch APIs (OpenAI Batch, Anthropic Message Batches, Gemini batch; typically ~50 % discount, ≤24 h turnaround, tens of thousands of requests per file) or self-hosted vLLM/TGI on GPUs with a small fine-tuned detector (the stated goal of the code-pii datasets), where 2 T tokens at ~10–20k tokens/s per GPU is ~1,200–2,300 GPU-days — a few hundred GPUs for a week, again illustrative. HUD then does what it is built for, at sample scale: (1) calibrate prompt/model/grader on stratified `Batch.sample()` tasksets with the exact-offset grader and leaderboard (the 1,000-row POC is this); (2) regression tasksets per dataset revision; (3) QA agents on sampled traces; (4) an **agentic escalation lane** — route the low-confidence or high-value fraction (1–5 %) of bulk outputs into agentic HUD tasks with tools, which is where per-row traces earn their cost; (5) training (SFT/RL on graded rollouts) of the small detector that then runs the bulk pass. Rule of thumb: HUD per-row when a row needs an environment, tools, or a replayable trace; bulk inference when it needs one model call and a score.

## 6. Open questions

1. Hosted runtime: does one `POST /runtime/sessions` lease accept many concurrent control sessions (so `Shared(HUDRuntime(), width=W)` works today), and is the lease lifetime the 4 h environment cap? This decides whether §2-C is usable before any platform work.
2. One-shot mode: a new `AgentType.ONE_SHOT` (serializable for `HostedRuntime`) or an `agent_config` flag (`max_steps=1`, `tools=[]`, `response_format=returns`)? How should the gateway bill and rate-limit it (hosted exemption applies)?
3. `RowResult` as a first-class entity: retention, export limits, leaderboard/QA semantics over rows without traces, and whether invalidation applies per row or per trace.
4. Dataset credentials and egress: where the reader's HF/S3 token lives (environment runtime variables exist; a batch-level secret may be cleaner), and whether the platform reader or the environment fetches rows (reader-side fetch enables row-group-ordered prefetch and caching across instances).
5. Default slug: should `Task.slug` default widen beyond `sha1[:8]` (collisions at ~10⁵ rows), independent of batches?
6. `hud sync tasks`: chunked upload and paginated diff (`/tasksets/{id}/tasks` already pages) as a near-term fix for 10⁴–10⁶-row tasksets even if batches land later.
7. Comparability: when a dataset revision changes, is a batch job comparable to the previous one (new batch version, like a task version), and how does `group` (GRPO repeats) interact with rows for training on batch samples?
8. Non-determinism and retries: a retried row with a different model answer supersedes or accumulates attempts; what does the leaderboard count?
9. Row result granularity in agentic warm mode: per-row traces are small (≤~20 steps) but 1.888M of them at ~50 KB is ~90 GB; is sampling + on-error the right default, or should traces be always-on below some row count?
