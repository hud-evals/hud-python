"""The HUD platform as tests see it: records a test seeds, and requests it records.

One ``Platform`` answers the platform API through ``FakeServices``: projects,
the environment registry, tasksets, builds, Harbor imports, the model catalog
and its training service, jobs and traces, QA and CLI analytics. It answers from
what a test seeded and records what the code sent; it implements none of the
platform's own rules (what a cancel stops, which checkpoint is served, how QA
results settle), which the platform tests itself. It keeps state only where a
flow reads back what it wrote: a build triggered with the id its upload got, a
taskset found again by the name a sync created. A fault is a route override on
``platform.services``, which wins over the platform's own routes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from dataclasses import dataclass, field, replace
from itertools import count
from typing import TYPE_CHECKING, Any

from .services import Reply

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.websockets import WebSocket

    from .services import FakeServices, Request

    Handler = Callable[[Request], Reply | Awaitable[Reply]]

API_KEY = "sk-hud-test"
USER_ID = "11111111-1111-4111-8111-111111111111"
TEAM_ID = "22222222-2222-4222-8222-222222222222"
WEB_URL = "https://hud.example"
DEFAULT_PROJECT_ID = "aaaaaaaa-0000-4000-8000-000000000001"
BROWSER_PROJECT_ID = "aaaaaaaa-0000-4000-8000-000000000002"
LOCKED_PROJECT_ID = "aaaaaaaa-0000-4000-8000-000000000003"
LIVE_IMPORT = {"queued", "starting", "running", "cancelling"}


def key(identifier: str) -> str:
    """One spelling for an id the platform receives dashed or compact."""
    try:
        return uuid.UUID(identifier).hex
    except ValueError:
        return identifier


def not_found(what: str) -> Reply:
    return Reply(status=404, json={"detail": f"{what} not found"})


@dataclass
class Build:
    """A build: the context uploaded for it, the trigger that started it, how it ended."""

    id: str
    context: bytes | None = None
    trigger: dict[str, Any] | None = None
    status: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "SUCCEEDED",
            "version": "1",
            "uri": "registry.hud.test/env:1",
            "manifest": {"tasks": [], "capabilities": []},
        }
    )
    #: The frames the build-log socket sends; ``None`` refuses the socket.
    log: list[dict[str, Any]] | None = field(
        default_factory=lambda: [
            {"type": "log", "message": "Step 1/2 : FROM python:3.12\n", "timestamp": 0},
            {"type": "status_update", "status": "SUCCEEDED"},
            {"type": "complete", "message": "Build SUCCEEDED"},
        ]
    )


@dataclass
class Trace:
    """What the platform knows of one trace.

    A rollout reported from this process has an ``enter`` and an ``exit``; a hosted
    one has the ``submitted`` request and the ``states`` its polls read, each read
    moving to the next and the last one staying.
    """

    enter: dict[str, Any] | None = None
    exit: dict[str, Any] | None = None
    spans: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    submitted: dict[str, Any] | None = None
    states: list[dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False


@dataclass
class Import:
    """A Harbor import: the bundle uploaded for it and the statuses it moves through."""

    id: str
    taskset_id: str
    bundle: bytes | None = None
    statuses: list[str] = field(default_factory=lambda: ["completed"])
    format: str = "harbor"
    error: str | None = None
    report: dict[str, Any] | None = None
    cancelled: bool = False

    def record(self, status: str) -> dict[str, Any]:
        reported = status in {"completed", "stopped"}
        return {
            "id": self.id,
            "format": self.format,
            "taskset_id": self.taskset_id,
            "file": {
                "id": self.id,
                "filename": "bundle.zip",
                "size_bytes": len(self.bundle or b""),
            },
            "status": status,
            "summary": "Imported 1 task(s) and 1 run(s)" if reported else None,
            "metrics": {"tasks_imported": 1.0} if reported else None,
            "error": self.error,
            "created_at": "2026-10-07T12:00:00Z",
            "completed_at": None if status in LIVE_IMPORT else "2026-10-07T12:01:00Z",
        }

    def advance(self) -> dict[str, Any]:
        """The record at its current status; each read moves to the next, the last one stays."""
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return self.record(status)


@dataclass
class Training:
    """One model's training: the gradient batches sent to it and the optimizer steps taken."""

    batches: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)


class Platform:
    """The fake HUD platform; see the module docstring."""

    def __init__(self, services: FakeServices) -> None:
        self.services = services
        self._ids = count(1)
        #: The keys the platform accepts; any other bearer is answered 401.
        self.keys = {API_KEY}
        #: The most items a list endpoint returns at once, whatever the caller asks for.
        self.page_size = 500
        #: How the next build ends: what its status reports and what its log sends.
        self.next_build = Build("")
        self.projects: list[dict[str, Any]] = [
            {
                "id": DEFAULT_PROJECT_ID,
                "name": "default",
                "is_default": True,
                "capabilities": {"view": True, "create": True},
            },
            {"id": BROWSER_PROJECT_ID, "name": "browser-evals", "capabilities": {"create": True}},
            {"id": LOCKED_PROJECT_ID, "name": "locked-down", "capabilities": {"view": True}},
        ]
        self.environments: dict[str, dict[str, Any]] = {}
        self.tasksets: dict[str, dict[str, Any]] = {}
        self.uploads: list[dict[str, Any]] = []
        self.builds: dict[str, Build] = {}
        self.models: list[dict[str, Any]] = []
        self.checkpoints: dict[str, list[dict[str, Any]]] = {}
        self.heads: dict[str, str] = {}
        self.training: dict[str, Training] = {}
        #: What every optimizer step answers.
        self.optim_result: dict[str, Any] = {
            "step": 1,
            "checkpoint_id": "ckpt-trained",
            "sampler_path": "sampler/ckpt-trained",
            "state_path": "state/ckpt-trained",
            "model": "owner/model",
        }
        self.jobs: dict[str, dict[str, Any]] = {}
        self.job_traces: dict[str, list[dict[str, Any]]] = {}
        #: Rollout cancellations as (scope, request body): ``job``, ``trace`` or ``all``.
        self.cancellations: list[tuple[str, dict[str, Any]]] = []
        self.traces: dict[str, Trace] = {}
        #: The states the next hosted rollout's trace moves through as it is polled.
        self.next_rollout: list[dict[str, Any]] = [{"status": "completed", "reward": 1.0}]
        self.imports: dict[str, Import] = {}
        #: How the next import runs: the statuses it moves through and the report it leaves.
        self.next_import = Import("", taskset_id="")
        self.import_formats: list[dict[str, Any]] = [
            {"format": "harbor", "title": "Harbor", "max_input_bytes": 1024**3}
        ]
        self.qa_checks: list[dict[str, Any]] = []
        self.qa_results: list[dict[str, Any]] = []
        #: What successive reads of the results return, the last one repeating; empty
        #: returns ``qa_results``, which starting a run adds to.
        self.qa_polls: list[list[dict[str, Any]]] = []
        self.cli_events: list[dict[str, Any]] = []
        self._serve()

    # ─── seeding ──────────────────────────────────────────────────────

    def _id(self, prefix: str) -> str:
        return f"{prefix}-0000-4000-8000-{next(self._ids):012d}"

    def add_environment(self, name: str, **record: Any) -> str:
        """A deployed environment in the registry; returns its id."""
        environment_id = record.pop("id", None) or self._id("bbbbbbbb")
        self.environments[environment_id] = {
            "id": environment_id,
            "name": name,
            "latest_build": {"version": 1},
            **record,
        }
        return environment_id

    def add_taskset(
        self, name: str, tasks: list[dict[str, Any]] | None = None, **record: Any
    ) -> str:
        """A taskset holding ``tasks`` as the platform exports them; returns its id."""
        taskset_id = record.pop("id", None) or self._id("eeeeeeee")
        self.tasksets[taskset_id] = {
            "id": taskset_id,
            "name": name,
            "can_edit": True,
            "tasks": {task["name"]: task for task in tasks or []},
            **record,
        }
        return taskset_id

    def taskset(self, name: str) -> dict[str, Any]:
        (found,) = [taskset for taskset in self.tasksets.values() if taskset["name"] == name]
        return found

    def add_model(self, **record: Any) -> str:
        """A model in the gateway catalog; returns its id."""
        model_id = record.setdefault("id", self._id("cccccccc"))
        self.models.append(record)
        return model_id

    def add_job(self, traces: list[dict[str, Any]] | None = None, **record: Any) -> str:
        """A finished job the platform lists, with the traces it lists under it."""
        job_id = record.setdefault("id", self._id("dddddddd"))
        self.jobs[key(job_id)] = record
        self.job_traces[key(job_id)] = traces or []
        return job_id

    def trace(self, trace_id: str) -> Trace:
        return self.traces.setdefault(key(trace_id), Trace())

    # ─── serving ──────────────────────────────────────────────────────

    def _serve(self) -> None:
        def route(service: str, method: str, path: str, handler: Handler) -> None:
            self.services.route(service, method, path, handler=self._signed_in(handler))

        route(
            "api",
            "GET",
            "/v2/auth/me",
            lambda _: Reply(json={"user_id": USER_ID, "team_id": TEAM_ID}),
        )
        route("api", "GET", "/v2/projects", self._list_projects)
        route("api", "GET", "/v2/projects/{id}", self._get_project)
        route("api", "POST", "/v2/projects", self._create_project)
        route("api", "GET", "/v2/registry", self._list_environments)
        route("api", "GET", "/v2/registry/{id}", self._get_environment)
        route("api", "GET", "/v2/tasksets/by-name/{name}", self._taskset_by_name)
        route("api", "GET", "/v2/tasksets/{id}", self._get_taskset)
        route("api", "GET", "/v2/tasksets/{id}/export", self._export_taskset)
        route("api", "POST", "/v2/tasksets", self._create_taskset)
        route("api", "POST", "/v2/tasks/upload", self._upload_tasks)
        route("api", "POST", "/v2/builds/upload-url", self._build_upload_url)
        self.services.route("api", "PUT", "/uploads/{id}", handler=self._receive_upload)
        route("api", "POST", "/v2/builds/trigger", self._trigger_build)
        route("api", "GET", "/v2/builds/{id}/status", self._build_status)
        self.services.websocket("api", "/v2/builds/{id}/logs", self._build_log)
        route("api", "GET", "/v2/models", self._list_models)
        route("api", "GET", "/v2/models/{id}/checkpoints", self._checkpoints)
        route("api", "PUT", "/v2/models/{id}/head", self._set_head)
        route("api", "POST", "/v2/models/fork", self._fork_model)
        route("rl", "POST", "/v1/models/{id}/train/forward-backward", self._accumulate)
        route("rl", "POST", "/v1/models/{id}/train/backward", self._accumulate)
        route("rl", "POST", "/v1/models/{id}/train/optim-step", self._optim_step)
        route("api", "POST", "/v2/trace/job/{id}/enter", self._enter_job)
        route("api", "POST", "/v2/trace/{id}/enter", self._enter_trace)
        route("api", "POST", "/v2/trace/{id}/exit", self._exit_trace)
        route("api", "GET", "/v2/trace/{id}/events", self._trace_events)
        route("api", "POST", "/v2/rollouts/submit", self._submit_rollout)
        route("api", "GET", "/v2/trace/{id}", self._poll_trace)
        route("telemetry", "POST", "/trace/{id}/telemetry-upload", self._upload_spans)
        route("api", "GET", "/v2/jobs", self._list_jobs)
        route("api", "GET", "/v2/jobs/{id}/traces", self._job_traces)
        route("api", "POST", "/v2/rollouts/cancel_job", self._cancel_job)
        route("api", "POST", "/v2/rollouts/cancel", self._cancel_trace)
        route("api", "POST", "/v2/rollouts/cancel_user_jobs", self._cancel_every_job)
        route("api", "POST", "/v2/imports/uploads", self._import_upload_url)
        route("api", "POST", "/v2/data/{id}/complete", self._complete_upload)
        route("api", "DELETE", "/v2/data/{id}", self._delete_upload)
        route("api", "POST", "/v2/imports", self._start_import)
        route("api", "GET", "/v2/imports", self._list_imports)
        route("api", "GET", "/v2/imports/{id}", self._get_import)
        route("api", "GET", "/v2/imports/{id}/report", self._import_report)
        route("api", "POST", "/v2/imports/{id}/cancel", self._cancel_import)
        route("api", "GET", "/v2/imports/formats", self._import_formats)
        route("api", "GET", "/v2/qa/checks", lambda _: Reply(json={"checks": self.qa_checks}))
        route("api", "POST", "/v2/qa/runs", self._start_qa)
        route("api", "GET", "/v2/qa/results", self._qa_results)
        self.services.route("telemetry", "POST", "/sdk-events/cli", handler=self._cli_events)

    def _signed_in(self, handler: Handler) -> Handler:
        """``handler``, answering 401 to a request without an accepted key."""

        def answer(request: Request) -> Reply | Awaitable[Reply]:
            if request.bearer not in self.keys:
                return Reply(status=401, json={"detail": "Invalid API key"})
            return handler(request)

        return answer

    # projects and the registry

    def _list_projects(self, request: Request) -> Reply:
        offset = int(request.query.get("offset", ["0"])[0])
        limit = min(int(request.query.get("limit", ["500"])[0]), self.page_size)
        page = self.projects[offset : offset + limit]
        return Reply(json={"items": page, "total": len(self.projects)})

    def _get_project(self, request: Request) -> Reply:
        found = [project for project in self.projects if project["id"] == request.params["id"]]
        return Reply(json=found[0]) if found else not_found("Project")

    def _create_project(self, request: Request) -> Reply:
        body = request.json
        project = {
            "id": self._id("aaaaaaaa"),
            "name": body["name"],
            "capabilities": {"view": True, "create": True},
            **({"description": body["description"]} if "description" in body else {}),
        }
        self.projects.append(project)
        return Reply(json=project)

    def _list_environments(self, request: Request) -> Reply:
        del request
        listed = list(self.environments.values())
        return Reply(json={"items": listed, "total": len(listed)})

    def _get_environment(self, request: Request) -> Reply:
        found = self.environments.get(request.params["id"])
        return Reply(json=found) if found else not_found("Environment")

    # tasksets

    def _taskset_by_name(self, request: Request) -> Reply:
        for taskset in self.tasksets.values():
            if taskset["name"] == request.params["name"]:
                return Reply(json={"taskset_id": taskset["id"], "name": taskset["name"]})
        return not_found("Taskset")

    def _get_taskset(self, request: Request) -> Reply:
        taskset = self.tasksets.get(request.params["id"])
        if taskset is None:
            return not_found("Taskset")
        return Reply(json={key: value for key, value in taskset.items() if key != "tasks"})

    def _export_taskset(self, request: Request) -> Reply:
        taskset = self.tasksets.get(request.params["id"])
        if taskset is None:
            return not_found("Taskset")
        return Reply(json={"name": taskset["name"], "tasks": list(taskset["tasks"].values())})

    def _create_taskset(self, request: Request) -> Reply:
        body = request.json
        taskset_id = self.add_taskset(body["name"], project_id=body.get("project_id"))
        taskset = self.tasksets[taskset_id]
        return Reply(json={key: value for key, value in taskset.items() if key != "tasks"})

    def _upload_tasks(self, request: Request) -> Reply:
        body = request.json
        self.uploads.append(body)
        taskset_id = body.get("taskset_id") or self.add_taskset(
            body["taskset_name"], project_id=body.get("project_id")
        )
        tasks = self.tasksets[taskset_id]["tasks"]
        created = sum(task["name"] not in tasks for task in body["tasks"])
        for task in body["tasks"]:
            tasks[task["name"]] = {
                "name": task["name"],
                "env": task["env"]["name"],
                "scenario": task["task_id"],
                "args": task.get("args") or {},
            }
        return Reply(
            json={
                "taskset_id": taskset_id,
                "tasks_created": created,
                "tasks_updated": len(body["tasks"]) - created,
            }
        )

    # builds

    def _build_upload_url(self, request: Request) -> Reply:
        del request
        build = replace(self.next_build, id=f"build-{next(self._ids)}")
        self.builds[build.id] = build
        url = f"{self.services.url('api')}/uploads/{build.id}"
        return Reply(json={"upload_url": url, "build_id": build.id})

    def _receive_upload(self, request: Request) -> Reply:
        upload_id = request.params["id"]
        if upload_id in self.builds:
            self.builds[upload_id].context = request.body
        else:
            self.imports[upload_id].bundle = request.body
        return Reply(json={})

    def _trigger_build(self, request: Request) -> Reply:
        body = request.json
        build = self.builds[body["build_id"]]
        build.trigger = body
        environment_id = body.get("registry_id") or next(
            (env["id"] for env in self.environments.values() if env["name"] == body["name"]),
            None,
        )
        if environment_id is None:
            environment_id = self.add_environment(body["name"], project_id=body.get("project_id"))
        return Reply(json={"id": build.id, "registry_id": environment_id})

    def _build_status(self, request: Request) -> Reply:
        build = self.builds.get(request.params["id"])
        return Reply(json=build.status) if build else not_found("Build")

    async def _build_log(self, websocket: WebSocket, params: dict[str, str]) -> None:
        log = self.builds[params["id"]].log
        if log is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        for frame in log:
            await websocket.send_text(json.dumps(frame))
        await websocket.close()

    # models

    def _list_models(self, request: Request) -> Reply:
        offset = int(request.query.get("offset", ["0"])[0])
        limit = min(int(request.query.get("limit", ["100"])[0]), self.page_size)
        page = self.models[offset : offset + limit]
        return Reply(json={"items": page, "total": len(self.models)})

    def _checkpoints(self, request: Request) -> Reply:
        return Reply(json=self.checkpoints.get(request.params["id"], []))

    def _set_head(self, request: Request) -> Reply:
        self.heads[request.params["id"]] = request.json["checkpoint_id"]
        return Reply(json={})

    def _fork_model(self, request: Request) -> Reply:
        body = request.json
        self.add_model(
            name=body["name"],
            model_name=f"team/{body['name']}",
            forked_from=body["source_model_id"],
        )
        return Reply(json=self.models[-1])

    # training

    def _accumulate(self, request: Request) -> Reply:
        batch = request.json
        self.training.setdefault(request.params["id"], Training()).batches.append(batch)
        datums = batch.get("inputs") or batch.get("weights") or []
        return Reply(json={"metrics": {}, "num_datums": len(datums)})

    def _optim_step(self, request: Request) -> Reply:
        training = self.training.setdefault(request.params["id"], Training())
        training.steps.append(request.json)
        return Reply(json=self.optim_result)

    # jobs and traces

    def _enter_job(self, request: Request) -> Reply:
        self.jobs[key(request.params["id"])] = request.json
        return Reply(json={})

    def _enter_trace(self, request: Request) -> Reply:
        self.trace(request.params["id"]).enter = request.json
        return Reply(json={})

    def _exit_trace(self, request: Request) -> Reply:
        self.trace(request.params["id"]).exit = request.json
        return Reply(json={})

    def _trace_events(self, request: Request) -> Reply:
        trace = self.traces.get(key(request.params["id"]))
        return Reply(json={"events": trace.events}) if trace else not_found("Trace")

    def _upload_spans(self, request: Request) -> Reply:
        self.trace(request.params["id"]).spans.extend(request.json["telemetry"])
        return Reply(json={})

    def _list_jobs(self, request: Request) -> Reply:
        del request
        return Reply(json={"items": list(self.jobs.values())})

    def _job_traces(self, request: Request) -> Reply:
        traces = self.job_traces.get(key(request.params["id"]))
        return Reply(json={"items": traces}) if traces is not None else not_found("Job")

    def _cancel_job(self, request: Request) -> Reply:
        self.cancellations.append(("job", request.json))
        return Reply(json={"cancelled": 2})

    def _cancel_trace(self, request: Request) -> Reply:
        self.cancellations.append(("trace", request.json))
        self.trace(request.json["trace_id"]).cancelled = True
        return Reply(json={"status": "accepted"})

    def _submit_rollout(self, request: Request) -> Reply:
        trace = self.trace(request.json["trace_id"])
        trace.submitted = request.json
        trace.states = [dict(state) for state in self.next_rollout]
        return Reply(json={"status": "queued"})

    def _poll_trace(self, request: Request) -> Reply:
        trace = self.traces.get(key(request.params["id"]))
        if trace is None or not trace.states:
            return not_found("Trace")
        return Reply(json=trace.states.pop(0) if len(trace.states) > 1 else trace.states[0])

    def _cancel_every_job(self, request: Request) -> Reply:
        self.cancellations.append(("all", request.json))
        return Reply(json={"jobs_cancelled": 1, "total_tasks_cancelled": 2, "job_details": []})

    # Harbor imports

    def _import_formats(self, request: Request) -> Reply:
        del request
        return Reply(json={"formats": self.import_formats})

    def _import_upload_url(self, request: Request) -> Reply:
        del request
        upload_id = self._id("ffffffff")
        self.imports[upload_id] = replace(
            self.next_import, id=upload_id, statuses=list(self.next_import.statuses)
        )
        url = f"{self.services.url('api')}/uploads/{upload_id}?signature=secret"
        return Reply(json={"file": {"id": upload_id, "filename": "bundle.zip"}, "upload_url": url})

    def _complete_upload(self, request: Request) -> Reply:
        return Reply(json={"id": request.params["id"]})

    def _delete_upload(self, request: Request) -> Reply:
        self.imports.pop(request.params["id"], None)
        return Reply(status=204)

    def _start_import(self, request: Request) -> Reply:
        body = request.json
        started = self.imports[body["file_id"]]
        started.taskset_id = body["taskset_id"]
        return Reply(json=started.advance())

    def _list_imports(self, request: Request) -> Reply:
        del request
        listed = [
            started.record(started.statuses[0])
            for started in self.imports.values()
            if started.taskset_id
        ]
        return Reply(json={"imports": listed, "total": len(listed)})

    def _get_import(self, request: Request) -> Reply:
        started = self.imports.get(request.params["id"])
        return Reply(json=started.advance()) if started else not_found("Import")

    def _import_report(self, request: Request) -> Reply:
        started = self.imports.get(request.params["id"])
        if started is None or started.report is None:
            return not_found("Import report")
        return Reply(json={"taskset_id": started.taskset_id, **started.report})

    def _cancel_import(self, request: Request) -> Reply:
        started = self.imports[request.params["id"]]
        started.cancelled = True
        return Reply(json=started.record("cancelling"))

    # QA

    def _start_qa(self, request: Request) -> Reply:
        body = request.json
        level = "trace" if "trace_ids" in body else "task"
        started = [
            {
                "id": self._id("99999999"),
                "check_key": check,
                "subject_type": level,
                f"subject_{level}_id": subject_id,
                "status": "queued",
                "verdict": None,
                "result": None,
            }
            for check in body["check_keys"]
            for subject_id in body[f"{level}_ids"]
        ]
        self.qa_results.extend(started)
        return Reply(json={"results": started})

    def _qa_results(self, request: Request) -> Reply:
        del request
        polled = self.qa_polls.pop(0) if len(self.qa_polls) > 1 else (self.qa_polls or [None])[0]
        return Reply(json={"results": self.qa_results if polled is None else polled})

    # CLI analytics

    def _cli_events(self, request: Request) -> Reply:
        self.cli_events.extend(request.json["events"])
        return Reply(json={})


# ─── the runtime gateway ───────────────────────────────────────────────
#
# The HUD runtime service's tunnel: each WebSocket is spliced to a local control channel.

SESSIONS = "/runtime/sessions"
SESSION = "/runtime/sessions/{id}"
TUNNEL = "/runtime/tunnels/{id}"


def tunnel(port: int) -> Callable[[WebSocket, dict[str, str]], Awaitable[None]]:
    """A tunnel endpoint splicing each WebSocket to the control channel on ``port``."""

    async def handler(websocket: WebSocket, params: dict[str, str]) -> None:
        del params
        await websocket.accept()
        reader, writer = await asyncio.open_connection("127.0.0.1", port)

        async def upstream() -> None:
            while (message := await websocket.receive())["type"] != "websocket.disconnect":
                writer.write(message.get("bytes") or message.get("text", "").encode())
                await writer.drain()

        async def downstream() -> None:
            while data := await reader.read(65536):
                await websocket.send_bytes(data)

        tasks = [asyncio.create_task(upstream()), asyncio.create_task(downstream())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        with contextlib.suppress(Exception):
            await websocket.close()

    return handler


def host_on_runtime(services: FakeServices, port: int, session: str = "sess-1") -> None:
    """Let the fake runtime service lease ``session`` and tunnel it to the env on ``port``."""
    services.route("runtime", "POST", SESSIONS, json={"id": session})
    services.route("runtime", "DELETE", SESSION, json={})
    services.websocket("runtime", TUNNEL, tunnel(port))
