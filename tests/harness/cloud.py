"""Fake Modal and Daytona SDKs: the outside world the cloud runtimes call.

Each fake installs itself as the SDK module, records every call it receives as
plain data in ``calls`` (the transcript tests snapshot), and connects the
sandboxes it creates to environments this process serves, so a rollout placed
on a fake sandbox really runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    import pytest


def aio(function: Callable[..., Any]) -> SimpleNamespace:
    """Expose ``function`` the way both SDKs do: as ``method.aio(...)``."""
    return SimpleNamespace(aio=function)


class Stream:
    """An SDK output stream: yields its text, then stays open while ``hold`` is unset."""

    def __init__(self, text: str, hold: asyncio.Event | None) -> None:
        self.text = text
        self.hold = hold

    def __aiter__(self) -> AsyncIterator[str]:
        return self.chunks()

    async def chunks(self) -> AsyncIterator[str]:
        if self.text:
            yield self.text
        if self.hold is not None:
            await self.hold.wait()


@dataclass
class Exec:
    """How the fake answers a sandbox command containing ``match``."""

    match: str
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    hang: bool = False


@dataclass
class FakeModal:
    """The ``modal`` module. ``addresses`` maps each image to the env serving it."""

    addresses: dict[str, tuple[str, int]]
    calls: list[tuple[str, Any]] = field(default_factory=list)
    execs: list[Exec] = field(default_factory=list)
    output: tuple[str, str] = ("", "")
    hold: asyncio.Event | None = None
    sandboxes: int = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeModal:
        module = ModuleType("modal")
        module.__dict__.update(
            Image=SimpleNamespace(
                from_registry=lambda name, secret=None: self.image("from_registry", name, secret),
                from_id=lambda image_id: self.image("from_id", image_id),
                from_name=lambda name: self.image("from_name", name),
            ),
            App=SimpleNamespace(lookup=aio(self.lookup)),
            Probe=SimpleNamespace(with_tcp=lambda port: f"tcp:{port}"),
            Sandbox=SimpleNamespace(create=aio(self.create)),
        )
        monkeypatch.setitem(sys.modules, "modal", module)
        return self

    def image(self, method: str, name: str, secret: Any = None) -> str:
        self.calls.append((f"Image.{method}", [name] if secret is None else [name, secret]))
        return name

    async def lookup(self, name: str, *, create_if_missing: bool) -> str:
        self.calls.append(("App.lookup", {"name": name, "create_if_missing": create_if_missing}))
        return f"app:{name}"

    async def create(self, *args: str, **kwargs: Any) -> SimpleNamespace:
        self.sandboxes += 1
        sandbox_id = f"sb-{self.sandboxes}"
        self.calls.append(("Sandbox.create", {"args": list(args), **kwargs}))
        address = self.addresses[kwargs["image"]]
        port = kwargs["unencrypted_ports"][0]

        async def ready(**options: int) -> None:
            self.calls.append((f"{sandbox_id}.wait_until_ready", options))

        async def tunnels() -> dict[int, SimpleNamespace]:
            return {port: SimpleNamespace(tcp_socket=address)}

        async def terminate() -> None:
            self.calls.append((f"{sandbox_id}.terminate", {}))

        async def upload(source: Path, target: str) -> None:
            self.calls.append((f"{sandbox_id}.copy_from_local", {target: describe(source)}))

        async def download(source: str, target: Path) -> None:
            self.calls.append((f"{sandbox_id}.copy_to_local", {"source": source}))
            with tarfile.open(target, "w:gz"):
                pass

        async def run(*command: str, **options: Any) -> SimpleNamespace:
            self.calls.append((f"{sandbox_id}.exec", {"args": list(command), **options}))
            answer = next(
                (rule for rule in self.execs if rule.match in " ".join(command)), Exec("")
            )

            async def wait() -> int:
                if answer.hang:
                    await asyncio.Event().wait()
                return answer.returncode

            async def read(text: str) -> str:
                return text

            if "logs" in command:
                return SimpleNamespace(
                    wait=aio(wait),
                    stdout=Stream(answer.stdout, self.hold),
                    stderr=Stream(answer.stderr, self.hold),
                )
            return SimpleNamespace(
                wait=aio(wait),
                stdout=SimpleNamespace(read=aio(lambda: read(answer.stdout))),
                stderr=SimpleNamespace(read=aio(lambda: read(answer.stderr))),
            )

        return SimpleNamespace(
            object_id=sandbox_id,
            wait_until_ready=aio(ready),
            tunnels=aio(tunnels),
            terminate=aio(terminate),
            stdout=Stream(self.output[0], self.hold),
            stderr=Stream(self.output[1], self.hold),
            filesystem=SimpleNamespace(copy_from_local=aio(upload), copy_to_local=aio(download)),
            exec=aio(run),
        )


def describe(path: Path) -> Any:
    """What an uploaded file holds: an archive's members, a JSON document, or its text."""
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            return sorted(member.name for member in archive.getmembers())
    if path.suffix == ".json" and path.name != "docker-seccomp.json":
        return json.loads(path.read_text())
    return path.name if path.name == "docker-seccomp.json" else path.read_text()


def tree_hash(path: str) -> str:
    """A context tree's fingerprint: the same files give the same hash."""
    digest = hashlib.md5(usedforsecurity=False)
    root = Path(path)
    for file in sorted(entry for entry in root.rglob("*") if entry.is_file()):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


class NotFound(Exception):
    pass


class ObjectStorage:
    """The SDK's uploader, whose hasher the runtime borrows to compare builds."""

    async def _compute_hash_for_path_md5(
        self, path: str, archive_base_path: str | None = None
    ) -> str:
        del archive_base_path
        return tree_hash(path)


@dataclass
class BuiltImage:
    """A ``daytona.Image`` built from a Dockerfile and a context directory."""

    context: Path
    dockerfile_text: str = "FROM python:3.12\nCOPY . /app\n"

    def __post_init__(self) -> None:
        self._context_list = [SimpleNamespace(source_path=str(self.context), archive_path=".")]

    def dockerfile(self) -> str:
        return self.dockerfile_text


@dataclass
class FakeDaytona:
    """The ``daytona`` and ``asyncssh`` modules. Every sandbox forwards to ``port``."""

    port: int
    calls: list[tuple[str, Any]] = field(default_factory=list)
    snapshots: dict[str, Any] = field(default_factory=dict)
    deleting: dict[str, int] = field(default_factory=dict)
    output: tuple[str, str] = ("", "")
    hold: asyncio.Event | None = None
    delete_fails: bool = False
    env_output: str = ""
    sandboxes: int = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeDaytona:
        fake = self

        class AsyncDaytona:
            async def __aenter__(self) -> SimpleNamespace:
                return SimpleNamespace(
                    create=fake.create,
                    delete=fake.delete,
                    snapshot=SimpleNamespace(
                        get=fake.get_snapshot,
                        create=fake.create_snapshot,
                        delete=fake.delete_snapshot,
                    ),
                )

            async def __aexit__(self, *exc: object) -> None:
                fake.calls.append(("close", {}))

        daytona = ModuleType("daytona")
        daytona.__dict__.update(
            AsyncDaytona=AsyncDaytona,
            CreateSandboxFromImageParams=lambda **params: {"from_image": params},
            CreateSandboxFromSnapshotParams=lambda **params: {"from_snapshot": params},
            CreateSnapshotParams=lambda **params: params,
            DaytonaNotFoundError=NotFound,
            GpuType=lambda name: f"gpu:{name}",
            Image=SimpleNamespace(base=lambda name: f"base:{name}"),
            Resources=lambda **resources: SimpleNamespace(
                **{key: None for key in ("cpu", "memory", "disk", "gpu", "gpu_type")} | resources
            ),
            SessionExecuteRequest=lambda **request: request,
        )
        storage = ModuleType("daytona._async.object_storage")
        storage.__dict__["AsyncObjectStorage"] = ObjectStorage
        asyncssh = ModuleType("asyncssh")
        asyncssh.__dict__["connect"] = self.connect
        monkeypatch.setitem(sys.modules, "daytona", daytona)
        monkeypatch.setitem(sys.modules, "daytona._async", ModuleType("daytona._async"))
        monkeypatch.setitem(sys.modules, "daytona._async.object_storage", storage)
        monkeypatch.setitem(sys.modules, "asyncssh", asyncssh)
        return self

    async def create(self, params: dict[str, Any], **options: int) -> SimpleNamespace:
        self.sandboxes += 1
        sandbox_id = f"sandbox-{self.sandboxes}"
        self.calls.append(("create", {"params": plain(params), **options}))
        fake = self

        async def create_session(session: str) -> None:
            fake.calls.append((f"{sandbox_id}.create_session", session))

        async def execute(session: str, request: dict[str, Any]) -> SimpleNamespace:
            fake.calls.append((f"{sandbox_id}.execute_session_command", [session, request]))
            return SimpleNamespace(cmd_id="cmd-1")

        async def stream_logs(
            session: str,
            cmd_id: str,
            on_stdout: Callable[[str], None],
            on_stderr: Callable[[str], None],
        ) -> None:
            del session, cmd_id
            on_stdout(fake.output[0])
            on_stderr(fake.output[1])
            if fake.hold is not None:
                await fake.hold.wait()

        async def logs(session: str, cmd_id: str) -> SimpleNamespace:
            del session, cmd_id
            return SimpleNamespace(stderr=fake.env_output, output="", stdout="")

        async def ssh_access(*, expires_in_minutes: int) -> SimpleNamespace:
            fake.calls.append((f"{sandbox_id}.create_ssh_access", expires_in_minutes))
            return SimpleNamespace(token=f"token-{sandbox_id}")

        return SimpleNamespace(
            id=sandbox_id,
            process=SimpleNamespace(
                create_session=create_session,
                execute_session_command=execute,
                get_session_command_logs_async=stream_logs,
                get_session_command_logs=logs,
            ),
            create_ssh_access=ssh_access,
        )

    async def delete(self, sandbox: SimpleNamespace) -> None:
        self.calls.append(("delete", sandbox.id))
        if self.delete_fails:
            raise RuntimeError("daytona API unreachable")

    async def get_snapshot(self, name: str) -> SimpleNamespace:
        self.calls.append(("snapshot.get", name))
        if name in self.deleting:
            # The service frees a deleted name a little later, not at once.
            self.deleting[name] -= 1
            if self.deleting[name] >= 0:
                return SimpleNamespace(name=name)
            del self.deleting[name]
        if name not in self.snapshots:
            raise NotFound(name)
        return self.snapshots[name]

    async def create_snapshot(self, params: dict[str, Any]) -> None:
        name, image = params["name"], params["image"]
        self.calls.append(("snapshot.create", plain(params)))
        if name in self.snapshots or name in self.deleting:
            raise ValueError(f"snapshot {name} already exists")
        built = None
        if not isinstance(image, str):
            built = SimpleNamespace(
                dockerfile_content=image.dockerfile(),
                context_hashes=[tree_hash(str(image.context))],
            )
        self.snapshots[name] = SimpleNamespace(
            name=name, image_name=image if isinstance(image, str) else None, build_info=built
        )

    async def delete_snapshot(self, snapshot: SimpleNamespace) -> None:
        self.calls.append(("snapshot.delete", snapshot.name))
        del self.snapshots[snapshot.name]
        self.deleting[snapshot.name] = 1

    def connect(self, host: str, **options: Any) -> Any:
        fake = self

        class Connection:
            async def __aenter__(self) -> SimpleNamespace:
                fake.calls.append(("ssh.connect", {"host": host, **options}))

                async def forward(*target: Any) -> SimpleNamespace:
                    fake.calls.append(("ssh.forward_local_port", list(target)))
                    return SimpleNamespace(get_port=lambda: fake.port)

                return SimpleNamespace(forward_local_port=forward)

            async def __aexit__(self, *exc: object) -> None:
                fake.calls.append(("ssh.close", {}))

        return Connection()


def plain(value: Any) -> Any:
    """SDK parameter objects as plain data, for transcripts."""
    if isinstance(value, SimpleNamespace):
        return {key: plain(item) for key, item in vars(value).items() if item is not None}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, BuiltImage):
        return "built image"
    return value
