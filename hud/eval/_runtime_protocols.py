"""Structural contracts for lazily imported runtime provider SDKs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping, Sequence
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

T_co = TypeVar("T_co", covariant=True)


class AioMethod(Protocol[T_co]):
    async def aio(self, *args: object, **kwargs: object) -> T_co: ...


class ModalImage(Protocol):
    build: AioMethod[None]

    def env(self, variables: Mapping[str, str]) -> ModalImage: ...


class _ModalImageFactory(Protocol):
    def from_id(self, image_id: str) -> ModalImage: ...

    def from_registry(self, image: str) -> ModalImage: ...

    def from_name(self, name: str) -> ModalImage: ...


class _ModalAppFactory(Protocol):
    lookup: AioMethod[object]


class _ModalStream(Protocol):
    read: AioMethod[str]


class _ModalProcess(Protocol):
    wait: AioMethod[int]
    stderr: _ModalStream


class _ModalFilesystem(Protocol):
    copy_from_local: AioMethod[None]


class _ModalTunnel(Protocol):
    tcp_socket: tuple[str, int]


class _ModalSandbox(Protocol):
    object_id: str
    wait_until_ready: AioMethod[None]
    filesystem: _ModalFilesystem
    exec: AioMethod[_ModalProcess]
    tunnels: AioMethod[dict[int, _ModalTunnel]]
    terminate: AioMethod[None]


class _ModalSandboxFactory(Protocol):
    create: AioMethod[_ModalSandbox]


class _ModalProbeFactory(Protocol):
    def with_tcp(self, port: int) -> object: ...


class ModalModule(Protocol):
    Image: _ModalImageFactory
    App: _ModalAppFactory
    Sandbox: _ModalSandboxFactory
    Probe: _ModalProbeFactory


class DaytonaContextEntry(Protocol):
    @property
    def source_path(self) -> str | Path: ...

    @property
    def archive_path(self) -> str | Path: ...


class DaytonaImage(Protocol):
    @property
    def _context_list(self) -> Sequence[DaytonaContextEntry]: ...

    def dockerfile(self) -> str: ...


class _DaytonaBuildInfo(Protocol):
    dockerfile_content: str
    context_hashes: Sequence[str] | None


class DaytonaSnapshot(Protocol):
    image_name: str
    build_info: _DaytonaBuildInfo | None


class ObjectStorage(Protocol):
    async def _compute_hash_for_path_md5(
        self,
        source_path: str | Path,
        archive_path: str | Path,
    ) -> str: ...


class ObjectStorageModule(Protocol):
    AsyncObjectStorage: type[ObjectStorage]


class _DaytonaResources(Protocol):
    cpu: int | None
    memory: int | None
    gpu: int | None
    gpu_type: Sequence[object] | None


class _DaytonaSessionCommand(Protocol):
    cmd_id: str


class _DaytonaSessionLogs(Protocol):
    stderr: str | None
    output: str | None
    stdout: str | None


class _DaytonaProcess(Protocol):
    async def create_session(self, session: str) -> object: ...

    async def execute_session_command(
        self,
        session: str,
        request: object,
    ) -> _DaytonaSessionCommand: ...

    async def get_session_command_logs(
        self,
        session: str,
        command_id: str,
    ) -> _DaytonaSessionLogs: ...


class _DaytonaSshAccess(Protocol):
    token: str


class _DaytonaSandbox(Protocol):
    id: str
    process: _DaytonaProcess

    async def create_ssh_access(self, *, expires_in_minutes: int) -> _DaytonaSshAccess: ...


class _DaytonaSnapshotClient(Protocol):
    async def get(self, name: str) -> DaytonaSnapshot: ...

    async def delete(self, snapshot: DaytonaSnapshot) -> object: ...

    async def create(self, params: object) -> object: ...


class _DaytonaCreate(Protocol):
    def __call__(
        self,
        params: object,
        *,
        timeout: int,
    ) -> Awaitable[_DaytonaSandbox]: ...


class _DaytonaClient(Protocol):
    snapshot: _DaytonaSnapshotClient
    create: _DaytonaCreate

    async def delete(self, sandbox: _DaytonaSandbox) -> object: ...


class _DaytonaFactory(Protocol):
    def __call__(self) -> AbstractAsyncContextManager[_DaytonaClient]: ...


class _ObjectFactory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _ResourcesFactory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> _DaytonaResources: ...


class _GpuTypeFactory(Protocol):
    def __call__(self, value: str) -> object: ...


class _DaytonaImageFactory(Protocol):
    def base(self, image: str) -> object: ...


class DaytonaModule(Protocol):
    AsyncDaytona: _DaytonaFactory
    CreateSandboxFromImageParams: _ObjectFactory
    CreateSandboxFromSnapshotParams: _ObjectFactory
    CreateSnapshotParams: _ObjectFactory
    DaytonaNotFoundError: type[Exception]
    GpuType: _GpuTypeFactory
    Image: _DaytonaImageFactory
    Resources: _ResourcesFactory
    SessionExecuteRequest: _ObjectFactory
