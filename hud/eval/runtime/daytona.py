"""Daytona runtime provider."""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from hud.utils.process import finish_output, output_writer

from .core import Runtime, RuntimeConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from daytona import Image as DaytonaImage
    from daytona.common.snapshot import Snapshot as DaytonaSnapshot

    from hud.eval.task import Task

logger = logging.getLogger("hud.eval.runtime")


async def _snapshot_is_current(
    snapshot: DaytonaSnapshot,
    image: str | DaytonaImage,
) -> bool:
    """Whether *snapshot* was built from *image* as it exists right now.

    Daytona records what a snapshot was built from — the registry ref, or for a
    built ``Image`` its Dockerfile text plus the hashes of the context it
    uploads (``build_info``). The hashes are recomputed here with the SDK's own
    hasher so they are comparable to what ``snapshot.create`` would upload.
    ``_context_list`` is the same private attribute ``snapshot.create`` reads.
    """
    if isinstance(image, str):
        return bool(snapshot.image_name == image)
    build = snapshot.build_info
    if build is None:
        return False
    from daytona._async.object_storage import AsyncObjectStorage

    # The hasher is an instance method only for code organization; credentials
    # are needed to upload, not to hash, so skip the credentialed __init__.
    storage = AsyncObjectStorage.__new__(AsyncObjectStorage)
    hashes = [
        await storage._compute_hash_for_path_md5(entry.source_path, entry.archive_path)
        for entry in image._context_list
    ]
    return bool(
        build.dockerfile_content == image.dockerfile()
        and list(build.context_hashes or []) == hashes
    )


class DaytonaRuntime:
    """The Daytona provider: each acquisition creates a fresh sandbox from a snapshot.

    The Daytona runtime boots a sandbox from a pre-built *snapshot*
    (the durable handle, the snapshot equivalent of Modal's image name), starts the
    env's control channel inside it, then reaches it over an SSH local-forward:
    Daytona exposes services only as HTTPS previews, but :func:`hud.clients.connect`
    dials ``tcp://``, so the raw control channel is tunneled over SSH to a local
    port. Yields its :class:`Runtime`, deletes the sandbox on exit.

    Pass a snapshot name — ``DaytonaRuntime("hud-libero-env")`` — optionally with an
    ``image`` (Dockerfile/registry ref) to build that snapshot if it is missing.
    With *image*, an existing snapshot is compared against the image's recorded
    build (Dockerfile plus context hashes) and rebuilt under the same name when
    they differ, so editing the env never silently reuses the snapshot built
    before the edit.
    Resources (cpu/memory/gpu) live on the snapshot, not here. *workdir* defaults to
    ``/app`` (the scaffolded ``Dockerfile.hud`` WORKDIR) since a Daytona session
    starts in ``~``, not the image's WORKDIR; override only for a non-standard layout.
    Requires the ``daytona`` extra and ``DAYTONA_API_KEY``.
    """

    def __init__(
        self,
        snapshot_name: str | None = None,
        *,
        image: str | DaytonaImage | None = None,
        command: str | None = None,
        workdir: str | None = "/app",
        port: int = 8765,
        ssh_host: str = "ssh.app.daytona.io",
        ssh_expires_minutes: int = 24 * 60,
        runtime_config: RuntimeConfig | dict[str, Any] | None = None,
    ) -> None:
        self.snapshot_name = snapshot_name
        # Default command serves on *port*, so the SSH forward target always
        # matches what's listening; override only for a non-default layout.
        self.command = (
            command or f'PATH="$PWD/.venv/bin:$PATH" hud serve env.py --host 0.0.0.0 --port {port}'
        )
        self.workdir = workdir
        self.port = port
        self.ssh_host = ssh_host
        self.ssh_expires_minutes = ssh_expires_minutes
        config = None
        if runtime_config is not None:
            config = RuntimeConfig.model_validate(runtime_config)
        self.runtime_config = config
        # Resolve each snapshot name against the image once; lock so concurrent
        # first acquisitions resolve exactly once.
        self._image = image
        self._resolved: set[str] = set()
        self._snapshot_lock = asyncio.Lock()

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        import asyncssh
        from daytona import (
            AsyncDaytona,
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            CreateSnapshotParams,
            DaytonaNotFoundError,
            GpuType,
            Image,
            Resources,
            SessionExecuteRequest,
        )

        async with AsyncDaytona() as daytona:
            config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
            if config.mounts:
                raise ValueError("DaytonaRuntime does not support runtime_config.mounts")
            if config.compose is not None:
                raise ValueError("DaytonaRuntime does not support runtime_config.compose")
            if config.limits is not None and config.limits.run_timeout_s is not None:
                raise ValueError("DaytonaRuntime does not support runtime_config.run_timeout_s")

            daytona_resources = None
            resources = config.resources
            if resources is not None:
                resources._require_support(
                    "DaytonaRuntime", {"cpu", "memory_mb", "storage_mb", "gpu"}
                )
                resource_kwargs: dict[str, Any] = {}
                if resources.cpu is not None:
                    # Daytona allocates whole cores; truncating resizes silently.
                    if isinstance(resources.cpu, float) and not resources.cpu.is_integer():
                        raise ValueError(
                            f"DaytonaRuntime needs a whole number of CPUs, got {resources.cpu}"
                        )
                    resource_kwargs["cpu"] = int(resources.cpu)
                if resources.memory_mb is not None:
                    resource_kwargs["memory"] = max(
                        1,
                        (resources.memory_mb + 1023) // 1024,
                    )
                if resources.storage_mb is not None:
                    resource_kwargs["disk"] = max(
                        1,
                        (resources.storage_mb + 1023) // 1024,
                    )
                if resources.gpu is not None:
                    resource_kwargs["gpu"] = resources.gpu.count
                    gpu_types = resources.gpu.acceptable_types
                    if gpu_types:
                        resource_kwargs["gpu_type"] = [GpuType(item) for item in gpu_types]
                if resource_kwargs:
                    daytona_resources = Resources(**resource_kwargs)

            if config.image is not None:
                sandbox_params = CreateSandboxFromImageParams(
                    image=Image.base(config.image),
                    ephemeral=True,
                    auto_stop_interval=0,
                    resources=daytona_resources,
                )
            else:
                snapshot_name = self.snapshot_name
                snapshot_image = self._image
                if snapshot_name is None:
                    raise ValueError(
                        "DaytonaRuntime requires snapshot_name or runtime_config.image"
                    )
                if daytona_resources is not None and snapshot_image is None:
                    raise ValueError(
                        "DaytonaRuntime cannot resize an already-built snapshot: resources "
                        "are fixed when it is built, so pass image= to build one"
                    )
                if snapshot_image is not None:
                    if daytona_resources is not None:
                        # Sizing is baked in at build time, so each sizing is its
                        # own snapshot under a readable suffix (env-4cpu-8gb).
                        sizing = []
                        if daytona_resources.cpu:
                            sizing.append(f"{daytona_resources.cpu}cpu")
                        if daytona_resources.memory:
                            sizing.append(f"{daytona_resources.memory}gb")
                        if resources is not None and resources.storage_mb:
                            storage_gb = max(
                                1,
                                (resources.storage_mb + 1023) // 1024,
                            )
                            sizing.append(f"{storage_gb}gb-disk")
                        if daytona_resources.gpu:
                            sizing.append(f"{daytona_resources.gpu}gpu")
                            sizing.extend(
                                str(getattr(t, "value", t)).lower()
                                for t in daytona_resources.gpu_type or []
                            )
                        snapshot_name = "-".join([snapshot_name, *sizing])
                    async with self._snapshot_lock:
                        if snapshot_name not in self._resolved:
                            try:
                                existing = await daytona.snapshot.get(snapshot_name)
                            except DaytonaNotFoundError:
                                existing = None
                            if existing is not None and not await _snapshot_is_current(
                                existing, snapshot_image
                            ):
                                logger.info(
                                    "Daytona snapshot %s is stale; rebuilding", snapshot_name
                                )
                                await daytona.snapshot.delete(existing)
                                # Deletion frees the name asynchronously (~10s
                                # observed); creating before it lands conflicts.
                                async with asyncio.timeout(120):
                                    while True:
                                        try:
                                            await daytona.snapshot.get(snapshot_name)
                                        except DaytonaNotFoundError:
                                            break
                                        await asyncio.sleep(0.5)
                                existing = None
                            if existing is None:
                                logger.info("building Daytona snapshot %s", snapshot_name)
                                await daytona.snapshot.create(
                                    CreateSnapshotParams(
                                        name=snapshot_name,
                                        image=snapshot_image,
                                        resources=daytona_resources,
                                    )
                                )
                            self._resolved.add(snapshot_name)
                sandbox_params = CreateSandboxFromSnapshotParams(
                    snapshot=snapshot_name,
                    ephemeral=True,
                    auto_stop_interval=0,
                )

            create_timeout = 120
            if config.limits is not None and config.limits.startup_timeout_s is not None:
                create_timeout = config.limits.startup_timeout_s
            # ephemeral: these sandboxes are per-rollout and deleted on exit anyway,
            # and some regions only permit ephemeral sandboxes.
            sandbox = await daytona.create(
                sandbox_params,
                timeout=create_timeout,
            )
            output_task: asyncio.Task[None] | None = None
            try:
                # Start the env server in a background session (the snapshot's CMD is
                # not the sandbox's main process). connect() retries the handshake,
                # so we don't poll for readiness here.
                session: str = "hud-serve"
                await sandbox.process.create_session(session)
                cmd = f"cd {self.workdir} && {self.command}" if self.workdir else self.command
                session_command = await sandbox.process.execute_session_command(
                    session, SessionExecuteRequest(command=cmd, run_async=True)
                )

                async def follow_logs() -> None:
                    write_stdout, finish_stdout = output_writer(sys.stdout)
                    write_stderr, finish_stderr = output_writer(sys.stderr)
                    try:
                        await sandbox.process.get_session_command_logs_async(
                            session,
                            session_command.cmd_id,
                            write_stdout,
                            write_stderr,
                        )
                    finally:
                        finish_stdout()
                        finish_stderr()

                output_task = asyncio.create_task(follow_logs())
                ssh = await sandbox.create_ssh_access(expires_in_minutes=self.ssh_expires_minutes)
                async with asyncssh.connect(
                    self.ssh_host, username=ssh.token, known_hosts=None
                ) as conn:
                    listener = await conn.forward_local_port("127.0.0.1", 0, "127.0.0.1", self.port)
                    try:
                        yield Runtime(
                            f"tcp://127.0.0.1:{listener.get_port()}",
                            params={"provider": "daytona", "instance_id": sandbox.id},
                            config=config if config.model_dump(exclude_none=True) else None,
                        )
                    except (EOFError, OSError) as exc:
                        # Why it died only exists inside the sandbox, and the
                        # sandbox may already be gone.
                        try:
                            logs = await sandbox.process.get_session_command_logs(
                                session, session_command.cmd_id
                            )
                            output = (logs.stderr or logs.output or logs.stdout or "").strip()
                        except Exception as log_exc:
                            exc.add_note(f"env output unavailable: {log_exc}")
                        else:
                            exc.add_note(
                                f"env output in sandbox {sandbox.id}:\n{output}"
                                if output
                                else "env printed nothing"
                            )
                        raise
            finally:
                try:
                    await daytona.delete(sandbox)
                except Exception:
                    # Swallowing this is how a billable sandbox outlives its process.
                    logger.warning(
                        "failed to delete Daytona sandbox %s; it may still be running",
                        sandbox.id,
                        exc_info=True,
                    )
                if output_task is not None:
                    await finish_output(output_task)
