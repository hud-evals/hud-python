"""Tensorlake runtime provider."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from .core import Runtime, RuntimeConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval.task import Task

logger = logging.getLogger("hud.eval.runtime")


class TensorlakeRuntime:
    """The Tensorlake provider: each acquisition creates a fresh sandbox from an image.

    Boots a Firecracker sandbox from a registered Tensorlake *sandbox image*
    (the durable handle, built with ``tl sbx image create`` or
    ``tensorlake.image.Image.build``), starts the env's control channel inside
    it, then reaches it over the SDK's TCP tunnel: each connection to an
    ephemeral local port is relayed into the sandbox over a WebSocket, which
    is what :func:`hud.clients.connect` dials as ``tcp://``. Yields its
    :class:`Runtime`, terminates the sandbox on exit.

    Pass an image name — ``TensorlakeRuntime("hud-libero-env")`` — or set
    ``runtime_config.image``. Tensorlake accepts registered image names only,
    not registry references. Resources (cpu/memory/disk/gpu) and both
    ``RuntimeLimits`` map onto the sandbox at creation time. *workdir*
    defaults to ``/app`` (the scaffolded ``Dockerfile.hud`` WORKDIR) since a
    sandbox process starts in ``~``, not the image's WORKDIR.
    Requires the ``tensorlake`` extra and ``TENSORLAKE_API_KEY``.
    """

    def __init__(
        self,
        image_name: str | None = None,
        *,
        command: str | None = None,
        workdir: str | None = "/app",
        port: int = 8765,
        runtime_config: RuntimeConfig | dict[str, Any] | None = None,
    ) -> None:
        self.image_name = image_name
        # Default command serves on *port*, so the tunnel target always
        # matches what's listening; override only for a non-default layout.
        self.command = (
            command or f'PATH="$PWD/.venv/bin:$PATH" hud serve env.py --host 0.0.0.0 --port {port}'
        )
        self.workdir = workdir
        self.port = port
        config = None
        if runtime_config is not None:
            config = RuntimeConfig.model_validate(runtime_config)
        self.runtime_config = config

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        from tensorlake.sandbox import AsyncSandboxClient, GpuModel, GpuRequest, SandboxPending

        config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
        if config.compose is not None:
            raise ValueError("TensorlakeRuntime does not support runtime_config.compose")
        image = config.image if config.image is not None else self.image_name
        if image is None:
            raise ValueError("TensorlakeRuntime requires image_name or runtime_config.image")

        create_kwargs: dict[str, Any] = {"image": image}
        resources = config.resources
        if resources is not None:
            resources._require_support(
                "TensorlakeRuntime", {"cpu", "memory_mb", "storage_mb", "gpu"}
            )
            if resources.cpu is not None:
                create_kwargs["cpus"] = resources.cpu
            if resources.memory_mb is not None:
                create_kwargs["memory_mb"] = resources.memory_mb
            if resources.storage_mb is not None:
                create_kwargs["disk_mb"] = resources.storage_mb
            if resources.gpu is not None:
                gpu_types = resources.gpu.acceptable_types
                if len(gpu_types) != 1:
                    raise ValueError(
                        "TensorlakeRuntime needs exactly one GPU type in "
                        "runtime_config.resources.gpu.type"
                    )
                create_kwargs["gpu"] = GpuRequest(
                    count=resources.gpu.count, model=GpuModel(gpu_types[0])
                )
        startup_timeout = None
        if config.limits is not None:
            startup_timeout = config.limits.startup_timeout_s
            if config.limits.run_timeout_s is not None:
                create_kwargs["timeout_secs"] = config.limits.run_timeout_s

        async with AsyncSandboxClient.for_cloud() as client:
            # A sandbox bills from the moment it is created, queued or not, so
            # creating and waiting are two steps: a rollout cancelled while
            # waiting for capacity still reaches the delete below.
            pending = await client.create(**create_kwargs, wait=False)
            sandbox = None
            try:
                try:
                    sandbox = await pending.ready(startup_timeout)
                except SandboxPending as exc:
                    # The SDK's message says the sandbox keeps its queue slot;
                    # the finally below deletes it.
                    raise TimeoutError(
                        f"Tensorlake sandbox {pending.sandbox_id} was still pending after "
                        f"{exc.timeout}s ({exc.pending_reason or 'no reason given'})"
                    ) from exc
                # The image's CMD is not the sandbox's main process, so start the
                # env server explicitly. connect() retries the handshake, so we
                # don't poll for readiness here.
                process = await sandbox.start_process(
                    "sh", ["-c", self.command], working_dir=self.workdir
                )
                tunnel = await sandbox.create_tunnel(self.port, local_port=0)
                try:
                    yield Runtime(
                        f"tcp://{tunnel.local_host}:{tunnel.local_port}",
                        params={"provider": "tensorlake", "instance_id": sandbox.sandbox_id},
                        config=config if config.model_dump(exclude_none=True) else None,
                    )
                except (EOFError, OSError) as exc:
                    # Why it died only exists inside the sandbox, and the
                    # sandbox may already be gone.
                    try:
                        output = await sandbox.get_output(process.pid)
                    except Exception as log_exc:
                        exc.add_note(f"env output unavailable: {log_exc}")
                    else:
                        text = "\n".join(output.lines).strip()
                        exc.add_note(
                            f"env output in sandbox {sandbox.sandbox_id}:\n{text}"
                            if text
                            else "env printed nothing"
                        )
                    raise
                finally:
                    await tunnel.close()
            finally:
                try:
                    if sandbox is None:
                        await client.delete(pending.sandbox_id)
                    else:
                        await sandbox.terminate()
                except Exception:
                    # Swallowing this is how a billable sandbox outlives its process.
                    logger.warning(
                        "failed to terminate Tensorlake sandbox %s; it may still be running",
                        pending.sandbox_id,
                        exc_info=True,
                    )
