"""Sandbox: the substrate spinup layer, decoupled from the client/server.

A ``Sandbox`` knows how to *bring up* a substrate that serves the HUD control
channel and expose its ``runtime`` — the connectable thing (a control-channel
url + params). It can do whatever it needs: run a local process, a container,
or call HUD infra / a third party to provision a remote box. The transport
(``HudClient``) and the env server know nothing about ``Sandbox``; the
client-side ``launch`` helper sits on top and wires the two together.

    sandbox = LocalSandbox(env)          # or HudSandbox(...), RemoteSandbox(...)
    async with sandbox as runtime:       # create() on enter, terminate() on exit
        ...                              # connect a client to runtime.url
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

    from hud.env import Env


@dataclass(frozen=True, slots=True)
class Runtime:
    """A created sandbox's connectable control channel.

    ``url`` is the control-channel address (``tcp://127.0.0.1:7000`` for a local
    process, or a remote ``tcp://sandbox-abc.hud.so:443``). ``params`` carries
    connection-time data a transport may need — e.g. an auth token or sandbox id.
    """

    url: str
    params: dict[str, Any] = field(default_factory=dict)


class Sandbox(ABC):
    """A spinnable substrate that exposes a HUD control channel.

    Subclasses implement ``create`` (provision + return the ``Runtime``) and
    ``terminate`` (release it) — they may do anything to get there. Use as an
    async context manager so teardown is guaranteed. Whoever creates it owns
    termination.
    """

    _runtime: Runtime | None = None

    @abstractmethod
    async def create(self) -> Runtime:
        """Bring the substrate up and return its connectable ``Runtime``."""

    @abstractmethod
    async def terminate(self) -> None:
        """Release the substrate (stop the process / container / remote box)."""

    @property
    def runtime(self) -> Runtime:
        """The connectable ``Runtime`` (after ``create``)."""
        if self._runtime is None:
            raise RuntimeError("sandbox not created; call create() first")
        return self._runtime

    async def __aenter__(self) -> Runtime:
        return await self.create()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.terminate()


class LocalSandbox(Sandbox):
    """Serve a live in-process ``Env`` on an ephemeral loopback port."""

    def __init__(self, env: Env, host: str = "127.0.0.1") -> None:
        self._env = env
        self._host = host
        self._server: asyncio.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None

    async def create(self) -> Runtime:
        self._server = await self._env.bind(self._host, 0)
        host, port = self._server.sockets[0].getsockname()[:2]
        self._serve_task = asyncio.create_task(self._server.serve_forever())
        self._runtime = Runtime(url=f"tcp://{host}:{port}")
        return self._runtime

    async def terminate(self) -> None:
        if self._serve_task is not None:
            self._serve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._serve_task
            self._serve_task = None
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        self._runtime = None


class RemoteSandbox(Sandbox):
    """Attach to a control channel provisioned elsewhere (an already-known url).

    Does not provision anything — ``create`` just returns the configured
    ``Runtime``. Use this to point at a box you (or some other system) brought up.
    """

    def __init__(self, url: str, **params: Any) -> None:
        self._url = url
        self._params = params

    async def create(self) -> Runtime:
        self._runtime = Runtime(url=self._url, params=self._params)
        return self._runtime

    async def terminate(self) -> None:
        self._runtime = None


class HudSandbox(Sandbox):
    """A HUD-hosted sandbox, provisioned via the HUD control plane.

    Lifecycle:
      ``create``    — provision a box from ``image`` (``_provision``) and return
                      its ``Runtime`` (control-channel url + auth token).
      ``terminate`` — release the box (``_deprovision``).

    The orchestration (provision → runtime, and teardown) is implemented here;
    only the two HTTP calls to the HUD control plane (``_provision`` /
    ``_deprovision``) are left as seams to wire to the backend. Waiting for the
    control channel to accept connections is the client's job (``launch`` retries
    the connect), not the sandbox's.
    """

    def __init__(
        self,
        image: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        **opts: Any,
    ) -> None:
        self.image = image
        self.base_url = base_url  # HUD control-plane base URL; defaults to settings
        self.api_key = api_key
        self.opts = opts
        self.sandbox_id: str | None = None

    async def create(self) -> Runtime:
        provisioned = await self._provision()
        self.sandbox_id = provisioned["id"]
        self._runtime = Runtime(
            url=provisioned["control_url"],
            params={"token": provisioned["token"], "sandbox_id": provisioned["id"]},
        )
        return self._runtime

    async def terminate(self) -> None:
        if self.sandbox_id is not None:
            with contextlib.suppress(Exception):
                await self._deprovision(self.sandbox_id)
            self.sandbox_id = None
        self._runtime = None

    # ─── HUD control-plane API (structure only — wire to the real endpoints) ───

    async def _provision(self) -> dict[str, Any]:
        """Provision a sandbox on HUD infra.

        Intended call: ``POST {base_url}/sandboxes`` with
        ``{"image": self.image, **self.opts}`` and a bearer ``api_key``, returning
        ``{"id": str, "control_url": "tcp://host:port", "token": str}``.
        """
        raise NotImplementedError("HudSandbox._provision: HUD spinup API not wired yet")

    async def _deprovision(self, sandbox_id: str) -> None:
        """Release a provisioned sandbox.

        Intended call: ``DELETE {base_url}/sandboxes/{sandbox_id}``.
        """
        raise NotImplementedError("HudSandbox._deprovision: HUD spinup API not wired yet")


def as_sandbox(ref: Sandbox | Env) -> Sandbox:
    """Resolve a ``ref`` to a ``Sandbox``: a ``Sandbox`` as-is, a live ``Env``
    wrapped in a ``LocalSandbox``."""
    from hud.env import Env  # local import: avoid import cycle at module load

    if isinstance(ref, Sandbox):
        return ref
    if isinstance(ref, Env):
        return LocalSandbox(ref)
    raise TypeError(
        f"expected a Sandbox or a live Env; got {type(ref).__name__}. "
        "For HUD-hosted / image envs, pass a Sandbox (e.g. HudSandbox, RemoteSandbox).",
    )


__all__ = [
    "HudSandbox",
    "LocalSandbox",
    "RemoteSandbox",
    "Runtime",
    "Sandbox",
    "as_sandbox",
]
