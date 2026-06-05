"""Sandbox: the substrate spinup layer, decoupled from the client/server.

A ``Sandbox`` brings up a substrate that serves the HUD control channel and exposes
its ``runtime`` (url + params) — a local process (``LocalSandbox``), an attached url
(``RemoteSandbox``), or a HUD-hosted box (``HudSandbox``). ``launch`` wires it to a
``HudClient``::

    async with LocalSandbox(env) as runtime:  # create() on enter, terminate() on exit
        ...                                   # connect a client to runtime.url
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType, TracebackType

    from hud.environment import Environment


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
    """Serve a live in-process ``Environment`` on an ephemeral loopback port."""

    def __init__(self, env: Environment, host: str = "127.0.0.1") -> None:
        self._env = env
        self._host = host
        self._server: asyncio.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None

    async def create(self) -> Runtime:
        await self._env.start()  # bring up backing cap daemons before publishing the manifest
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
        await self._env.stop()
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

    ``create`` provisions a box from ``image`` and returns its ``Runtime`` (url +
    token); ``terminate`` releases it. Only the two control-plane HTTP calls
    (``_provision`` / ``_deprovision``) are left as seams to wire to the backend.
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


def as_sandbox(ref: Sandbox | Environment) -> Sandbox:
    """Resolve a ``ref`` to a ``Sandbox``: a ``Sandbox`` as-is, a live
    ``Environment`` wrapped in a ``LocalSandbox``."""
    from hud.environment import Environment  # local import: avoid import cycle at module load

    if isinstance(ref, Sandbox):
        return ref
    if isinstance(ref, Environment):
        return LocalSandbox(ref)
    raise TypeError(
        f"expected a Sandbox or a live Environment; got {type(ref).__name__}. "
        "For HUD-hosted / image envs, pass a Sandbox (e.g. HudSandbox, RemoteSandbox).",
    )


def load_module(path: str | Path) -> ModuleType:
    """Import a Python file as a throwaway module and return it.

    Shared by env-ref resolution (``module`` refs) and the CLI's variant
    collector. The file's directory is on ``sys.path`` during import so sibling
    imports resolve; the temporary module name is cleaned up afterward.
    """
    file = Path(path).resolve()
    if not file.is_file():
        raise FileNotFoundError(f"module not found: {path}")

    mod_name = f"_hud_mod_{file.stem}_{abs(hash(str(file)))}"
    spec = importlib.util.spec_from_file_location(mod_name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module: {file}")

    parent = str(file.parent)
    inserted = parent not in sys.path
    if inserted:
        sys.path.insert(0, parent)
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(parent)
        sys.modules.pop(mod_name, None)


def sandbox_from_ref(ref: dict[str, Any]) -> Sandbox:
    """Resolve a serialized env reference to a :class:`Sandbox`.

    The ref is tagged by ``type`` — the one place a stored env identity becomes a
    runnable substrate:

    - ``{"type": "module", "module": "env.py", "name": "my-env"?}`` →
      :class:`LocalSandbox` over the ``Environment`` imported from that file.
    - ``{"type": "url", "url": "tcp://host:port", "params": {...}?}`` →
      :class:`RemoteSandbox` attached to an already-running control channel.
    - ``{"type": "hud", "name": "my-env", "opts": {...}?}`` →
      :class:`HudSandbox` provisioned from the HUD registry by name (HUD-hosted).
    """
    from hud.environment import Environment  # local import: avoid import cycle at module load

    kind = ref.get("type")
    if kind == "module":
        module = ref.get("module")
        if not isinstance(module, str):
            raise ValueError("env-ref type 'module' requires a string 'module' path")
        wanted = ref.get("name")
        envs = [v for v in vars(load_module(module)).values() if isinstance(v, Environment)]
        if wanted is not None:
            envs = [e for e in envs if e.name == wanted]
        if not envs:
            raise ValueError(
                f"no Environment{f' named {wanted!r}' if wanted else ''} found in {module}",
            )
        if len(envs) > 1:
            raise ValueError(f"multiple Environments in {module}; add a 'name' to the env-ref")
        return LocalSandbox(envs[0])
    if kind == "url":
        url = ref.get("url")
        if not isinstance(url, str):
            raise ValueError("env-ref type 'url' requires a string 'url'")
        return RemoteSandbox(url, **(ref.get("params") or {}))
    if kind == "hud":
        name = ref.get("name") or ref.get("image")
        if not isinstance(name, str):
            raise ValueError("env-ref type 'hud' requires a string 'name'")
        return HudSandbox(name, **(ref.get("opts") or {}))
    raise ValueError(f"unknown env-ref type {kind!r} (expected 'module', 'url', or 'hud')")


__all__ = [
    "HudSandbox",
    "LocalSandbox",
    "RemoteSandbox",
    "Runtime",
    "Sandbox",
    "as_sandbox",
    "load_module",
    "sandbox_from_ref",
]
