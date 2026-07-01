"""Integrations: loaders that bring foreign task formats into the HUD runtime.

Everything that authors tasks — HUD's own ``env.py``, platform rows, Harbor
dirs, Verifiers/Inspect datasets — is a *frontend* loading into the same
primitives. Integrations are **loaders, not converters**: no codegen roundtrip
to run foreign tasks.

This package lives outside ``hud`` on purpose: loaders are recipes built on the
public SDK surface (``Environment``, ``Task``, ``Taskset``, ``Runtime``). Copy a
loader into your project or run it from a checkout. The CLI may call selected
integrations explicitly for polished interop paths. A repo-maintained
integration may also expose a local provider for that explicit CLI path; that
provider is SDK implementation code, not the portable loader contract.

The contract: an integration module exposes ``detect(path) -> bool`` and
``load(path) -> Taskset``. Placement stays an execution-time concern — loaders
never bake in where the substrate runs; infra integrations are *providers*
(``Callable[[Task], AsyncContextManager[Runtime]]``) passed at run time via
``runtime=``. An integration may also expose the reverse direction (e.g.
``integrations.harbor.export``).
"""

from __future__ import annotations
