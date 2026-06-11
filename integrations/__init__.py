"""Integrations: loaders that bring foreign task formats into the HUD runtime.

Everything that authors tasks — HUD's own ``env.py``, platform rows, Harbor
dirs, Verifiers/Inspect datasets — is a *frontend* loading into the same
primitives. Integrations are **loaders, not converters**: no codegen roundtrip
to run foreign tasks.

This package lives outside ``hud`` on purpose: each module is a recipe built
**only on the public SDK surface** (``Environment``, ``Task``,
``Taskset``, ``Runtime``) — that constraint is the proof the core is
flexible. Copy a module into your project or run it from a checkout; nothing
in the SDK or CLI imports it.

The contract: an integration module exposes ``detect(path) -> bool`` and
``load(path) -> Taskset``. Placement stays an execution-time concern — loaders
never bake in where the substrate runs; infra integrations are *providers*
(``Callable[[Task], AsyncContextManager[Runtime]]``) passed at run time via
``runtime=``. An integration may also expose the reverse direction (e.g.
``integrations.harbor.export``).
"""

from __future__ import annotations
