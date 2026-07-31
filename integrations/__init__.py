"""Interop frontends: foreign benchmark formats as HUD primitives.

These live outside the ``hud`` package and ship no part of it — core knows
only :class:`hud.environment.Integration` and imports no implementation.

Each format implements that contract (``load`` / ``environment``) and keeps
an ergonomic function surface (``harbor.load(...)``) plus format extras:
``harbor.detect`` recognizes the layout, ``harbor.adapt`` packages the
constructor into container images and returns bound rows, and
``harbor.export`` is the reverse direction.
"""
