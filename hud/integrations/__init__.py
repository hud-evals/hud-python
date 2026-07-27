"""Interop frontends: foreign benchmark formats as HUD primitives.

Each format implements :class:`hud.environment.Integration` — see its module
docstring for the contract (``load`` / ``environment``). Format
modules also keep an ergonomic function surface (``harbor.load(...)``) and
format extras: ``harbor.detect`` recognizes the layout, ``harbor.adapt``
packages the constructor into container images, ``harbor.export`` is the
reverse direction.
"""
