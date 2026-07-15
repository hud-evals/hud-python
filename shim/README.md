# hud-python → hud

The HUD SDK is now published on PyPI as [`hud`](https://pypi.org/project/hud/).
The import name (`import hud`) and the CLI (`hud`) are unchanged — only the
package name on PyPI moved.

This package is an empty shim that depends on `hud`, so existing installs and
requirements files keep working. Please migrate at your convenience:

```bash
# instead of
pip install hud-python
uv tool install hud-python

# use
pip install hud
uv tool install hud
```

And replace `hud-python` with `hud` in `pyproject.toml`, `requirements.txt`,
Dockerfiles, and CI configs.

uv upgrades handle the rename in place: `uv sync --upgrade-package hud-python`,
`uv tool upgrade hud-python`, and `uv pip install -U hud-python` all land on
this shim plus `hud` cleanly.

pip in-place upgrades do not. Pre-rename `hud-python` (< 0.6.9) and `hud` ship
the same top-level `hud` package, and `pip install -U hud-python` installs
`hud` first, then deletes its files while removing the old `hud-python`. In an
environment that already has `hud-python`, uninstall first:

```bash
python -m pip uninstall -y hud-python
python -m pip install hud
```

If an in-place pip upgrade already broke an environment (`pip list` shows
`hud`, but importing it fails or the CLI reports `No module named 'hud.cli'`),
recover with:

```bash
python -m pip install --force-reinstall --no-deps hud
```

Docs: [docs.hud.ai](https://docs.hud.ai) · Source:
[github.com/hud-evals/hud-python](https://github.com/hud-evals/hud-python)
