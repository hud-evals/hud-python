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

Note: if you have a pre-rename `hud-python` (< 0.6.9) already installed, run
`pip uninstall hud-python` before installing `hud` — both ship the same
top-level `hud` package and would overwrite each other. Upgrading in place
(`pip install -U hud-python`) is also safe: it moves you onto this shim.

Docs: [docs.hud.ai](https://docs.hud.ai) · Source:
[github.com/hud-evals/hud-python](https://github.com/hud-evals/hud-python)
