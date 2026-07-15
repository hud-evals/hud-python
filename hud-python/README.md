# hud-python → hud

The HUD SDK was renamed on PyPI: [`hud-python`](https://pypi.org/project/hud-python/)
is now [`hud`](https://pypi.org/project/hud/). To migrate:

Uninstalling both names first is safe in every state, including a broken or
half-migrated install:

```bash
pip uninstall -y hud-python hud
pip install hud
```

Then replace `hud-python` with `hud` in requirements files, `pyproject.toml`,
Dockerfiles, and CI configs. The import name (`import hud`) and the CLI
(`hud`) are unchanged.

Docs: [docs.hud.ai](https://docs.hud.ai) · Source:
[github.com/hud-evals/hud-python](https://github.com/hud-evals/hud-python)
