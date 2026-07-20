"""Final release under the old 'hud-python' name; the SDK now ships as 'hud'.

Installers run this file at build time, before modifying the environment, so
it exits with migration instructions and leaves any existing install intact.
"""

import os
import sys

from setuptools import setup

MESSAGE = """
The HUD SDK was renamed on PyPI: 'hud-python' is now 'hud'.

Your environment has not been changed. To migrate (uninstalling both names
first is safe in every state, including a broken or half-migrated install):

    pip uninstall -y hud-python hud
    pip install hud

    # or for a uv-managed tool:
    uv tool uninstall hud-python; uv tool install hud --reinstall

Then replace 'hud-python' with 'hud' in requirements files, pyproject.toml,
Dockerfiles, and CI configs. The import name ('import hud') and the CLI
('hud') are unchanged.
"""

if os.environ.get("HUD_PYTHON_ALLOW_BUILD") != "1":
    sys.exit(MESSAGE)

setup()
