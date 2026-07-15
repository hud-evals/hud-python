"""Console-script entry point that survives the hud-python → hud rename.

``pip install -U hud-python`` across the rename installs the renamed ``hud``
distribution first, then deletes the shared ``hud/`` files while removing the
old ``hud-python`` (pip removes every path on the old distribution's RECORD).
What survives imports as an empty namespace package — plus any file that is
new in the renamed release, like this one. Routing the console scripts through
here turns that broken state into an actionable message instead of a bare
``ModuleNotFoundError``. This module must never gain heavy imports and must
not be moved or renamed while pre-rename installs remain in the wild.
"""

from __future__ import annotations

import sys

_RECOVERY_MESSAGE = """\
Your 'hud' installation is missing its files. This happens when pip upgrades
'hud-python' in place across its rename to 'hud': pip installs 'hud' first,
then deletes the files it shares with the old 'hud-python'. Recover with:

    python -m pip install --force-reinstall --no-deps hud
"""


def main() -> None:
    import hud

    if getattr(hud, "__file__", None) is None:
        sys.stderr.write(_RECOVERY_MESSAGE)
        raise SystemExit(1)

    from hud.cli import main as cli_main

    cli_main()
