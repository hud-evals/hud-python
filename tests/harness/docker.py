"""A fake ``docker`` executable for runtime and Harbor scenarios.

:class:`FakeDocker` puts a ``docker`` script first on ``PATH``. Every invocation
appends its argv, and the contents of each file it names with ``-f``/``--file``,
to a log; then it answers from the first rule whose regex matches the argv
joined by spaces. Tests add rules ahead of the defaults with :meth:`FakeDocker.on`.

A rule given a ``rootfs`` directory emulates ``docker cp`` and ``docker exec``
instead of answering: ``<rootfs>/<container>`` stands in for each container's
filesystem, ``cp`` copies between it and the host, and ``exec`` runs the
command on the host with its absolute-path arguments moved under it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

SCRIPT = """#!{python}
import json, os, pathlib, re, shutil, subprocess, sys, time

state = pathlib.Path(os.environ["FAKE_DOCKER_STATE"])
argv = sys.argv[1:]
files = {{}}
for flag, value in zip(argv, argv[1:]):
    if flag in ("-f", "--file") and pathlib.Path(value).is_file():
        files[value] = pathlib.Path(value).read_text()
with (state / "calls.jsonl").open("a") as log:
    log.write(json.dumps({{"argv": argv, "files": files}}) + "\\n")
joined = " ".join(argv)


def inside(root, reference):
    container, _, path = reference.partition(":")
    return root / container / path.lstrip("/")


def emulate(root):
    if argv[0] == "cp":
        source, target = (
            inside(root, item) if re.match(r"^[^:/]+:/", item) else pathlib.Path(item)
            for item in argv[-2:]
        )
        if argv[-2].endswith("/."):
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            if target.is_dir():
                target = target / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, target)
        return 0
    index = 1
    while argv[index].startswith("-"):
        index += 2 if argv[index] in ("--user", "-u", "--env", "-e", "--workdir", "-w") else 1
    container_root = root / argv[index]
    command = [
        str(container_root / item.lstrip("/")) if item.startswith("/") else item
        for item in argv[index + 1 :]
    ]
    return subprocess.run(command, check=False).returncode


for rule in json.loads((state / "rules.json").read_text()):
    if re.search(rule["match"], joined):
        time.sleep(rule["delay"])
        if rule.get("rootfs"):
            sys.exit(emulate(pathlib.Path(rule["rootfs"])))
        sys.stdout.write(rule["stdout"])
        sys.stderr.write(rule["stderr"])
        sys.exit(rule["exit"])
"""

DF_OUTPUT = (
    "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
    "overlay 104857600 0 104857600 0% /\n"
)


@dataclass(frozen=True)
class Call:
    argv: list[str]
    files: dict[str, str]

    @property
    def command(self) -> str:
        return " ".join(self.argv)


class FakeDocker:
    """The fake ``docker`` on ``PATH``. Use the ``fake_docker`` fixture."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if sys.platform == "win32":
            raise RuntimeError("the fake docker executable is POSIX-only")
        self._state = root / "state"
        bin_dir = root / "bin"
        self._state.mkdir(parents=True)
        bin_dir.mkdir(parents=True)
        executable = bin_dir / "docker"
        executable.write_text(SCRIPT.format(python=sys.executable))
        executable.chmod(0o755)
        (self._state / "calls.jsonl").touch()
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv("FAKE_DOCKER_STATE", str(self._state))
        self._rules: list[dict[str, Any]] = []
        self.on(r".", stdout="")
        self.on(r"^volume create ", stdout="volume\n")
        self.on(r"^exec .* df -Pk", stdout=DF_OUTPUT)
        self.on(r"^run ", stdout="cid-1\n")

    def on(
        self,
        match: str,
        *,
        stdout: str = "",
        stderr: str = "",
        exit: int = 0,
        delay: float = 0.0,
        rootfs: Path | None = None,
    ) -> None:
        """Answer invocations whose argv matches the regex ``match``; newest rules win.

        With ``rootfs``, a matching ``cp`` or ``exec`` runs against container
        filesystems under that directory instead of answering.
        """
        self._rules.insert(
            0,
            {
                "match": match,
                "stdout": stdout,
                "stderr": stderr,
                "exit": exit,
                "delay": delay,
                "rootfs": str(rootfs) if rootfs is not None else None,
            },
        )
        (self._state / "rules.json").write_text(json.dumps(self._rules))

    @property
    def calls(self) -> list[Call]:
        lines = (self._state / "calls.jsonl").read_text().splitlines()
        return [Call(**json.loads(line)) for line in lines if line]

    def commands(self, prefix: str = "") -> list[str]:
        """Invocations so far as command lines, those starting with ``prefix`` when given."""
        return [call.command for call in self.calls if call.command.startswith(prefix)]
