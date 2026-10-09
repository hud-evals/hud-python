"""A fake ``docker`` executable for runtime and Harbor scenarios.

:class:`FakeDocker` puts a ``docker`` script first on ``PATH``. Every invocation
appends its argv, and the contents of each file it names with ``-f``/``--file``,
to a log; then it answers from the first rule whose regex matches the argv
joined by spaces. Tests add rules ahead of the defaults with :meth:`FakeDocker.on`.
:meth:`FakeDocker.images` adds an image store that answers ``version``,
``build``, ``compose build|pull`` and ``image inspect`` between the two.
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
import json, os, pathlib, re, runpy, sys, time

state = pathlib.Path(os.environ["FAKE_DOCKER_STATE"])
argv = sys.argv[1:]
files = {{}}
for flag, value in zip(argv, argv[1:]):
    if flag in ("-f", "--file") and pathlib.Path(value).is_file():
        files[value] = pathlib.Path(value).read_text()
with (state / "calls.jsonl").open("a") as log:
    log.write(json.dumps({{"argv": argv, "files": files}}) + "\\n")
joined = " ".join(argv)
rules = json.loads((state / "rules.json").read_text())
for default in (False, True):
    for rule in rules:
        if rule.get("default", False) == default and re.search(rule["match"], joined):
            time.sleep(rule["delay"])
            sys.stdout.write(rule["stdout"])
            sys.stderr.write(rule["stderr"])
            sys.exit(rule["exit"])
    if not default and (state / "store.py").is_file():
        runpy.run_path(str(state / "store.py"), init_globals={{"state": state, "argv": argv}})
"""

# Runs inside the fake ``docker`` with ``state`` and ``argv`` bound; exits when it
# answers, returns to the default rules otherwise.
STORE = """
import json, pathlib, sys

store = json.loads((state / "store.json").read_text())
root = pathlib.Path(store["root"])


def done(stdout="", stderr="", code=0):
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    sys.exit(code)


def save():
    (state / "store.json").write_text(json.dumps(store))


def built(tag, context):
    try:
        key = pathlib.Path(context).resolve().relative_to(root).as_posix()
    except ValueError:
        key = str(context)
    record = store["builds"].get(key, {})
    store["images"][tag] = {
        "config": record.get("config", {}),
        "platform": record.get("platform", store["platform"]),
    }
    save()


def option(name):
    return argv[argv.index(name) + 1] if name in argv else None


if argv[:1] == ["version"]:
    fmt = option("--format")
    answers = {
        "{{.Server.Version}}": store["version"],
        "{{.Server.Os}}/{{.Server.Arch}}": store["platform"],
    }
    done(answers.get(fmt, store["version"]) + "\\n")
if argv[:2] == ["image", "inspect"]:
    image = argv[-1]
    record = store["images"].get(image)
    if record is None:
        done(stderr=f"Error response from daemon: No such image: {image}\\n", code=1)
    if option("--format") == "{{.Os}}/{{.Architecture}}":
        done(record["platform"] + "\\n")
    done(json.dumps([{"Config": record["config"]}]))
if argv[:1] == ["build"]:
    built(option("--tag"), argv[-1])
    done()
if argv[:1] == ["compose"] and argv[-2] in ("build", "pull"):
    operation, name = argv[-2:]
    directory = pathlib.Path(option("--project-directory") or ".")
    service = {}
    for flag, value in zip(argv, argv[1:]):
        if flag in ("-f", "--file"):
            document = json.loads(pathlib.Path(value).read_text())
            service.update(document.get("services", {}).get(name, {}))
    image = service.get("image")
    if operation == "build":
        build = service.get("build", {})
        context = build if isinstance(build, str) else build.get("context", ".")
        built(image, directory / context)
        done()
    if image in store["pulls"]:
        record = store["pulls"][image]
        store["images"][image] = {
            "config": record.get("config", {}),
            "platform": record.get("platform", store["platform"]),
        }
        save()
        done()
    done(stderr=f"Error response from daemon: pull access denied for {image}\\n", code=1)
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
        self._rule(r".", default=True)
        self._rule(r"^volume create ", stdout="volume\n", default=True)
        self._rule(r"^exec .* df -Pk", stdout=DF_OUTPUT, default=True)
        self._rule(r"^run ", stdout="cid-1\n", default=True)

    def on(
        self,
        match: str,
        *,
        stdout: str = "",
        stderr: str = "",
        exit: int = 0,
        delay: float = 0.0,
    ) -> None:
        """Answer invocations whose argv matches the regex ``match``; newest rules win."""
        self._rule(match, stdout=stdout, stderr=stderr, exit=exit, delay=delay, default=False)

    def images(
        self,
        root: Path,
        *,
        builds: dict[str, dict[str, Any]] | None = None,
        pulls: dict[str, dict[str, Any]] | None = None,
        platform: str = "linux/amd64",
        version: str = "28.3.3",
    ) -> None:
        """Answer image commands from a store, after the rules added with :meth:`on`.

        A ``build`` (``docker build`` or ``docker compose build``) stores its tag
        with the record in ``builds`` keyed by the build context's path relative
        to ``root``; a ``compose pull`` stores the record in ``pulls`` for its
        image and fails for any other. A record is ``{"config": <OCI config>,
        "platform": "os/arch"}``; both keys are optional, and the platform
        defaults to the server's. ``image inspect`` answers from the stored
        images and ``version`` reports ``version`` and ``platform``.
        """
        store = {
            "root": str(root.resolve()),
            "builds": builds or {},
            "pulls": pulls or {},
            "images": {},
            "platform": platform,
            "version": version,
        }
        (self._state / "store.json").write_text(json.dumps(store))
        (self._state / "store.py").write_text(STORE)

    def _rule(
        self,
        match: str,
        *,
        stdout: str = "",
        stderr: str = "",
        exit: int = 0,
        delay: float = 0.0,
        default: bool,
    ) -> None:
        self._rules.insert(
            0,
            {
                "match": match,
                "stdout": stdout,
                "stderr": stderr,
                "exit": exit,
                "delay": delay,
                "default": default,
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
