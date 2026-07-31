"""The Harbor integration as data: load, provenance, grouping, adapt contexts.

Docker-side serving needs a daemon and is covered by the e2e integration
scripts; here ``load``'s rows/provenance/stamping and ``adapt``'s build
contexts are checked without one, exercising the integration directly (no
eval/taskset wiring).
"""

from __future__ import annotations

import os
import re
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from integrations import harbor
from integrations.harbor import _load as harbor_load

if TYPE_CHECKING:
    from pathlib import Path


def _write_harbor_task(root: Path, name: str, marker: str = "FROM python:3.12-slim\n") -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text(f"Fix {name}.\n", encoding="utf-8")
    (task / "task.toml").write_text(
        f'schema_version = "1.3"\n\n[task]\nname = "demo/{name}"\n', encoding="utf-8"
    )
    (task / "environment" / "Dockerfile").write_text(marker, encoding="utf-8")
    (task / "tests" / "test.sh").write_text(
        "#!/usr/bin/env bash\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n",
        encoding="utf-8",
    )
    return task


def test_load_stamps_rows_with_provenance(tmp_path) -> None:
    _write_harbor_task(tmp_path, "task-a")
    _write_harbor_task(tmp_path, "task-b")

    taskset = harbor.load(tmp_path)

    assert taskset.origin == f"harbor:{tmp_path.resolve()}"
    assert len(taskset) == 2
    assert all(t.runtime_config is None for t in taskset)


async def test_adapt_contexts_bake_the_serving_layer(tmp_path) -> None:
    _write_harbor_task(tmp_path, "task-a")
    _write_harbor_task(tmp_path, "task-b", marker="FROM python:3.11-slim\n")

    images = await harbor.adapt(tmp_path, build=False)

    assert images == {}
    contexts = sorted(p.name for p in (tmp_path / ".hud-adapt").iterdir())
    assert len(contexts) == 2  # two env groups (distinct Dockerfiles)
    context = tmp_path / ".hud-adapt" / contexts[0]
    dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")
    # The CMD serves the contract constructor by module reference — no baked env.py.
    assert "harbor:environment" in dockerfile
    assert f'"name={context.name}"' in dockerfile
    assert "EXPOSE 8765" in dockerfile
    assert not (context / "_hud" / "env.py").exists()
    baked = context / "_hud" / "tasks"
    (task_dir,) = list(baked.iterdir())
    assert (task_dir / "instruction.md").is_file()
    assert (task_dir / "tests" / "test.sh").is_file()


async def test_the_layer_keeps_its_own_state_out_of_the_task(tmp_path) -> None:
    # Everything the layer installs belongs under the harness's own tree: uv's
    # binary, its managed interpreter and that interpreter's shims, its cache
    # and config. At uv's defaults these land in the invoking user's home —
    # inside the task's own filesystem, where the agent finds a HUD runtime the
    # task never declared, and shims into a hidden tree resolving to nothing.
    _write_harbor_task(tmp_path, "task-a")
    await harbor.adapt(tmp_path, build=False)
    (context,) = sorted((tmp_path / ".hud-adapt").iterdir())
    script = (context / "_hud" / "install.sh").read_text(encoding="utf-8")

    directories = dict(re.findall(r"\b(UV_\w*(?:DIR|HOME)|XDG_CONFIG_HOME)=(\S+)", script))
    assert directories, "the layer pins no uv directories"
    assert all(path.startswith(f"{harbor_load.HUD_ROOT}/") for path in directories.values()), (
        directories
    )
    # Nothing is written relative to the image's home, so there is no scattered
    # state to clean up afterwards.
    assert "$HOME" not in script
    # The CLI's update check writes into that home too, and calls PyPI on every
    # rollout since a fresh container never has a warm cache.
    dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV HUD_SKIP_VERSION_CHECK=1" in dockerfile


async def test_the_agent_toolchain_installs_only_what_the_image_lacks(tmp_path) -> None:
    # Harbor's agent install provisions python3/pip/git/curl into the task
    # container before the agent phase; an adapted image bakes the same set at
    # build time instead. An image shipping its own interpreter must keep it,
    # so every tool is presence-checked rather than installed outright.
    _write_harbor_task(tmp_path, "task-a")
    await harbor.adapt(tmp_path, build=False)
    (context,) = sorted((tmp_path / ".hud-adapt").iterdir())
    script = (context / "_hud" / "install.sh").read_text(encoding="utf-8")
    decision = script[script.index('apt_pkgs=""') : script.index("\ndone") + len("\ndone")]

    def queued(path: str) -> set[str]:
        # Runs under ``set -eu`` as the image does: a tool already present must
        # not abort the script mid-build.
        result = subprocess.run(
            ["/bin/sh", "-c", f'set -eu\nPATH={path}\n{decision}\nprintf "%s" "$apt_pkgs"'],
            capture_output=True,
            text=True,
            check=True,
        )
        return set(result.stdout.split())

    assert queued("/nonexistent") == {
        "python3",
        "python3-venv",
        "python3-pip",
        "git",
        "curl",
        "ca-certificates",
    }

    stub = tmp_path / "bin"
    stub.mkdir()
    for tool in ("python3", "pip3"):
        (stub / tool).write_text("#!/bin/sh\n", encoding="utf-8")
        (stub / tool).chmod(0o755)

    assert queued(str(stub)) == {"git", "curl", "ca-certificates"}


def test_adapt_images_stamp_rows_when_the_caller_passes_them(tmp_path) -> None:
    # The mapping is a value the caller holds — nothing is written into the
    # dataset, so there is no cache to go stale.
    _write_harbor_task(tmp_path, "task-a")
    ((env_name, _),) = harbor.grouped(tmp_path)
    image = f"registry.io/acme/{env_name}:abc123"

    (task,) = list(harbor.load(tmp_path, images={env_name: image}))

    assert task.runtime_config is not None
    assert task.runtime_config.image == image
    # ...and without the mapping the rows carry only what the task declared.
    (bare,) = list(harbor.load(tmp_path))
    assert bare.runtime_config is None or bare.runtime_config.image is None


async def test_environment_serves_the_baked_tasks(tmp_path, monkeypatch) -> None:
    # The constructor refuses to build an unsandboxed env (hiding the
    # harness's tree is an integrity property); tests run outside a
    # container, so stub bwrap.
    monkeypatch.setattr("hud.environment.workspace.usable_bwrap", lambda: "/usr/bin/true")
    _write_harbor_task(tmp_path, "task-a")
    _write_harbor_task(tmp_path, "task-b")
    await harbor.adapt(tmp_path, build=False)
    (context,) = sorted((tmp_path / ".hud-adapt").iterdir())

    env = harbor.environment(context / "_hud" / "tasks", name=context.name)

    assert env.name == context.name
    assert sorted(env.tasks) == ["task-a", "task-b"]
    # The adapted CMD serves exactly this constructor.
    dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")
    assert "harbor:environment" in dockerfile


def test_harbor_implements_the_integration_contract() -> None:
    from hud.environment import Integration

    assert isinstance(harbor.integration, Integration)
    assert harbor.integration.name == "harbor"


def test_load_translates_declared_requirements(tmp_path) -> None:
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n'
        "[agent]\ntimeout_sec = 2400.0\n\n"
        "[environment]\ncpus = 4\nmemory_mb = 8192\ngpus = 2\n"
        "build_timeout_sec = 600.0\nstorage_mb = 10240\n",
        encoding="utf-8",
    )

    (row,) = list(harbor.load(tmp_path))

    assert row.runtime_config is not None
    assert row.runtime_config.resources is not None
    assert row.runtime_config.resources.cpu == 4.0
    assert row.runtime_config.resources.memory_mb == 8192
    assert row.runtime_config.resources.gpu is not None
    assert row.runtime_config.resources.gpu.count == 2
    # Time budgets are the engine's, not the substrate's.
    assert row.runtime_config.limits is None
    assert harbor_load.agent_timeout(task) == 2400.0


def test_load_omits_requirements_a_task_does_not_declare(tmp_path) -> None:
    _write_harbor_task(tmp_path, "task-a")  # minimal task.toml: no resources

    (row,) = list(harbor.load(tmp_path))

    assert row.runtime_config is None


def test_load_carries_metadata_as_columns(tmp_path) -> None:
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n'
        'description = "Fix the thing properly."\nkeywords = ["shell", "debug"]\n\n'
        '[metadata]\ndifficulty = "hard"\ncategory = "systems"\ntags = ["a", "b"]\n',
        encoding="utf-8",
    )

    (row,) = list(harbor.load(tmp_path))

    assert row.columns is not None
    assert row.columns["difficulty"] == "hard"
    assert row.columns["category"] == "systems"
    assert row.columns["keywords"] == ["shell", "debug"]


async def test_served_templates_use_the_declared_description(tmp_path, monkeypatch) -> None:
    # The constructor refuses to build an unsandboxed env (hiding the
    # harness's tree is an integrity property); tests run outside a
    # container, so stub bwrap.
    monkeypatch.setattr("hud.environment.workspace.usable_bwrap", lambda: "/usr/bin/true")
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n'
        'description = "Fix the thing properly."\n',
        encoding="utf-8",
    )
    await harbor.adapt(tmp_path, build=False)
    (context,) = sorted((tmp_path / ".hud-adapt").iterdir())

    env = harbor.environment(context / "_hud" / "tasks", name=context.name)

    assert env.tasks["task-a"].description == "Fix the thing properly."


async def test_multi_step_tasks_load_but_cannot_be_adapted_yet(tmp_path) -> None:
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n[[steps]]\nname = "first"\n',
        encoding="utf-8",
    )
    (task / "instruction.md").unlink()  # multi-step: instructions live per step

    assert [row.id for row in harbor.load(tmp_path)] == ["task-a"]
    with pytest.raises(NotImplementedError, match="multi-step"):
        await harbor.adapt(tmp_path, build=False)


def test_declared_workspace_policy_is_translated(tmp_path) -> None:
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n'
        '[environment]\nworkdir = "/srv/app"\n\n'
        '[environment.env]\nTOKEN = "abc"\n',
        encoding="utf-8",
    )

    policy = harbor_load.workspace_policy(task)

    assert policy == {
        "network": True,
        "env": {"TOKEN": "abc"},
        "agent_env": {},
        "workdir": "/srv/app",
        # Public, but reached through the workspace's own way out rather than
        # by sharing the substrate's network.
        "allowed_hosts": ["*"],
        # And the verifier's are its own, not a copy of the agent's.
        "verifier_allowed_hosts": ["*"],
        "user": None,
        # Per phase, because Harbor's are: a task may restrict the agent and
        # still verify as root.
        "agent_user": None,
        "verifier_user": None,
    }


def test_each_phase_is_held_to_the_hosts_it_declared(tmp_path) -> None:
    """An allowlist written for one phase must not become the other's. The
    agent's applied to the verifier is a grader that cannot install its own
    tooling — a zero the task's author never wrote."""
    isolated = _write_harbor_task(tmp_path, "isolated")
    (isolated / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/isolated"\n\n'
        '[environment]\nnetwork_mode = "no-network"\n',
        encoding="utf-8",
    )
    assert harbor_load.unsupported_features(isolated) == []
    assert harbor_load.workspace_policy(isolated)["network"] is False
    assert harbor_load.workspace_policy(isolated)["allowed_hosts"] == []
    assert harbor_load.workspace_policy(isolated)["verifier_allowed_hosts"] == []

    # Declared for the whole environment: both phases named those hosts.
    shared = _write_harbor_task(tmp_path, "shared")
    (shared / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/shared"\n\n'
        '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
        encoding="utf-8",
    )
    assert harbor_load.unsupported_features(shared) == []
    assert harbor_load.workspace_policy(shared)["allowed_hosts"] == ["pypi.org"]
    assert harbor_load.workspace_policy(shared)["verifier_allowed_hosts"] == ["pypi.org"]

    # Declared for the agent alone: the verifier said nothing, so it is public.
    agent_only = _write_harbor_task(tmp_path, "agent-only")
    (agent_only / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/agent-only"\n\n'
        '[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
        encoding="utf-8",
    )
    assert harbor_load.unsupported_features(agent_only) == []
    assert harbor_load.workspace_policy(agent_only)["allowed_hosts"] == ["pypi.org"]
    assert harbor_load.workspace_policy(agent_only)["verifier_allowed_hosts"] == ["*"]


def test_tasks_with_different_policies_get_separate_envs(tmp_path) -> None:
    # Same build context, different declared workdir: one env serves one
    # policy, so these must not share an environment.
    for name, workdir in (("here", "/app"), ("there", "/srv")):
        task = _write_harbor_task(tmp_path, name)
        (task / "task.toml").write_text(
            f'schema_version = "1.3"\n\n[task]\nname = "demo/{name}"\n\n'
            f'[environment]\nworkdir = "{workdir}"\n',
            encoding="utf-8",
        )

    envs = {row.env for row in harbor.load(tmp_path)}

    assert len(envs) == 2


def test_adapted_cmd_serves_the_contract_constructor(tmp_path) -> None:
    import asyncio

    _write_harbor_task(tmp_path, "task-a")
    asyncio.get_event_loop_policy()
    asyncio.run(harbor.adapt(tmp_path, build=False))
    (context,) = sorted((tmp_path / ".hud-adapt").iterdir())
    dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")

    assert "harbor:environment" in dockerfile


async def test_planted_reward_files_are_discarded_before_grading(tmp_path) -> None:
    # /logs is agent-reachable: a reward.json planted before grading must not
    # out-rank the verifier's own output.
    import asyncio
    import json

    from integrations.harbor._adapt import _grade_with_verifier

    task = _write_harbor_task(tmp_path, "task-a")
    logs = tmp_path / "logs"
    (logs / "verifier").mkdir(parents=True)
    (logs / "verifier" / "reward.json").write_text(json.dumps({"reward": 1.0}), encoding="utf-8")

    async def run_tests():
        from hud.utils.process import create_process_group_exec

        return await create_process_group_exec(
            "bash",
            "-c",
            f"echo 0 > {logs}/verifier/reward.txt",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    grade = await _grade_with_verifier(harbor_load.TaskConfig.read(task), logs, None, run_tests)

    assert grade["score"] == 0.0


async def test_a_declared_identity_governs_the_phases_not_the_harness(tmp_path) -> None:
    # Harbor runs the agent and the verifier as the identity the task declares
    # for each; its harness is not subject to either. Serving from inside the
    # image, a USER directive would demote the harness too, leaving it unable
    # to create /tests at the filesystem root or its own state under the
    # harness tree.
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN useradd -m agent\nUSER agent\n", encoding="utf-8"
    )
    await harbor.adapt(tmp_path, build=False)
    (context,) = sorted((tmp_path / ".hud-adapt").iterdir())
    layer = (context / "Dockerfile").read_text(encoding="utf-8").split("adaptation layer", 1)[1]

    assert "USER root" in layer  # installs, and goes on serving, as root
    assert "USER agent" not in layer
    assert "mkdir -p /tests" not in layer  # root makes them when grading needs them
    # The image's own declaration is not lost: it is kept for serve time,
    # where the build context no longer exists.
    assert (context / "_hud" / "image-user").read_text(encoding="utf-8").strip() == "agent"


def test_each_phase_takes_the_identity_declared_for_it(tmp_path, monkeypatch) -> None:
    # A task may hand the agent a restricted account and still verify as root.
    from integrations.harbor import _adapt

    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n[agent]\nuser = "agent"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(_adapt, "_image_user", lambda _task: None)
    monkeypatch.setattr(_adapt.pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=1000))

    assert _adapt.phase_uid(task, "agent") == 1000
    assert _adapt.phase_uid(task, "verifier") is None  # root, as declared by omission


def test_build_stage_entrypoint_does_not_refuse(tmp_path) -> None:
    # Adaptation replaces only final-stage startup: an ENTRYPOINT confined to
    # a build stage is no reason to refuse the task.
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "environment" / "Dockerfile").write_text(
        'FROM golang:1.22 AS build\nENTRYPOINT ["/tool"]\nRUN true\n'
        "FROM python:3.12-slim\nWORKDIR /app\n",
        encoding="utf-8",
    )

    assert harbor_load.unsupported_features(task) == []


def test_declared_uid_zero_beats_the_source_user(tmp_path) -> None:
    # uid 0 is a declaration, not an absence: it must not fall through to the
    # Dockerfile's own USER.
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n'
        "[agent]\nuser = 0\n\n[verifier]\nuser = 0\n",
        encoding="utf-8",
    )

    assert harbor_load.workspace_policy(task)["user"] == 0


def test_rewards_are_finite_numbers_not_booleans(tmp_path) -> None:
    import json as jsonlib

    from integrations.harbor._adapt import _read_reward

    logs = tmp_path / "verifier"
    logs.mkdir()
    for planted in (
        "true",
        jsonlib.dumps({"reward": True}),
        jsonlib.dumps({"reward": float("inf")}),
    ):
        (logs / "reward.json").write_text(planted, encoding="utf-8")
        score, _info = _read_reward(logs)
        assert score is None, planted
    (logs / "reward.json").unlink()
    (logs / "reward.txt").write_text("nan", encoding="utf-8")
    score, _info = _read_reward(logs)
    assert score is None
    (logs / "reward.txt").write_text("0.75", encoding="utf-8")
    assert _read_reward(logs)[0] == 0.75


def test_first_named_gpu_type_is_requested(tmp_path) -> None:
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n'
        '[environment]\ngpus = 1\ngpu_types = ["", "H100"]\n',
        encoding="utf-8",
    )

    (row,) = list(harbor.load(tmp_path))

    assert row.runtime_config is not None
    assert row.runtime_config.resources is not None
    assert row.runtime_config.resources.gpu is not None
    assert row.runtime_config.resources.gpu.type == "H100"


def test_an_invalid_task_toml_is_an_error_not_a_default(tmp_path) -> None:
    # Silently falling back to defaults would grade the task under
    # requirements its author never wrote.
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n[environment]\ncpus = "lots"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a valid Harbor task"):
        harbor.load(tmp_path)


def test_grading_directories_do_not_exist_during_the_agent_phase(tmp_path, monkeypatch) -> None:
    """The image ships /tests and the verdict dir empty, and a previous rollout
    leaves them behind — either way the agent phase must not find them."""
    from integrations.harbor import _adapt

    tests, verdict = tmp_path / "tests", tmp_path / "logs" / "verifier"
    for stale in (tests, verdict):
        stale.mkdir(parents=True)
    (tests / "test.sh").write_text("the assertions", encoding="utf-8")
    monkeypatch.setattr(_adapt, "TESTS", tests)
    monkeypatch.setattr(_adapt, "VERIFIER_LOGS", verdict)

    _adapt._hide_grading_dirs()

    assert not tests.exists()
    assert not verdict.exists()
    # /logs itself is Harbor's and stays: the agent's own answer is written there.
    assert verdict.parent.exists()


async def test_masks_are_applied_after_the_workspace_bind(tmp_path, monkeypatch) -> None:
    # bwrap applies ``mounts`` after the workspace bind; as system mounts the
    # masks would be re-covered when the guest path is "/" (no WORKDIR).
    from hud.environment import workspace as workspace_mod

    monkeypatch.setattr(workspace_mod, "usable_bwrap", lambda: "/usr/bin/true")
    built: list[workspace_mod.Workspace] = []
    original = workspace_mod.Workspace

    def record(*args, **kwargs):
        ws = original(*args, **kwargs)
        built.append(ws)
        return ws

    monkeypatch.setattr("hud.environment.env.Workspace", record)
    _write_harbor_task(tmp_path, "task-a")
    await harbor.adapt(tmp_path, build=False)
    (context,) = sorted((tmp_path / ".hud-adapt").iterdir())

    harbor.environment(context / "_hud" / "tasks", name=context.name)

    (workspace,) = built
    masked = [m.dst for m in workspace.mounts]
    # The harness's tree is hidden by rebuilding its parent, so the entry is
    # absent rather than present-and-empty: a directory named after the
    # harness is the tell, not what is inside it.
    assert str(harbor_load.HUD_ROOT.parent) in masked
    assert str(harbor_load.HUD_ROOT) not in masked
    # The assertions and the verdict are kept from the agent phase by not
    # existing during it, not by masking: an empty directory the base image
    # does not have is itself a signal.
    assert "/tests" not in masked
    assert "/logs/verifier" not in masked
    assert [m.dst for m in workspace._system_mounts] == ["/", "/proc", "/dev"]


async def test_dataset_symlinks_are_never_dereferenced(tmp_path) -> None:
    # A dataset is untrusted: a link out of it must stay a link, not pull
    # host files into the build context or the served /tests.
    secret = tmp_path / "outside" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("host-only", encoding="utf-8")
    task = _write_harbor_task(tmp_path / "ds", "task-a")
    (task / "tests" / "leak.txt").symlink_to(secret)

    await harbor.adapt(tmp_path / "ds", build=False)
    (context,) = sorted((tmp_path / "ds" / ".hud-adapt").iterdir())

    # The link is copied as a link: its target was never read, so no host
    # content entered the context (inside the image the link simply dangles).
    baked = context / "_hud" / "tasks" / "task-a" / "tests" / "leak.txt"
    assert baked.is_symlink()
    assert os.readlink(baked) == str(secret)


def test_one_network_decision_serves_both_phases(tmp_path) -> None:
    # Container-wide isolation binds agent and verifier alike; a phase-level
    # declaration binds that phase. One function answers for both.
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n'
        '[environment]\nnetwork_mode = "no-network"\n',
        encoding="utf-8",
    )

    assert harbor_load.TaskConfig.read(task).network("agent") is False
    assert harbor_load.TaskConfig.read(task).network("verifier") is False

    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n'
        '[verifier]\nnetwork_mode = "no-network"\n',
        encoding="utf-8",
    )

    assert harbor_load.TaskConfig.read(task).network("agent") is True
    assert harbor_load.TaskConfig.read(task).network("verifier") is False


def test_final_stage_reads_only_what_the_shipped_image_declares() -> None:
    """The one Dockerfile parser, across the shapes that misled it before.

    Each ``FROM`` opens a stage, so a build stage's ``USER``/``ENTRYPOINT``
    is not the shipped image's; ``user[:group]`` is a legal operand; and a
    heredoc body is data, not instructions.
    """
    from integrations.harbor._load import final_stage

    multistage = final_stage(
        'FROM golang:1.22 AS build\nUSER builder\nENTRYPOINT ["/tool"]\nRUN true\n'
        "FROM python:3.12-slim\nWORKDIR /app\nUSER app:app\n"
    )
    assert multistage.user == "app:app"  # group form preserved
    assert "ENTRYPOINT" not in multistage.directives  # build stage's, not shipped

    # A build stage's USER alone leaves the shipped stage's identity unset.
    assert final_stage("FROM golang AS build\nUSER builder\nFROM python:3.12-slim\n").user is None

    # Root in any spelling is "no declared identity to restore".
    assert final_stage("FROM x\nUSER root\n").user is None
    assert final_stage("FROM x\nUSER 0:0\n").user is None

    # A heredoc writing another Dockerfile is not this one's instructions.
    heredoc = final_stage(
        "FROM python:3.12-slim\n"
        "RUN <<EOF cat > /tmp/generated.Dockerfile\n"
        'FROM someone-else:latest\nUSER hacker\nENTRYPOINT ["/elsewhere"]\n'
        "EOF\nUSER app\n"
    )
    assert heredoc.user == "app"
    assert "ENTRYPOINT" not in heredoc.directives


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ('[environment.healthcheck]\ncommand = "curl -sf localhost/health"\n', "healthcheck"),
        (
            '[[environment.mcp_servers]]\nname = "db"\nurl = "http://localhost:9000/sse"\n',
            "mcp_servers",
        ),
        ('[verifier]\nenvironment_mode = "separate"\n', "own environment"),
        ('[environment]\ntpu = {type = "v5", topology = "2x2"}\n', "tpu"),
        ('[environment]\nos = "windows"\n', "os"),
        ('[agent]\nuser = "agent"\n\n[verifier]\nuser = "root"\n', "one USER"),
        ('[[steps]]\nname = "first"\n', "multi-step"),
    ],
)
def test_declarations_this_integration_cannot_reproduce_are_refused(
    tmp_path, declaration: str, expected: str
) -> None:
    """A wrong score is worse than a refused task, so each of these names
    itself in the refusal rather than being silently dropped."""
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        f'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n{declaration}',
        encoding="utf-8",
    )

    assert expected in " ".join(harbor_load.unsupported_features(task))


async def test_a_refused_task_never_reaches_a_build_context(tmp_path) -> None:
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n'
        '[environment.healthcheck]\ncommand = "curl -sf localhost/health"\n',
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="healthcheck"):
        await harbor.adapt(tmp_path, build=False)


async def test_a_chatty_verifier_does_not_deadlock(tmp_path) -> None:
    # More output than a pipe buffer holds (~64KB): draining only after exit
    # would block the writer forever and score a finished script as a timeout.
    import asyncio

    from hud.utils.process import create_process_group_exec
    from integrations.harbor._adapt import _grade_with_verifier

    task = _write_harbor_task(tmp_path, "task-a")
    logs = tmp_path / "logs"
    (logs / "verifier").mkdir(parents=True)

    async def run_tests():
        return await create_process_group_exec(
            "bash",
            "-c",
            f"head -c 400000 /dev/zero | tr '\\0' 'x'; echo 1 > {logs}/verifier/reward.txt",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    grade = await asyncio.wait_for(
        _grade_with_verifier(harbor_load.TaskConfig.read(task), logs, None, run_tests),
        timeout=30,
    )

    assert grade["score"] == 1.0
    assert len(grade["info"]["stdout"]) > 0


def test_phase_env_reaches_the_phase_that_declared_it(tmp_path) -> None:
    # [environment.env] is container-wide, [agent.env] is the agent's, and
    # [verifier.env] is applied where the verifier runs — none silently lost.
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n'
        '[environment.env]\nSHARED = "both"\n\n'
        '[agent.env]\nAGENT_ONLY = "yes"\n\n'
        '[verifier.env]\nVERIFIER_ONLY = "yes"\n',
        encoding="utf-8",
    )

    policy = harbor_load.workspace_policy(task)
    config = harbor_load.TaskConfig.read(task)

    # Container-wide reaches every process; the agent's reaches its sessions
    # only; the verifier's is applied where the verifier runs.
    assert policy["env"] == {"SHARED": "both"}
    assert policy["agent_env"] == {"AGENT_ONLY": "yes"}
    assert config.verifier.env == {"VERIFIER_ONLY": "yes"}


def test_only_a_real_user_conflict_is_refused(tmp_path) -> None:
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n[agent]\nuser = "app"\n',
        encoding="utf-8",
    )

    # One phase naming an identity is fine: the image's single USER is it.
    assert harbor_load.unsupported_features(task) == []
    assert harbor_load.TaskConfig.read(task).user == "app"


def test_an_explicit_zero_timeout_is_not_silently_extended(tmp_path) -> None:
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/task-a"\n\n[verifier]\ntimeout_sec = 0\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a valid Harbor task"):
        harbor.load(tmp_path)


async def test_a_cancelled_grade_leaves_nothing_running(tmp_path) -> None:
    # However grading exits — including a cancelled rollout — the verifier's
    # process group and the pipe readers are released.
    import asyncio

    from hud.utils.process import create_process_group_exec
    from integrations.harbor._adapt import _grade_with_verifier

    task = _write_harbor_task(tmp_path, "task-a")
    logs = tmp_path / "logs"
    (logs / "verifier").mkdir(parents=True)
    started: list[int] = []

    async def run_tests():
        group = await create_process_group_exec(
            "bash",
            "-c",
            "sleep 30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        started.append(group.process.pid)
        return group

    grading = asyncio.create_task(
        _grade_with_verifier(harbor_load.TaskConfig.read(task), logs, None, run_tests)
    )
    await asyncio.sleep(0.2)
    grading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await grading

    # The verifier process is gone rather than orphaned for 30 seconds.
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(started[0], 0)


def test_image_tags_do_not_depend_on_host_state(tmp_path) -> None:
    # Links are copied as links, so hashing must read the link — not what it
    # points at — or the same dataset tags differently on another machine.
    from integrations.harbor._load import hash_directory

    target = tmp_path / "outside.txt"
    target.write_text("first", encoding="utf-8")
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "link").symlink_to(target)

    before = hash_directory(context)
    target.write_text("second, changed on this host only", encoding="utf-8")

    assert hash_directory(context) == before


def test_a_workdir_inside_the_reserved_path_is_refused(tmp_path) -> None:
    # That tree belongs to the adaptation layer and is hidden from sessions;
    # a task working there would find it empty.
    task = _write_harbor_task(tmp_path, "task-a")
    (task / "environment" / "Dockerfile").write_text(
        f"FROM python:3.12-slim\nWORKDIR {harbor_load.HUD_ROOT}/app\n", encoding="utf-8"
    )

    assert "reserved" in " ".join(harbor_load.unsupported_features(task))
