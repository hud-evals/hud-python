from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from hud.cli.analyze import analyze_command
from hud.cli.build import build_command
from hud.cli.dev import dev_command
from hud.cli.pull import pull_command
from hud.cli.push import push_command
from hud.cli.run import run_command

if TYPE_CHECKING:
    from pathlib import Path


@patch("hud.cli.utils.metadata.analyze_from_metadata", new_callable=AsyncMock)
@patch("hud.cli.analyze.asyncio.run")
def test_analyze_params_metadata(mock_run, mock_analyze):
    analyze_command(params=["img:latest"], output_format="json", verbose=False)
    assert mock_run.called


@patch("hud.cli.analyze.analyze_environment", new_callable=AsyncMock)
@patch("hud.cli.utils.docker.build_run_command")
@patch("hud.cli.analyze.asyncio.run")
def test_analyze_params_live(mock_run, mock_build_cmd, mock_analyze_env):
    mock_build_cmd.return_value = ["docker", "run", "img", "-e", "K=V"]
    analyze_command(params=["img:latest", "-e", "K=V"], output_format="json", verbose=True)
    assert mock_run.called


def test_analyze_no_params_errors():
    import typer

    with pytest.raises(typer.Exit):
        analyze_command(params=None, config=None, output_format="json", verbose=False)  # type: ignore


@patch("hud.cli.analyze.analyze_environment_from_config", new_callable=AsyncMock)
@patch("hud.cli.analyze.asyncio.run")
def test_analyze_from_config(mock_run, mock_func, tmp_path: Path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    analyze_command(params=None, config=cfg, output_format="json", verbose=False)  # type: ignore
    assert mock_run.called


@patch("hud.cli.build.build_environment")
def test_build_env_var_parsing(mock_build_env):
    build_command(
        params=[".", "-e", "A=B", "--env=C=D", "--env", "E=F"],
        tag=None,
        no_cache=False,
        verbose=False,
        platform=None,
    )
    assert mock_build_env.called
    _, kwargs = mock_build_env.call_args
    # build_environment is called positionally
    args = mock_build_env.call_args[0]
    # args: directory, tag, no_cache, verbose, env_vars, platform, secrets, remote_cache, build_args
    env_vars = args[4]
    assert env_vars == {"A": "B", "C": "D", "E": "F"}


@patch("hud.cli.utils.runner.run_mcp_server")
def test_run_local_calls_runner(mock_runner):
    run_command(
        params=["img:latest"],
        local=True,
        transport="stdio",
        port=1234,
        url=None,  # type: ignore
        api_key=None,
        run_id=None,
        verbose=False,
    )
    assert mock_runner.called


@patch("hud.cli.utils.remote_runner.run_remote_server")
def test_run_remote_calls_remote(mock_remote):
    run_command(
        params=["img:latest"],
        local=False,
        transport="http",
        port=8765,
        url="https://x",
        api_key=None,
        run_id=None,
        verbose=True,
    )
    assert mock_remote.called


def test_run_no_params_errors():
    import typer

    with pytest.raises(typer.Exit):
        run_command(params=None)  # type: ignore


@patch("hud.cli.dev.run_mcp_dev_server")
def test_dev_calls_runner(mock_dev):
    dev_command(
        params=["server.main"],
        docker=False,
        stdio=False,
        port=9000,
        verbose=False,
        inspector=False,
        interactive=False,
        watch=None,  # type: ignore
    )
    assert mock_dev.called


@patch("hud.cli.pull.pull_environment")
def test_pull_command_wrapper(mock_pull):
    pull_command(target="org/name:tag", lock_file=None, yes=True, verify_only=True, verbose=False)
    assert mock_pull.called


@patch("hud.cli.push.push_environment")
def test_push_command_wrapper(mock_push, tmp_path: Path):
    push_command(directory=str(tmp_path), image=None, tag=None, sign=False, yes=True, verbose=True)
    assert mock_push.called
