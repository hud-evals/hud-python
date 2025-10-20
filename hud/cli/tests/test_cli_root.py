from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

import hud.cli as cli

if TYPE_CHECKING:
    from pathlib import Path


@patch("hud.cli.utils.metadata.analyze_from_metadata", new_callable=AsyncMock)
@patch("asyncio.run")
def test_analyze_params_metadata(mock_run, mock_analyze):
    # image only -> metadata path
    cli.analyze(params=["img:latest"], output_format="json", verbose=False)
    assert mock_run.called


@patch("hud.cli.analyze.analyze_environment", new_callable=AsyncMock)
@patch("hud.cli.utils.docker.build_run_command")
@patch("asyncio.run")
def test_analyze_params_live(mock_run, mock_build_cmd, mock_analyze_env):
    mock_build_cmd.return_value = ["docker", "run", "img", "-e", "K=V"]
    # docker args trigger live path
    cli.analyze(params=["img:latest", "-e", "K=V"], output_format="json", verbose=True)
    assert mock_run.called


def test_analyze_no_params_errors():
    import typer

    # When no params provided, analyze prints help and exits(1)
    with pytest.raises(typer.Exit):
        cli.analyze(params=None, config=None, cursor=None, output_format="json", verbose=False)  # type: ignore


@patch("hud.cli.analyze.analyze_environment_from_config", new_callable=AsyncMock)
@patch("asyncio.run")
def test_analyze_from_config(mock_run, mock_func, tmp_path: Path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    cli.analyze(params=None, config=cfg, cursor=None, output_format="json", verbose=False)  # type: ignore
    assert mock_run.called


@patch("hud.cli.parse_cursor_config")
@patch("hud.cli.analyze.analyze_environment_from_mcp_config", new_callable=AsyncMock)
@patch("asyncio.run")
def test_analyze_from_cursor(mock_run, mock_analyze, mock_parse):
    mock_parse.return_value = (["cmd", "arg"], None)
    cli.analyze(params=None, config=None, cursor="server", output_format="json", verbose=False)  # type: ignore
    assert mock_run.called


@patch("hud.cli.build_command")
def test_build_env_var_parsing(mock_build):
    cli.build(
        params=[".", "-e", "A=B", "--env=C=D", "--env", "E=F"],
        tag=None,
        no_cache=False,
        verbose=False,
        platform=None,
    )
    assert mock_build.called
    # args: directory, tag, no_cache, verbose, env_vars, platform
    env_vars = mock_build.call_args[0][4]
    assert env_vars == {"A": "B", "C": "D", "E": "F"}


@patch("hud.cli.utils.runner.run_mcp_server")
def test_run_local_calls_runner(mock_runner):
    cli.run(
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
    cli.run(
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
        cli.run(params=None)  # type: ignore


@patch("hud.cli.run_mcp_dev_server")
def test_dev_calls_runner(mock_dev):
    cli.dev(
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


@patch("hud.cli.pull_command")
def test_pull_command_wrapper(mock_pull):
    cli.pull(target="org/name:tag", lock_file=None, yes=True, verify_only=True, verbose=False)
    assert mock_pull.called


@patch("hud.cli.push_command")
def test_push_command_wrapper(mock_push, tmp_path: Path):
    cli.push(directory=str(tmp_path), image=None, tag=None, sign=False, yes=True, verbose=True)
    assert mock_push.called
