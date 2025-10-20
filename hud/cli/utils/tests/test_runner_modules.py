from __future__ import annotations

from unittest import mock

import pytest

from hud.cli.utils.remote_runner import build_remote_headers, parse_env_vars, parse_headers
from hud.cli.utils.runner import run_mcp_server


def test_parse_headers_and_env_vars():
    assert parse_headers(["A:B", "C=D"]) == {"A": "B", "C": "D"}
    assert parse_env_vars(["API_KEY=xxx"]) == {"Env-Api-Key": "xxx"}


def test_build_remote_headers_combines():
    headers = build_remote_headers(
        image="img:latest", env_args=["X=1"], header_args=["H:V"], api_key="k", run_id="r"
    )
    assert headers["Mcp-Image"] == "img:latest"
    assert headers["Authorization"].startswith("Bearer ")
    assert headers["Run-Id"] == "r"
    assert headers["Env-X"] == "1"
    assert headers["H"] == "V"


@mock.patch("hud.cli.utils.runner.run_stdio_server")
def test_run_mcp_server_stdio(mock_stdio):
    run_mcp_server("img", [], "stdio", 8765, verbose=False, interactive=False)
    assert mock_stdio.called


def test_run_mcp_server_stdio_interactive_fails():
    with pytest.raises(SystemExit):
        run_mcp_server("img", [], "stdio", 8765, verbose=False, interactive=True)


@mock.patch("hud.cli.utils.runner.run_http_server")
def test_run_mcp_server_http(mock_http):
    run_mcp_server("img", [], "http", 8765, verbose=True, interactive=False)
    assert mock_http.called


@mock.patch("hud.cli.utils.runner.run_http_server_interactive")
def test_run_mcp_server_http_interactive(mock_http_int):
    run_mcp_server("img", [], "http", 8765, verbose=False, interactive=True)
    assert mock_http_int.called


def test_run_mcp_server_unknown():
    with pytest.raises(SystemExit):
        run_mcp_server("img", [], "bad", 8765, verbose=False)
