from __future__ import annotations

from unittest.mock import patch

import hud.cli as cli


def test_version_does_not_crash():
    # Just ensure it runs without raising
    cli.version()


@patch("hud.cli.list_module.list_command")
def test_list_environments_wrapper(mock_list):
    cli.list_environments(filter_name=None, json_output=False, show_all=False, verbose=False)
    assert mock_list.called


@patch("hud.cli.clone_repository", return_value=(True, "/tmp/repo"))
@patch("hud.cli.get_clone_message", return_value={})
@patch("hud.cli.print_tutorial")
def test_clone_wrapper(mock_tutorial, _msg, _clone):
    cli.clone("https://example.com/repo.git")
    assert mock_tutorial.called


@patch("hud.cli.remove_command")
def test_remove_wrapper(mock_remove):
    cli.remove(target="all", yes=True, verbose=False)
    assert mock_remove.called
