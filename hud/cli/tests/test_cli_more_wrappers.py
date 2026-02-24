from __future__ import annotations

from unittest.mock import patch

from hud.cli import version
from hud.cli.list_func import list_command
from hud.cli.remove import remove_command


def test_version_does_not_crash():
    version()


@patch("hud.cli.list_func.list_environments")
def test_list_command_wrapper(mock_list):
    list_command(filter_name=None, json_output=False, show_all=False, verbose=False)
    assert mock_list.called


@patch("hud.cli.remove.remove_environment")
def test_remove_wrapper(mock_remove):
    remove_command(target="some-digest", yes=True, verbose=False)
    assert mock_remove.called
