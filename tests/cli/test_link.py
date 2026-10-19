"""The directory link (``.hud/config.json``): the platform ids it accumulates, the older
form it rewrites, and the Project a directory places work in."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.support import write_files
from tests.support.platform import BROWSER_PROJECT_ID, TEAM_ID, USER_ID

if TYPE_CHECKING:
    from tests.support import Hud, HudEnv, Platform

OTHER_ID = "cccccccc-0000-4000-8000-000000000001"


def link(hud: Hud) -> dict[str, Any] | None:
    path = hud.cwd / ".hud" / "config.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_linking_a_directory_accumulates_its_platform_ids(hud: Hud, platform: Platform) -> None:
    example = platform.add_environment("example")

    used = hud("project", "use", "Browser Evals", "--json")
    linked = hud("sync", "env", "example", "--json")
    written = (hud.cwd / ".hud" / "config.json").read_bytes()
    relinked = hud("sync", "env", example, "--json")
    shown = hud("project", "--json")

    assert [used.exit_code, linked.exit_code, relinked.exit_code, shown.exit_code] == [0] * 4
    stored = link(hud)
    assert stored is not None
    assert (stored["registry_id"], stored["project_id"], stored["taskset_id"]) == (
        example,
        BROWSER_PROJECT_ID,
        None,
    )
    assert (stored["scope"]["user_id"], stored["scope"]["team_id"]) == (USER_ID, TEAM_ID)
    assert (linked.json["changed"], relinked.json["changed"]) == (True, False)
    assert (hud.cwd / ".hud" / "config.json").read_bytes() == written
    assert (shown.json["project"]["id"], shown.json["source"]) == (
        BROWSER_PROJECT_ID,
        ".hud/config.json",
    )
    assert "sk-hud-test" not in written.decode()


@pytest.mark.parametrize("dry_run", [False, True], ids=["rewritten", "left-alone-on-a-dry-run"])
def test_an_older_camelcase_link_is_rewritten_with_its_ids(
    hud: Hud, platform: Platform, dry_run: bool
) -> None:
    example = platform.add_environment("example")
    older = json.dumps({"registryId": example, "tasksetId": OTHER_ID})
    write_files(hud.cwd / ".hud", {"config.json": older})

    result = hud("project", "use", "browser-evals", *(["--dry-run"] if dry_run else []), "--json")

    assert result.exit_code == 0, result
    stored = link(hud)
    assert stored is not None
    if dry_run:
        assert json.dumps(stored) == older
    else:
        assert (stored["registry_id"], stored["taskset_id"], stored["project_id"]) == (
            example,
            OTHER_ID,
            BROWSER_PROJECT_ID,
        )


# ─── which Project a directory uses ─────────────────────────────────────


@pytest.mark.parametrize(
    ("linked", "default", "project", "source"),
    [
        pytest.param(None, None, None, "team default", id="team-default"),
        pytest.param(
            None, "Browser Evals", BROWSER_PROJECT_ID, "HUD_DEFAULT_PROJECT", id="default-by-name"
        ),
        pytest.param(
            None,
            BROWSER_PROJECT_ID.upper(),
            BROWSER_PROJECT_ID,
            "HUD_DEFAULT_PROJECT",
            id="default-by-id",
        ),
        pytest.param(
            BROWSER_PROJECT_ID,
            "locked-down",
            BROWSER_PROJECT_ID,
            ".hud/config.json",
            id="the-link-beats-the-default",
        ),
    ],
)
def test_project_shows_the_selected_project_and_why(
    hud: Hud,
    platform: Platform,
    hud_env: HudEnv,
    linked: str | None,
    default: str | None,
    project: str | None,
    source: str,
) -> None:
    hud_env.set(HUD_DEFAULT_PROJECT=default)
    if linked is not None:
        assert hud("project", "use", linked).exit_code == 0

    result = hud("project", "--json")

    assert result.exit_code == 0, result
    assert ((result.json["project"] or {}).get("id"), result.json["source"]) == (project, source)


def test_the_directory_option_after_the_verb_wins(hud: Hud, platform: Platform) -> None:
    result = hud("project", "-C", "group", "use", BROWSER_PROJECT_ID, "-C", "override")

    assert result.exit_code == 0, result
    linked = [path.parent.parent.name for path in hud.cwd.glob("*/.hud/config.json")]
    assert linked == ["override"]
