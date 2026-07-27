"""Task-row tests for both flavors: rows are well-formed and slugs readable."""

import json
from pathlib import Path

import swe_tasks
import tasks

FIXTURE_ROW = json.loads((Path(__file__).parent / "fixtures" / "instance" / "instance.json").read_text("utf-8"))


def test_generic_rows_parameterize_the_coding_task_template():
    slugs = {task.slug for task in tasks.tasks}
    assert slugs == {"sentry-fix", "notif-bug", "settings-v2", "webhook-bug", "sentry-fix-pr"}
    for task in tasks.tasks:
        assert task.env == "coding"
        assert task.args["base_ref"].endswith("_baseline")
        assert task.args["test_ref"].endswith("_test")
        assert task.args["test_files"]

    by_slug = {task.slug: task for task in tasks.tasks}
    assert by_slug["sentry-fix"].id == "coding-task"
    sdlc = by_slug["sentry-fix-pr"]
    assert sdlc.id == "sdlc-task"
    assert sdlc.args["issues"][0]["number"] == 42


def test_swe_slug_is_repo_tail_plus_commit_prefix():
    assert swe_tasks._slug(FIXTURE_ROW) == "widgets-00000000"
    assert (
        swe_tasks._slug(
            {
                "repo": "NodeBB/NodeBB",
                "instance_id": "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan",
            }
        )
        == "nodebb-04998908"
    )


def test_swe_rows_are_wellformed():
    """Whatever `swe_tasks.py` has fetched loads as valid, uniquely-slugged rows."""
    slugs = [task.slug for task in swe_tasks.tasks]
    assert len(set(slugs)) == len(slugs)
    for task in swe_tasks.tasks:
        assert task.env == "coding"
        assert task.id.startswith("instance_")
        assert task.columns and task.columns["repo"]
