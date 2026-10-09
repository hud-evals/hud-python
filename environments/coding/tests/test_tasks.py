"""Task-row tests for the bundled coding tasks."""

from pathlib import Path

import tasks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_REFS = {
    "flask-4992": "origin/flask_4992_golden",
    "flask-5063": "origin/flask_5063_golden",
}


def test_rows_parameterize_the_same_coding_task_template():
    assert [task.slug for task in tasks.tasks] == ["flask-4992", "flask-5063"]

    for task in tasks.tasks:
        assert task.env == tasks.env.name
        assert task.id == "coding-task"
        assert task.args["base_ref"]
        assert task.args["test_patch"]
        assert task.args["test_path"] == "tests"
        assert "{junit_path}" in task.args["test_command"]
        assert task.args["fail_to_pass"]
        assert task.args["pass_to_pass"]
        assert task.args["binary"] is True
        assert "test_ref" not in task.args
        assert "golden_ref" not in task.args
