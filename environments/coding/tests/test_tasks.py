"""Task-row tests for the bundled coding tasks."""

import tasks
from env import env


def test_rows_parameterize_the_same_coding_task_template():
    assert [task.slug for task in tasks.tasks] == ["flask-4992", "flask-5063"]

    for task in tasks.tasks:
        assert task.env == env.name
        assert task.id == "coding-task"
        assert task.args["base_ref"]
        assert task.args["test_patch"]
        assert task.args["test_path"] == "tests"
        assert "{junit_path}" in task.args["test_command"]
        assert task.args["fail_to_pass"]
        assert task.args["pass_to_pass"]
        assert task.args["binary"] is True
