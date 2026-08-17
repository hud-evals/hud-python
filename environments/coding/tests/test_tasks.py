"""Task-row tests for the bundled coding task."""

import shlex
import subprocess
from pathlib import Path

import tasks
from coding.grading import parse_junit, score_tests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_rows_parameterize_the_coding_task_template():
    [task] = tasks.tasks
    assert task.slug == "flask-4992"
    assert task.env == tasks.env.name
    assert task.id == "coding-task"
    assert task.args["base_ref"] == "origin/flask_4992_baseline"
    assert task.args["test_ref"] == "origin/flask_4992_test"
    assert task.args["golden_ref"] == "origin/flask_4992_golden"
    assert task.args["test_files"]
    assert "{junit_path}" in task.args["test_script"]
    assert task.args["f2p_test_nodeids"]
    assert task.args["p2p_test_nodeids"]
    assert task.args["use_binary_score"] is True
    assert task.args["description"].startswith("Add a file mode parameter to flask.Config.from_file()")
    assert 'mode="b"' in task.args["description"]


def test_bundled_task_baseline_fails_and_golden_ref_passes(tmp_path: Path):
    [task] = tasks.tasks

    for ref, expected_exit_code, expected_reward in (
        (task.args["base_ref"], 1, 0.0),
        (task.args["golden_ref"], 0, 1.0),
    ):
        repo = tmp_path / ref.rsplit("/", 1)[-1]
        subprocess.run(
            ["git", "clone", "-q", str(PROJECT_ROOT / "flask-4992.bundle"), str(repo)],
            check=True,
        )
        subprocess.run(["git", "checkout", "-qf", ref], cwd=repo, check=True)
        subprocess.run(
            ["git", "checkout", task.args["test_ref"], "--", *task.args["test_files"]],
            cwd=repo,
            check=True,
        )

        junit_path = tmp_path / f"{repo.name}.xml"
        command = (
            task.args["test_script"]
            .replace("{test_files}", shlex.join(task.args["test_files"]))
            .replace("{junit_path}", shlex.quote(str(junit_path)))
        )
        completed = subprocess.run(["bash", "-lc", command], cwd=repo, check=False)
        assert completed.returncode == expected_exit_code

        result = score_tests(
            parse_junit(junit_path),
            task.args["f2p_test_nodeids"],
            task.args["p2p_test_nodeids"],
            task.args["use_binary_score"],
        )
        assert result.reward == expected_reward
