"""Task-row tests for the bundled coding tasks."""

import shlex
import subprocess
from pathlib import Path

import tasks
from coding.grading import parse_junit, score_tests

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
        assert task.args["test_files"]
        assert "{junit_path}" in task.args["test_script"]
        assert task.args["f2p_test_nodeids"]
        assert task.args["p2p_test_nodeids"]
        assert task.args["use_binary_score"] is True
        assert "test_ref" not in task.args
        assert "golden_ref" not in task.args


def test_bundled_task_baselines_fail_and_reference_fixes_pass(tmp_path: Path):
    for task in tasks.tasks:
        for ref, expected_exit_code, expected_reward in (
            (task.args["base_ref"], 1, 0.0),
            (GOLDEN_REFS[task.slug], 0, 1.0),
        ):
            repo = tmp_path / f"{task.slug}-{ref.rsplit('/', 1)[-1]}"
            subprocess.run(
                ["git", "clone", "-q", str(PROJECT_ROOT / "flask.bundle"), str(repo)],
                check=True,
            )
            subprocess.run(["git", "checkout", "-qf", ref], cwd=repo, check=True)
            subprocess.run(
                ["git", "apply", "-"],
                cwd=repo,
                input=task.args["test_patch"],
                text=True,
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
