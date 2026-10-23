"""A Bash grader that scores selected tests from a JUnit report."""

from __future__ import annotations

import shlex
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree

from hud.environment import Workspace
from hud.graders import SubScore


@dataclass(frozen=True, slots=True)
class JUnitCase:
    nodeid: str
    passed: bool
    skipped: bool
    message: str | None = None


def parse_junit(path: Path) -> list[JUnitCase]:
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"invalid JUnit XML: {exc}") from exc

    cases: list[JUnitCase] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "testcase":
            continue
        classname = element.attrib.get("classname", "")
        name = element.attrib.get("name", "")
        nodeid = ".".join(part for part in (classname, name) if part).replace("::", ".")
        if not nodeid:
            raise ValueError("JUnit testcase is missing both classname and name")
        failure = next(
            (child for child in element if child.tag.rsplit("}", 1)[-1] in {"failure", "error"}),
            None,
        )
        skipped = any(child.tag.rsplit("}", 1)[-1] == "skipped" for child in element)
        message = None if failure is None else failure.attrib.get("message") or failure.text
        cases.append(
            JUnitCase(
                nodeid=nodeid,
                passed=failure is None and not skipped,
                skipped=skipped,
                message=message,
            )
        )
    return cases


def check_selection(fail_to_pass: list[str] | None, pass_to_pass: list[str] | None) -> None:
    """Reject a task's scored test selection before any test runs. These are authoring errors."""
    f2p = [nodeid.replace("::", ".") for nodeid in fail_to_pass or []]
    p2p = [nodeid.replace("::", ".") for nodeid in pass_to_pass or []]
    if len(set(f2p)) != len(f2p) or len(set(p2p)) != len(p2p):
        raise ValueError("test node IDs must be unique")
    if overlap := set(f2p) & set(p2p):
        raise ValueError(f"test node IDs cannot be in both groups: {sorted(overlap)}")
    if (fail_to_pass is not None or pass_to_pass is not None) and not f2p + p2p:
        raise ValueError("at least one scored test node ID is required")


def score_tests(
    cases: list[JUnitCase],
    fail_to_pass: list[str] | None,
    pass_to_pass: list[str] | None,
    binary: bool,
) -> SubScore:
    check_selection(fail_to_pass, pass_to_pass)
    f2p = [nodeid.replace("::", ".") for nodeid in fail_to_pass or []]
    p2p = [nodeid.replace("::", ".") for nodeid in pass_to_pass or []]

    selected = fail_to_pass is not None or pass_to_pass is not None
    expected = f2p + p2p if selected else [case.nodeid for case in cases if not case.skipped]

    reported = [case.nodeid for case in cases]
    if len(set(reported)) != len(reported):
        raise ValueError("JUnit test case IDs must be unique")
    passed = {case.nodeid for case in cases if case.passed}

    def passed_expected(nodeid: str) -> bool:
        if nodeid in reported:
            return nodeid in passed
        if nodeid.count("[") <= nodeid.count("]"):
            return False
        matches = [case_id for case_id in reported if case_id.startswith(nodeid)]
        return bool(matches) and all(case_id in passed for case_id in matches)

    passed_count = sum(passed_expected(nodeid) for nodeid in expected)
    value = passed_count / len(expected) if expected else 0.0
    if binary:
        value = float(value == 1.0)

    children = None
    info = {
        "passed": passed_count,
        "total": len(expected),
        "all_testcases": [asdict(case) for case in cases],
    }
    if selected:
        f2p_passed = sum(passed_expected(nodeid) for nodeid in f2p)
        p2p_passed = sum(passed_expected(nodeid) for nodeid in p2p)
        children = [
            SubScore(
                name="fail_to_pass",
                value=f2p_passed / len(f2p) if f2p else 1.0,
                weight=0.0,
            ),
            SubScore(
                name="pass_to_pass",
                value=p2p_passed / len(p2p) if p2p else 1.0,
                weight=0.0,
            ),
        ]
        info.update(
            {
                "f2p_passed": f2p_passed,
                "f2p_total": len(f2p),
                "p2p_passed": p2p_passed,
                "p2p_total": len(p2p),
            }
        )

    return SubScore(name="tests", value=value, children=children, info=info)


async def grade_tests(
    workspace: Workspace,
    command: str,
    *,
    timeout_seconds: float = 600,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    binary: bool = False,
) -> SubScore:
    if "{junit_path}" not in command:
        raise ValueError("test command must contain {junit_path}")
    # A bad selection is the task's fault, not the agent's: report it as a grading error
    # before the hidden tests run the agent's code.
    try:
        check_selection(fail_to_pass, pass_to_pass)
    except ValueError as exc:
        return SubScore(name="tests", value=0.0, info={"error": str(exc)})

    with tempfile.TemporaryDirectory(prefix=".hud-junit-", dir=workspace.root) as directory:
        Path(directory).chmod(0o733)
        report = Path(directory) / "report.xml"
        process = await workspace.run(
            ["/bin/bash", "-lc", command.replace("{junit_path}", shlex.quote(str(report)))],
            cwd=str(workspace.root),
            env={"HOME": "/tmp", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            max_wait=timeout_seconds,
            scope="environment",
        )
        info = {
            "exit_code": process.returncode,
            "stdout": process.stdout.decode(errors="replace"),
            "stderr": process.stderr.decode(errors="replace"),
            "timed_out": process.timed_out,
        }
        failed = SubScore(name="tests", value=0.0, info=info)
        # From here on the hidden tests have run against the agent's code, which can kill
        # the run, hang it, or tamper with the report. A timeout or a missing, unreadable or
        # malformed report is therefore a graded failure, not a grading error: the reason
        # stays in `report_error`, but the run still counts as 0 in the job's mean reward.
        if process.timed_out:
            return failed
        if not report.is_file():
            return failed.model_copy(update={"info": {**info, "report_error": "test command did not write JUnit XML"}})
        if binary and process.returncode != 0:
            return failed
        try:
            result = score_tests(parse_junit(report), fail_to_pass, pass_to_pass, binary)
        except (OSError, ValueError) as exc:
            return failed.model_copy(update={"info": {**info, "report_error": str(exc)}})
        return result.model_copy(update={"info": {**(result.info or {}), **info}})
