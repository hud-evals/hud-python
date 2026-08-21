"""A Bash grader that scores selected tests from a JUnit report."""

from __future__ import annotations

import shlex
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from hud.graders import BashGrader, SubScore


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


def score_tests(
    cases: list[JUnitCase],
    fail_to_pass: list[str] | None,
    pass_to_pass: list[str] | None,
    binary: bool,
) -> SubScore:
    f2p = [nodeid.replace("::", ".") for nodeid in fail_to_pass or []]
    p2p = [nodeid.replace("::", ".") for nodeid in pass_to_pass or []]
    if len(set(f2p)) != len(f2p) or len(set(p2p)) != len(p2p):
        raise ValueError("test node IDs must be unique")
    if overlap := set(f2p) & set(p2p):
        raise ValueError(f"test node IDs cannot be in both groups: {sorted(overlap)}")

    selected = fail_to_pass is not None or pass_to_pass is not None
    expected = f2p + p2p if selected else [case.nodeid for case in cases if not case.skipped]
    if selected and not expected:
        raise ValueError("at least one scored test node ID is required")

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


class JUnitGrader(BashGrader):
    """Run a command and score selected JUnit test cases."""

    name = "JUnitGrader"

    @classmethod
    async def compute_score(
        cls,
        command: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        fail_to_pass: list[str] | None = None,
        pass_to_pass: list[str] | None = None,
        binary: bool = False,
        **kwargs: Any,
    ) -> SubScore:
        if command is None or "{junit_path}" not in command:
            raise ValueError("JUnitGrader command must contain {junit_path}")

        with tempfile.TemporaryDirectory(prefix="hud-junit-") as directory:
            report = Path(directory) / "report.xml"
            bash = await super().compute_score(
                command=command.replace("{junit_path}", shlex.quote(str(report))),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )
            if not report.is_file():
                raise RuntimeError("test command did not write JUnit XML")
            result = score_tests(parse_junit(report), fail_to_pass, pass_to_pass, binary)
            return result.model_copy(update={"info": {**(result.info or {}), **(bash.info or {})}})
