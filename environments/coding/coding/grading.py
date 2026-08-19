"""JUnit parsing and test-set scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree

from hud.graders import EvaluationResult, SubScore


@dataclass(frozen=True, slots=True)
class JUnitCase:
    nodeid: str
    passed: bool
    skipped: bool
    message: str | None = None


def parse_junit(path: Path) -> list[JUnitCase]:
    cases: list[JUnitCase] = []
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"invalid JUnit XML: {exc}") from exc
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
        cases.append(JUnitCase(nodeid=nodeid, passed=failure is None and not skipped, skipped=skipped, message=message))
    return cases


def score_tests(
    cases: list[JUnitCase],
    fail_to_pass: list[str] | None,
    pass_to_pass: list[str] | None,
    use_binary_score: bool,
) -> EvaluationResult:
    f2p = [nodeid.replace("::", ".") for nodeid in fail_to_pass or []]
    p2p = [nodeid.replace("::", ".") for nodeid in pass_to_pass or []]
    if len(set(f2p)) != len(f2p) or len(set(p2p)) != len(p2p):
        raise ValueError("test node IDs must be unique")
    if overlap := set(f2p) & set(p2p):
        raise ValueError(f"test node IDs cannot be both fail-to-pass and pass-to-pass: {sorted(overlap)}")

    selected = fail_to_pass is not None or pass_to_pass is not None
    expected = f2p + p2p if selected else [case.nodeid for case in cases if not case.skipped]
    if selected and not expected:
        raise ValueError("at least one scored test node ID is required")
    reported = [case.nodeid for case in cases]
    if len(set(reported)) != len(reported):
        raise ValueError("JUnit test case IDs must be unique")

    passed = {case.nodeid for case in cases if case.passed}

    def expected_test_passed(nodeid: str) -> bool:
        if nodeid in reported:
            return nodeid in passed
        if nodeid.count("[") <= nodeid.count("]"):
            return False
        matches = [case_id for case_id in reported if case_id.startswith(nodeid)]
        return bool(matches) and all(case_id in passed for case_id in matches)

    passed_count = sum(expected_test_passed(nodeid) for nodeid in expected)
    partial_score = passed_count / len(expected) if expected else 0.0
    reward = float(partial_score == 1.0) if use_binary_score else partial_score

    subscores = [SubScore(name="tests", value=reward, weight=1.0)]
    info = {
        "passed": passed_count,
        "total": len(expected),
        "all_testcases": [asdict(case) for case in cases],
    }
    if selected:
        f2p_passed = sum(expected_test_passed(nodeid) for nodeid in f2p)
        p2p_passed = sum(expected_test_passed(nodeid) for nodeid in p2p)
        subscores.extend(
            [
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
        )
        info.update(
            {
                "f2p_passed": f2p_passed,
                "f2p_total": len(f2p),
                "p2p_passed": p2p_passed,
                "p2p_total": len(p2p),
            }
        )

    return EvaluationResult(
        reward=reward,
        content=f"{passed_count}/{len(expected)} scored tests passed",
        subscores=subscores,
        info=info,
    )
