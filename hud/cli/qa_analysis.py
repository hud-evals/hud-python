"""Interpret QA-agent result blobs the same way the platform review UI does."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

_QA_V1_VERDICTS = frozenset({"passed", "failed", "unknown"})
_BOOLEAN_KEYS = (
    ("is_false_negative", "False Negative"),
    ("is_false_positive", "False Positive"),
    ("is_reward_hacking", "Reward Hacking"),
    ("is_prompt_misaligned", "Prompt Misaligned"),
)
_REWARD_HACKING_STRATEGIES = frozenset(
    {
        "none",
        "test_manipulation",
        "output_hardcoding",
        "check_disabling",
        "environment_exploitation",
        "grader_exploitation",
        "method_substitution",
        "shortcut",
        "other",
    }
)
_SKIP_EXTRA_KEYS = frozenset(
    {
        "confidence",
        "reasoning",
        "summary",
        "output",
        "problems",
        "root_cause",
        "content",
        "reward",
        "score",
        "schema_version",
        "verdict",
        "findings",
        "issues",
        "gaps",
        "metadata",
    }
)


@dataclass(frozen=True)
class QaFinding:
    title: str
    description: str
    fault: str | None = None


@dataclass(frozen=True)
class QaPresentation:
    kind: str
    tag: str
    label: str
    answer: str | None
    summary: str | None
    confidence: str | None
    findings: tuple[QaFinding, ...]


def is_standard_result_blob(text: str) -> bool:
    """True when text is a structured QA result, not agent prose."""
    loaded = _loads_object(text)
    if loaded is None:
        return False
    return str(_parse_object(loaded).get("kind") or "unknown") != "unknown"


def presentation_for_result(row: dict[str, Any]) -> QaPresentation:
    """Review chrome for one QA result row: pass/fail tag plus agent-shaped body."""
    status = str(row.get("status") or "")
    error = row.get("error")
    if status == "error":
        return QaPresentation(
            kind="unknown",
            tag="failed",
            label="QA Result",
            answer=None,
            summary=str(error) if error else "QA run failed.",
            confidence=None,
            findings=(),
        )
    if status and status != "completed":
        return QaPresentation(
            kind="pending",
            tag="unknown",
            label=status,
            answer=None,
            summary=str(error) if error else None,
            confidence=None,
            findings=(),
        )
    payload = row.get("canonical_result")
    if not isinstance(payload, dict):
        payload = row.get("result")
    if not isinstance(payload, dict):
        return _unknown(str(error) if error else None)
    return _present(_parse(payload))


def _parse(payload: dict[str, Any] | str) -> dict[str, Any]:
    parsed = _coerce_object(payload)
    if parsed is None:
        if isinstance(payload, str) and payload.strip():
            return {
                "kind": "unknown",
                "label": "QA Result",
                "value": "unknown",
                "summary": payload.strip(),
            }
        return {"kind": "unknown", "label": "QA Result", "value": "unknown"}
    return _parse_object(parsed)


def _coerce_object(payload: dict[str, Any] | str) -> dict[str, Any] | None:
    parsed: dict[str, Any] | None
    if isinstance(payload, str):
        parsed = _loads_object(payload)
    elif isinstance(payload, dict):
        parsed = payload
    else:
        return None
    if parsed is None:
        return None
    output = parsed.get("output")
    if isinstance(output, str):
        inner = _loads_object(output)
        if inner is not None:
            parsed = inner
    content = parsed.get("content")
    if isinstance(content, str):
        inner = _loads_object(content)
        if inner is not None:
            parsed = inner
    return parsed if isinstance(parsed, dict) else None


def _loads_object(raw: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _parse_object(parsed: dict[str, Any]) -> dict[str, Any]:
    confidence = _confidence_text(parsed.get("confidence"))
    reasoning = parsed.get("reasoning") if isinstance(parsed.get("reasoning"), str) else None
    summary = parsed.get("summary") if isinstance(parsed.get("summary"), str) else None
    if summary is None and isinstance(parsed.get("root_cause"), str):
        summary = parsed["root_cause"]

    qa_findings = _first_finding_list(parsed)
    verdict = parsed.get("verdict")
    if (
        isinstance(verdict, str)
        and verdict in _QA_V1_VERDICTS
        and (parsed.get("schema_version") == "qa_agent_result.v1" or qa_findings is not None)
    ):
        return {
            "kind": "qa_result",
            "label": "QA Result",
            "value": verdict,
            "summary": summary,
            "reasoning": reasoning,
            "confidence": confidence,
            "findings": tuple(
                finding
                for finding in (_normalize_qa_finding(item) for item in (qa_findings or []))
                if finding.title
            ),
        }

    for key, label in _BOOLEAN_KEYS:
        if key in parsed:
            return _boolean_verdict(parsed, key, label, summary, reasoning, confidence)

    if _is_prompt_alignment(parsed):
        return _boolean_verdict(
            parsed,
            "is_prompt_misaligned",
            "Prompt Misaligned",
            summary,
            reasoning,
            confidence,
            inferred=_inferred_prompt_alignment(parsed),
        )
    if _is_reward_hacking(parsed):
        return _boolean_verdict(
            parsed,
            "is_reward_hacking",
            "Reward Hacking",
            summary,
            reasoning,
            confidence,
            inferred=_inferred_reward_hacking(parsed),
        )

    if isinstance(parsed.get("problems"), list):
        findings = tuple(
            finding
            for finding in (_normalize_problem(item) for item in parsed["problems"])
            if finding is not None
        )
        return {
            "kind": "problems",
            "label": "Failure Analysis",
            "value": _failure_analysis_label(findings),
            "summary": summary,
            "reasoning": reasoning,
            "confidence": confidence,
            "findings": findings,
        }

    category = parsed.get("failure_category", parsed.get("failure_mode"))
    if "failure_category" in parsed or "failure_mode" in parsed:
        if (
            not isinstance(category, str)
            or not category.strip()
            or category.strip().lower() in {"unknown", "unavailable", "null"}
        ):
            return {
                "kind": "unknown",
                "label": "Failure Analysis",
                "value": "unknown",
                "summary": summary,
                "reasoning": reasoning,
                "confidence": confidence,
            }
        return {
            "kind": "category",
            "label": "Failure Analysis",
            "value": category.strip(),
            "summary": summary,
            "reasoning": reasoning,
            "confidence": confidence,
            "findings": (QaFinding(title=category.strip().replace("_", " "), description=""),),
        }

    return {
        "kind": "unknown",
        "label": "QA Result",
        "value": "unknown",
        "summary": summary or reasoning,
        "reasoning": reasoning,
        "confidence": confidence,
    }


def _boolean_verdict(
    parsed: dict[str, Any],
    key: str,
    label: str,
    summary: str | None,
    reasoning: str | None,
    confidence: str | None,
    *,
    inferred: bool | None = None,
) -> dict[str, Any]:
    raw = parsed.get(key)
    value: bool | None
    if key in parsed and not isinstance(raw, bool):
        value = None
    elif isinstance(raw, bool):
        value = raw
    else:
        value = inferred
    extras = _string_extras(parsed, extra_exclude={key})
    if value is None:
        return {
            "kind": "unknown",
            "label": label,
            "value": "unknown",
            "summary": summary,
            "reasoning": reasoning,
            "confidence": confidence,
            "extras": extras,
        }
    return {
        "kind": "boolean",
        "label": label,
        "value": value,
        "summary": summary,
        "reasoning": reasoning,
        "confidence": confidence,
        "extras": extras,
    }


def _as_findings(parsed: dict[str, Any], *, require_title: bool = False) -> tuple[QaFinding, ...]:
    items = parsed.get("findings") or ()
    findings: list[QaFinding] = []
    for item in items:
        if not isinstance(item, QaFinding):
            continue
        if require_title and not item.title:
            continue
        findings.append(item)
    return tuple(findings)


def _present(parsed: dict[str, Any]) -> QaPresentation:
    kind = str(parsed.get("kind") or "unknown")
    label = str(parsed.get("label") or "QA Result")
    summary = _narrative(parsed)
    confidence = parsed.get("confidence") if isinstance(parsed.get("confidence"), str) else None
    if kind == "qa_result":
        tag = str(parsed.get("value") or "unknown")
        if tag not in _QA_V1_VERDICTS:
            tag = "unknown"
        findings = _as_findings(parsed, require_title=True)
        return QaPresentation(kind, tag, label, None, summary, confidence, findings)
    if kind == "boolean":
        value = parsed.get("value")
        if value is True:
            tag = "failed"
            answer = "yes"
        elif value is False:
            tag = "passed"
            answer = "no"
        else:
            tag = "unknown"
            answer = "unknown"
        return QaPresentation(kind, tag, label, answer, summary, confidence, ())
    if kind == "problems":
        findings = _as_findings(parsed)
        tag = "failed" if findings else "passed"
        answer = str(parsed.get("value") or ("Agent failure" if findings else "No failure"))
        return QaPresentation(kind, tag, label, answer, summary, confidence, findings)
    if kind == "category":
        answer = str(parsed.get("value") or "unknown").replace("_", " ")
        findings = _as_findings(parsed)
        return QaPresentation(kind, "failed", label, answer, summary, confidence, findings)
    return QaPresentation(
        "unknown",
        "unknown",
        label,
        None,
        summary,
        confidence,
        (),
    )


def _narrative(parsed: dict[str, Any]) -> str | None:
    parts: list[str] = []
    summary = parsed.get("summary")
    reasoning = parsed.get("reasoning")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())
    if isinstance(reasoning, str) and reasoning.strip() and reasoning.strip() not in parts:
        parts.append(reasoning.strip())
    extras = parsed.get("extras")
    if isinstance(extras, dict):
        for key, value in extras.items():
            parts.append(f"{key.replace('_', ' ')}: {value}")
    return "\n\n".join(parts) if parts else None


def _unknown(summary: str | None) -> QaPresentation:
    return QaPresentation("unknown", "unknown", "QA Result", None, summary, None, ())


def _confidence_text(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        lower = raw.strip().lower()
        if lower in {"high", "medium", "low", "very high", "very low"}:
            return lower
        try:
            number = float(lower)
        except ValueError:
            return raw.strip()
        raw = number
    if isinstance(raw, int | float):
        value = raw / 100 if raw > 1 else raw
        return f"{round(value * 100)}%"
    return None


def _first_finding_list(parsed: dict[str, Any]) -> list[Any] | None:
    for key in ("findings", "issues", "gaps"):
        value = parsed.get(key)
        if isinstance(value, list):
            return value
    return None


def _normalize_problem(raw: Any) -> QaFinding | None:
    if not isinstance(raw, dict):
        return None
    title = (
        raw.get("problem") or raw.get("title") or raw.get("name") or raw.get("failure_mode") or ""
    )
    if not isinstance(title, str) or not title.strip():
        return None
    description = _problem_description(raw)
    fault = raw.get("fault") if isinstance(raw.get("fault"), str) else None
    return QaFinding(title=title.strip(), description=description, fault=fault)


def _normalize_qa_finding(raw: Any) -> QaFinding:
    if not isinstance(raw, dict):
        return QaFinding(title="", description="")
    title = raw.get("summary") if isinstance(raw.get("summary"), str) else ""
    description = _problem_description(raw)
    return QaFinding(title=title.strip(), description=description)


def _problem_description(raw: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("description", "root_cause", "impact"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    remedy = raw.get("what_should_have_happened") or raw.get("recommended_action")
    if isinstance(remedy, str) and remedy.strip():
        parts.append(remedy.strip())
    return "\n\n".join(parts)


def _normalize_owner(raw: str | None) -> str:
    if raw is None:
        return "unclear"
    owner = raw.strip().lower()
    if owner in {"agent", "agent_failure"}:
        return "agent"
    if owner in {"eval", "evaluation", "eval_failure", "evaluation_failure"}:
        return "eval"
    if owner in {"platform", "platform_failure"}:
        return "platform"
    return "unclear"


def _failure_analysis_label(findings: tuple[QaFinding, ...]) -> str:
    if not findings:
        return "No failure"
    owners = {_normalize_owner(item.fault) for item in findings}
    if len(owners) != 1:
        return "Mixed failure"
    owner = next(iter(owners))
    return {
        "agent": "Agent failure",
        "eval": "Evaluation failure",
        "platform": "Platform failure",
        "unclear": "Unclear",
    }[owner]


def _is_prompt_alignment(parsed: dict[str, Any]) -> bool:
    return (
        "is_prompt_misaligned" in parsed
        or isinstance(parsed.get("grader_check"), str)
        or isinstance(parsed.get("prompt_quote"), str)
        or isinstance(parsed.get("misalignment_proof"), str)
    )


def _is_reward_hacking(parsed: dict[str, Any]) -> bool:
    strategy = parsed.get("hacking_strategy")
    return "is_reward_hacking" in parsed or (
        isinstance(strategy, str) and strategy in _REWARD_HACKING_STRATEGIES
    )


def _inferred_prompt_alignment(parsed: dict[str, Any]) -> bool | None:
    proof = parsed.get("misalignment_proof")
    if isinstance(proof, str):
        return bool(proof.strip())
    return None


def _inferred_reward_hacking(parsed: dict[str, Any]) -> bool | None:
    strategy = parsed.get("hacking_strategy")
    if not isinstance(strategy, str) or strategy not in _REWARD_HACKING_STRATEGIES:
        return None
    return strategy != "none"


def _string_extras(parsed: dict[str, Any], *, extra_exclude: set[str]) -> dict[str, str]:
    extras: dict[str, str] = {}
    skip = _SKIP_EXTRA_KEYS | extra_exclude | {key for key, _ in _BOOLEAN_KEYS}
    for key, value in parsed.items():
        if key in skip or value is None:
            continue
        if isinstance(value, str | int | float | bool):
            extras[key] = str(value)
    return extras
