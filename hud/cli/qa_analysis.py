"""Turn a QA result row into pass/fail review chrome for the CLI TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

_QA_V1 = frozenset({"passed", "failed", "unknown"})
_BOOLEAN_KEYS = (
    ("is_false_negative", "False Negative"),
    ("is_false_positive", "False Positive"),
    ("is_reward_hacking", "Reward Hacking"),
    ("is_prompt_misaligned", "Prompt Misaligned"),
)
_CAUSE = {
    "agent": "Agent failure",
    "eval": "Evaluation failure",
    "platform": "Platform failure",
}


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
    parsed = _loads(text)
    return parsed is not None and _from_blob(parsed).kind != "unknown"


def presentation_for_result(row: dict[str, Any]) -> QaPresentation:
    status = str(row.get("status") or "")
    error = row.get("error")
    if status == "error":
        return _view("unknown", "failed", summary=str(error) if error else "QA run failed.")
    if status and status != "completed":
        return _view("pending", "unknown", label=status, summary=str(error) if error else None)
    payload = row.get("canonical_result")
    if not isinstance(payload, dict):
        payload = row.get("result")
    blob = _unwrap(payload) if isinstance(payload, dict | str) else None
    if blob is None:
        return _view("unknown", "unknown", summary=str(error) if error else None)
    return _from_blob(blob)


def _from_blob(parsed: dict[str, Any]) -> QaPresentation:
    summary = _text(parsed.get("summary"), parsed.get("reasoning"))
    confidence = _confidence(parsed.get("confidence"))

    verdict = parsed.get("verdict")
    findings_raw = parsed.get("findings")
    if (
        isinstance(verdict, str)
        and verdict in _QA_V1
        and (parsed.get("schema_version") == "qa_agent_result.v1" or isinstance(findings_raw, list))
    ):
        findings = _findings(findings_raw if isinstance(findings_raw, list) else [], "summary")
        return _view(
            "qa_result",
            verdict,
            summary=summary,
            confidence=confidence,
            findings=findings,
        )

    for key, label in _BOOLEAN_KEYS:
        if key not in parsed:
            continue
        value = parsed[key]
        if not isinstance(value, bool):
            return _view("unknown", "unknown", label=label, summary=summary, confidence=confidence)
        return _view(
            "boolean",
            "failed" if value else "passed",
            label=label,
            answer="yes" if value else "no",
            summary=summary,
            confidence=confidence,
        )

    if isinstance(parsed.get("problems"), list):
        findings = _findings(parsed["problems"], "problem", "title")
        owners = {_finding_owner(item.fault) for item in findings}
        if not findings:
            cause, tag = "No failure", "passed"
        elif len(owners) != 1:
            cause, tag = "Mixed failure", "failed"
        elif "unclear" in owners:
            cause, tag = "Unclear", "failed"
        else:
            cause, tag = _CAUSE[next(iter(owners))], "failed"
        return _view(
            "problems",
            tag,
            label="Failure Analysis",
            answer=cause,
            summary=summary,
            confidence=confidence,
            findings=findings,
        )

    return _view("unknown", "unknown", summary=summary, confidence=confidence)


def _view(
    kind: str,
    tag: str,
    *,
    label: str = "QA Result",
    answer: str | None = None,
    summary: str | None = None,
    confidence: str | None = None,
    findings: tuple[QaFinding, ...] = (),
) -> QaPresentation:
    return QaPresentation(kind, tag, label, answer, summary, confidence, findings)


def _unwrap(payload: dict[str, Any] | str) -> dict[str, Any] | None:
    parsed: dict[str, Any] | None = _loads(payload) if isinstance(payload, str) else payload
    if parsed is None:
        return None
    for key in ("output", "content"):
        raw = parsed.get(key)
        if isinstance(raw, str):
            inner = _loads(raw)
            if inner is not None:
                parsed = inner
    return parsed if isinstance(parsed, dict) else None


def _loads(raw: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _text(*values: Any) -> str | None:
    parts: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    return "\n\n".join(parts) if parts else None


def _confidence(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        lower = raw.strip().lower()
        labels = {"high", "medium", "low", "very high", "very low"}
        return lower if lower in labels else raw.strip()
    if isinstance(raw, int | float):
        value = raw / 100 if raw > 1 else raw
        return f"{round(value * 100)}%"
    return None


def _findings(items: list[Any], *title_keys: str) -> tuple[QaFinding, ...]:
    findings: list[QaFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = next(
            (
                item[key].strip()
                for key in title_keys
                if isinstance(item.get(key), str) and item[key].strip()
            ),
            "",
        )
        if not title:
            continue
        description = item.get("description")
        fault = item.get("fault") if isinstance(item.get("fault"), str) else None
        findings.append(
            QaFinding(
                title=title,
                description=description.strip() if isinstance(description, str) else "",
                fault=fault,
            )
        )
    return tuple(findings)


def _finding_owner(fault: str | None) -> str:
    if fault is None:
        return "unclear"
    owner = fault.strip().lower()
    if owner in _CAUSE:
        return owner
    return "unclear"
