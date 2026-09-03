"""Normalize QA-agent blobs to ``qa_agent_result.v1`` and render review chrome."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast

QA_AGENT_RESULT_V1 = "qa_agent_result.v1"
QaVerdict = Literal["passed", "failed", "unknown"]

_QA_V1 = frozenset({"passed", "failed", "unknown"})
_BOOLEAN_KEYS = (
    ("is_false_negative", "False Negative"),
    ("is_false_positive", "False Positive"),
    ("is_reward_hacking", "Reward Hacking"),
    ("is_prompt_misaligned", "Prompt Misaligned"),
)
_BOOLEAN_EXTRAS = {
    "is_reward_hacking": ("hacking_strategy",),
    "is_prompt_misaligned": ("grader_check", "prompt_quote", "misalignment_proof"),
}
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
    if parsed is None:
        return False
    return str(_blob_to_v1(parsed)["metadata"].get("kind") or "unknown") != "unknown"


def to_qa_agent_result_v1(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return a ``qa_agent_result.v1`` object for a platform QA result row.

    Pending rows (queued, running, …) return ``None``. Completed analysis blobs
    — v1, boolean judges, and Failure Analysis ``problems`` — are rewritten to
    the same schema. Error rows become ``verdict=failed``. Unrecognized
    completed payloads become ``verdict=unknown``.
    """
    status = str(row.get("status") or "")
    error = row.get("error")
    if status == "error":
        return _v1(
            "failed",
            summary=str(error) if error else "QA run failed.",
            metadata={"kind": "unknown"},
        )
    if status and status != "completed":
        return None
    payload = row.get("canonical_result")
    if not isinstance(payload, dict):
        payload = row.get("result")
    blob = _unwrap(payload) if isinstance(payload, dict | str) else None
    if blob is None:
        return _v1("unknown", summary=str(error) if error else None)
    return _blob_to_v1(blob)


def presentation_for_result(row: dict[str, Any]) -> QaPresentation:
    status = str(row.get("status") or "")
    error = row.get("error")
    if status and status not in {"completed", "error"}:
        return _view("pending", "unknown", label=status, summary=str(error) if error else None)
    canonical = to_qa_agent_result_v1(row)
    if canonical is None:
        return _view("unknown", "unknown", summary=str(error) if error else None)
    return _present_v1(canonical)


def _present_v1(parsed: dict[str, Any]) -> QaPresentation:
    metadata = parsed.get("metadata")
    extra = metadata if isinstance(metadata, dict) else {}
    kind = extra.get("kind") if isinstance(extra.get("kind"), str) else "qa_result"
    label = extra.get("label") if isinstance(extra.get("label"), str) else "QA Result"
    answer = extra.get("answer") if isinstance(extra.get("answer"), str) else None
    confidence = extra.get("confidence") if isinstance(extra.get("confidence"), str) else None
    verdict = parsed.get("verdict")
    tag = verdict if isinstance(verdict, str) and verdict in _QA_V1 else "unknown"
    findings = tuple(
        QaFinding(
            title=str(item["summary"]),
            description=str(item.get("description") or ""),
            fault=item.get("fault") if isinstance(item.get("fault"), str) else None,
        )
        for item in parsed.get("findings") or []
        if isinstance(item, dict) and isinstance(item.get("summary"), str) and item["summary"]
    )
    summary = parsed.get("summary")
    return _view(
        kind,
        tag,
        label=label,
        answer=answer,
        summary=summary if isinstance(summary, str) else None,
        confidence=confidence,
        findings=findings,
    )


def _blob_to_v1(parsed: dict[str, Any]) -> dict[str, Any]:
    summary = _text(parsed.get("summary"), parsed.get("reasoning"))
    confidence = _confidence(parsed.get("confidence"))

    verdict = parsed.get("verdict")
    findings_raw = parsed.get("findings")
    if (
        isinstance(verdict, str)
        and verdict in _QA_V1
        and (parsed.get("schema_version") == QA_AGENT_RESULT_V1 or isinstance(findings_raw, list))
    ):
        metadata = dict(parsed["metadata"]) if isinstance(parsed.get("metadata"), dict) else {}
        metadata.setdefault("kind", "qa_result")
        if confidence and "confidence" not in metadata:
            metadata["confidence"] = confidence
        raw_findings = findings_raw if isinstance(findings_raw, list) else []
        return _v1(
            cast("QaVerdict", verdict),
            summary=summary,
            findings=_finding_dicts(raw_findings, "summary"),
            metadata=metadata,
        )

    for key, label in _BOOLEAN_KEYS:
        if key not in parsed:
            continue
        value = parsed[key]
        metadata: dict[str, Any] = {"kind": "boolean", "label": label, key: value}
        if confidence:
            metadata["confidence"] = confidence
        for extra in _BOOLEAN_EXTRAS.get(key, ()):
            extra_value = parsed.get(extra)
            if isinstance(extra_value, str) and extra_value.strip():
                metadata[extra] = extra_value.strip()
        if not isinstance(value, bool):
            return _v1("unknown", summary=summary, metadata=metadata)
        metadata["answer"] = "yes" if value else "no"
        return _v1("failed" if value else "passed", summary=summary, metadata=metadata)

    if isinstance(parsed.get("problems"), list):
        findings = _finding_dicts(parsed["problems"], "problem", "title")
        owners = {_finding_owner(item.get("fault")) for item in findings}
        if not findings:
            cause, tag = "No failure", "passed"
        elif len(owners) != 1:
            cause, tag = "Mixed failure", "failed"
        elif "unclear" in owners:
            cause, tag = "Unclear", "failed"
        else:
            cause, tag = _CAUSE[next(iter(owners))], "failed"
        metadata = {"kind": "problems", "label": "Failure Analysis", "answer": cause}
        if confidence:
            metadata["confidence"] = confidence
        return _v1(cast("QaVerdict", tag), summary=summary, findings=findings, metadata=metadata)

    metadata = {"kind": "unknown"}
    if confidence:
        metadata["confidence"] = confidence
    return _v1("unknown", summary=summary, metadata=metadata)


def _v1(
    verdict: QaVerdict,
    *,
    summary: str | None = None,
    findings: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": QA_AGENT_RESULT_V1,
        "verdict": verdict,
        "summary": summary,
        "findings": findings or [],
        "metadata": metadata or {},
    }


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


def _finding_dicts(items: list[Any], *title_keys: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
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
        finding: dict[str, Any] = {"summary": title}
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            finding["description"] = description.strip()
        fault = item.get("fault") if isinstance(item.get("fault"), str) else None
        if fault:
            finding["fault"] = fault
        findings.append(finding)
    return findings


def _finding_owner(fault: Any) -> str:
    if not isinstance(fault, str):
        return "unclear"
    owner = fault.strip().lower()
    if owner in _CAUSE:
        return owner
    return "unclear"
