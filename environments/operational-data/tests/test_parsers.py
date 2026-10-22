"""How an unreadable upload is reported."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from data_sources.base import DataSource, ExportError, TaskContext
from data_sources.sources import SOURCES
from data_sources.sources import slack as slack_source


@pytest.mark.parametrize("source", list(SOURCES.values()), ids=list(SOURCES))
def test_an_unreadable_upload_raises_export_error(
    source: type[DataSource], tmp_path: Path, context: TaskContext
):
    path = tmp_path / "export.bin"
    path.write_bytes(b"\x00\x01 not an export \xff")
    with pytest.raises(ExportError):
        source.load([path], context)


def _slack_export(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)


@pytest.mark.parametrize("limit", ["member", "archive"])
def test_slack_rejects_oversized_expanded_json_before_reading_it(
    tmp_path: Path, context: TaskContext, monkeypatch: pytest.MonkeyPatch, limit: str
):
    path = tmp_path / "slack.zip"
    entries = [("users.json", b"[]"), ("ops/2026-03-31.json", b" " * 128)]
    _slack_export(path, entries)
    monkeypatch.setattr(
        slack_source, "MAX_JSON_BYTES", 127 if limit == "member" else 128, raising=False
    )
    monkeypatch.setattr(
        slack_source, "MAX_EXPANDED_JSON_BYTES", 130 if limit == "member" else 129, raising=False
    )

    def cannot_read(*args, **kwargs):
        raise AssertionError("oversized Slack exports must fail before decompression")

    monkeypatch.setattr(ZipFile, "read", cannot_read)
    with pytest.raises(ExportError, match="slack.zip: .*JSON limit"):
        slack_source.Slack.load([path], context)


@pytest.mark.parametrize("over_limit", [False, True])
def test_slack_counts_and_reads_each_duplicate_json_entry(
    tmp_path: Path, context: TaskContext, monkeypatch: pytest.MonkeyPatch, over_limit: bool
):
    path = tmp_path / "slack.zip"
    entries = [
        ("export/channels.json", json.dumps([{"id": "C1", "name": "ops"}]).encode()),
        (
            "export/ops/2026-03-31.json",
            json.dumps([{"ts": "1774954800.0", "text": "First message"}]).encode(),
        ),
        (
            "export/ops/2026-03-31.json",
            json.dumps([{"ts": "1774954801.0", "text": "Second message"}]).encode(),
        ),
    ]
    with pytest.warns(UserWarning, match="Duplicate name"):
        _slack_export(path, entries)
    total = sum(len(content) for _, content in entries)
    monkeypatch.setattr(
        slack_source, "MAX_EXPANDED_JSON_BYTES", total - int(over_limit), raising=False
    )
    if over_limit:
        with pytest.raises(ExportError, match="expanded JSON limit"):
            slack_source.Slack.load([path], context)
    else:
        slack = slack_source.Slack.load([path], context)
        assert [message.text for message in slack.read_channel("ops").items] == [
            "First message",
            "Second message",
        ]


def test_slack_expansion_budget_applies_to_json_members_only(
    tmp_path: Path, context: TaskContext, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "slack.zip"
    message = json.dumps([{"ts": "1774954800.0", "text": "Daily update"}]).encode()
    _slack_export(path, [("ops/2026-03-31.json", message), ("attachments/image.png", b"x" * 256)])
    monkeypatch.setattr(slack_source, "MAX_JSON_BYTES", len(message), raising=False)
    monkeypatch.setattr(slack_source, "MAX_EXPANDED_JSON_BYTES", len(message), raising=False)
    slack = slack_source.Slack.load([path], context)
    assert slack.read_channel("ops").items[0].text == "Daily update"
