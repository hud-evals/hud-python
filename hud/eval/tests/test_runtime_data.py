"""``RuntimeConfig.data``: platform data files declared by id for hosted rollouts.

The block is declarative: which data files, where the environment sees them.
It validates like the rest of ``RuntimeConfig`` (unknown keys rejected), is
omitted from JSON when unset, and is replaced whole by an override rather
than merged, so a per-task block never inherits a build's file list.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hud.eval import RuntimeConfig, RuntimeData, RuntimeDataFile

_FILE_ID = "e9032002-01bb-4511-af4f-d18477f75dda"


def test_data_block_defaults() -> None:
    """Only file ids are required; the mount path and mode have defaults."""
    config = RuntimeConfig(data=RuntimeData(files=[RuntimeDataFile(file_id=_FILE_ID)]))

    assert config.data is not None
    assert config.data.mount_path == "/data"
    assert config.data.mode == "overlay"
    assert config.data.files[0].path is None


def test_data_block_serializes_only_what_was_set() -> None:
    """JSON carries the block only when set, and only the fields that were set."""
    without = RuntimeConfig(image="my-env")
    with_data = RuntimeConfig.model_validate(
        {"data": {"files": [{"file_id": _FILE_ID, "path": "case_room/resume.pdf"}]}},
    )

    assert "data" not in without.model_dump(mode="json", exclude_unset=True)
    assert with_data.model_dump(mode="json", exclude_unset=True) == {
        "data": {"files": [{"file_id": _FILE_ID, "path": "case_room/resume.pdf"}]},
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"files": []}},
        {"data": {"files": [{"file_id": ""}]}},
        {"data": {"files": [{"file_id": _FILE_ID}], "mode": "rw"}},
        {"data": {"files": [{"file_id": _FILE_ID}], "bucket": "some-bucket"}},
        {"data": {"files": [{"file_id": _FILE_ID, "size_gb": 1}]}},
    ],
)
def test_data_block_rejects_malformed_payloads(payload: dict[str, object]) -> None:
    """Empty file lists, blank ids, unknown modes and unknown keys fail validation."""
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(payload)


def test_override_replaces_the_data_block_whole() -> None:
    """A per-task block replaces the base block's file list; it never merges into it."""
    base = RuntimeConfig(
        image="my-env",
        data=RuntimeData(files=[RuntimeDataFile(file_id=_FILE_ID)], mount_path="/mnt/base"),
    )
    override = RuntimeConfig(data=RuntimeData(files=[RuntimeDataFile(file_id="other-id")]))

    merged = base.with_overrides(override)

    assert merged.image == "my-env"
    assert merged.data is not None
    assert [f.file_id for f in merged.data.files] == ["other-id"]
    assert merged.data.mount_path == "/data"
    assert base.with_overrides(RuntimeConfig(image="new")).data == base.data
