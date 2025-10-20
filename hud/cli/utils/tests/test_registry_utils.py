from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.utils.registry import (
    extract_digest_from_image,
    extract_name_and_tag,
    list_registry_entries,
    load_from_registry,
    save_to_registry,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_extract_digest_from_image_variants():
    assert extract_digest_from_image("repo/name@sha256:abcdef1234567890") == "abcdef123456"
    assert extract_digest_from_image("sha256:deadbeefcafebabe") == "deadbeefcafe"
    assert extract_digest_from_image("org/name:tag") == "tag"
    assert extract_digest_from_image("org/name") == "latest"


def test_extract_name_and_tag():
    assert extract_name_and_tag("docker.io/hudpython/test_init:latest@sha256:abc") == (
        "hudpython/test_init",
        "latest",
    )
    assert extract_name_and_tag("myorg/myenv:v1.0") == ("myorg/myenv", "v1.0")
    assert extract_name_and_tag("myorg/myenv") == ("myorg/myenv", "latest")


def test_save_load_list_registry(tmp_path: Path, monkeypatch):
    # Redirect registry dir to temp
    from hud.cli.utils import registry as mod

    monkeypatch.setattr(mod, "get_registry_dir", lambda: tmp_path)

    data = {"image": "org/name:tag", "build": {"version": "0.1.0"}}
    saved = save_to_registry(data, "org/name:tag@sha256:abcdef0123456789", verbose=True)
    assert saved is not None and saved.exists()

    # Digest directory was created
    entries = list_registry_entries()
    assert len(entries) == 1

    digest, _ = entries[0]
    loaded = load_from_registry(digest)
    assert loaded and loaded.get("image") == "org/name:tag"
