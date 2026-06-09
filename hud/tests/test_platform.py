"""Platform transport models in ``hud._platform``."""

from __future__ import annotations

from hud._platform import PlatformClient, RegistryEnvironment


def test_registry_environment_from_record_prefers_display_name() -> None:
    env = RegistryEnvironment.from_record(
        {"id": "abc123456", "name": "raw", "name_display": "Pretty", "latest_version": "2"}
    )

    assert env.id == "abc123456"
    assert env.name == "Pretty"
    assert env.short_id == "abc12345"
    assert env.version_label == " v2"


def test_registry_environment_ref_accepts_uuid() -> None:
    envs = PlatformClient("https://api.example", {}).resolve_registry_environments(
        "12345678-1234-5678-1234-567812345678"
    )

    assert envs == [
        RegistryEnvironment(
            id="12345678-1234-5678-1234-567812345678",
            name="12345678...",
        )
    ]
