from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hud.environment.seatbelt import (
    SeatbeltPolicyInputs,
    generate_seatbelt_profile,
    policy_params,
    seatbelt_argv,
    usable_seatbelt,
)

pytestmark = [
    pytest.mark.darwin,
    pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS-only"),
]


@pytest.fixture(autouse=True)
def _require_usable_seatbelt() -> None:
    if usable_seatbelt() is None:
        pytest.skip("sandbox-exec not available or probe failed")


def test_usable_seatbelt_probe() -> None:
    assert usable_seatbelt() is not None


def test_denies_read_outside_writable_root(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("nope", encoding="utf-8")
    inputs = SeatbeltPolicyInputs(
        writable_roots=(ws,),
        readable_roots=(),
        allow_tmpdir_write=False,
    )
    profile = generate_seatbelt_profile(inputs)
    params = policy_params(inputs)
    argv = seatbelt_argv(["/bin/cat", str(secret)], profile=profile, params=params)
    proc = subprocess.run(argv, capture_output=True, timeout=15, check=False)
    assert proc.returncode != 0


def test_allows_write_inside_root(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "out.txt"
    inputs = SeatbeltPolicyInputs(writable_roots=(ws,), readable_roots=())
    profile = generate_seatbelt_profile(inputs)
    params = policy_params(inputs)
    argv = seatbelt_argv(
        ["/bin/sh", "-c", f"echo hi > {target}"],
        profile=profile,
        params=params,
    )
    proc = subprocess.run(argv, capture_output=True, timeout=15, check=False)
    assert proc.returncode == 0
    assert target.read_text(encoding="utf-8").strip() == "hi"


def test_denies_network_without_allowance(tmp_path: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        pytest.skip("curl not installed")

    ws = tmp_path / "ws"
    ws.mkdir()
    inputs = SeatbeltPolicyInputs(
        writable_roots=(ws,),
        readable_roots=(),
        allow_all_network=False,
        proxy_loopback_ports=(),
    )
    profile = generate_seatbelt_profile(inputs)
    params = policy_params(inputs)
    argv = seatbelt_argv(
        [curl, "-m", "2", "-sS", "https://example.com"],
        profile=profile,
        params=params,
    )
    proc = subprocess.run(argv, capture_output=True, timeout=15, check=False)
    assert proc.returncode != 0
