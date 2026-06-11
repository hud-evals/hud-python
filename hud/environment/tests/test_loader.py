"""``load_environment``: select an env from a source file by attr or env name."""

from __future__ import annotations

import pytest

from hud.environment import load_environment


def test_load_environment_selects_by_attr_or_env_name(tmp_path) -> None:
    module = tmp_path / "envs.py"
    module.write_text(
        """
from hud import Environment

first = Environment("env-one")
second = Environment("env-two")
""".strip(),
        encoding="utf-8",
    )

    assert load_environment(module, name="first").name == "env-one"
    assert load_environment(module, name="env-two").name == "env-two"
    with pytest.raises(ValueError, match="multiple Environments"):
        load_environment(module)
    with pytest.raises(ValueError, match="no Environment named 'missing'"):
        load_environment(module, name="missing")

    single = tmp_path / "single.py"
    single.write_text("from hud import Environment\nenv = Environment('only')\n", encoding="utf-8")
    assert load_environment(single).name == "only"
