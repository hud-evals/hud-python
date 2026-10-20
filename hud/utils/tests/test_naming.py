from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud.utils.naming import normalize_environment_name, rename_environment_in_source

if TYPE_CHECKING:
    from pathlib import Path


def test_normalize_environment_name() -> None:
    assert normalize_environment_name("terminal-bench") == "terminal-bench"
    assert normalize_environment_name("My Cool_Bench") == "my-cool-bench"
    assert normalize_environment_name("bench@2.0!") == "bench20"
    assert normalize_environment_name("--hello--") == "hello"
    assert normalize_environment_name("a---b") == "a-b"
    assert normalize_environment_name("@#$") == "environment"
    assert normalize_environment_name("", default="converted") == "converted"


@pytest.mark.parametrize("entrypoint", ["env.py", "environment/env.py"])
def test_rename_environment_in_source_preserves_the_template_layout(
    tmp_path: Path, entrypoint: str
) -> None:
    source = tmp_path / entrypoint
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('env = Environment(name="coding")\n# preserve this\n')

    rename_environment_in_source(tmp_path, "coding", "My Demo")

    assert source.read_text() == 'env = Environment(name="my-demo")\n# preserve this\n'


def test_rename_environment_requires_one_literal_declaration(tmp_path: Path) -> None:
    (tmp_path / "env.py").write_text('env = Environment(name="another")\n')

    with pytest.raises(ValueError, match="expected one"):
        rename_environment_in_source(tmp_path, "coding", "demo")

    assert 'name="another"' in (tmp_path / "env.py").read_text()
