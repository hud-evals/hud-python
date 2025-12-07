"""Tests for CLI init module."""

from __future__ import annotations

from hud.cli.init import _replace_placeholders


class TestReplacePlaceholders:
    """Test placeholder replacement in template files."""

    def test_replace_in_pyproject(self, tmp_path):
        """Test replacing placeholders in pyproject.toml."""
        # Create server directory structure
        server_dir = tmp_path / "server"
        server_dir.mkdir()

        pyproject = server_dir / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "blank"
description = "blank environment"
""")

        modified = _replace_placeholders(tmp_path, "my-cool-env")

        # Normalize paths for cross-platform comparison
        modified_normalized = [p.replace("\\", "/") for p in modified]
        assert "server/pyproject.toml" in modified_normalized
        content = pyproject.read_text()
        assert "my_cool_env" in content
        assert "blank" not in content

    def test_replace_in_readme(self, tmp_path):
        """Test replacing placeholders in README.md."""
        readme = tmp_path / "README.md"
        readme.write_text("# blank\n\nThis is the blank environment.")

        modified = _replace_placeholders(tmp_path, "test-env")

        assert "README.md" in modified
        content = readme.read_text()
        assert "test_env" in content
        assert "blank" not in content

    def test_replace_in_tasks_json(self, tmp_path):
        """Test replacing placeholders in tasks.json."""
        tasks = tmp_path / "tasks.json"
        tasks.write_text('{"name": "blank", "tasks": []}')

        modified = _replace_placeholders(tmp_path, "my-tasks")

        assert "tasks.json" in modified
        content = tasks.read_text()
        assert "my_tasks" in content

    def test_no_replace_in_non_placeholder_files(self, tmp_path):
        """Test that non-placeholder files are not modified."""
        other_file = tmp_path / "other.py"
        other_file.write_text("# blank comment")

        modified = _replace_placeholders(tmp_path, "test")

        assert "other.py" not in modified
        content = other_file.read_text()
        assert "blank" in content  # Should be unchanged

    def test_skip_pycache_directories(self, tmp_path):
        """Test that __pycache__ directories are skipped."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()

        cached_file = pycache / "module.pyc"
        cached_file.write_text("blank")

        modified = _replace_placeholders(tmp_path, "test")

        # __pycache__ files should not be in modified list
        assert not any("__pycache__" in f for f in modified)

    def test_normalize_special_characters(self, tmp_path):
        """Test that environment name is normalized for Python identifiers."""
        server_dir = tmp_path / "server"
        server_dir.mkdir()

        pyproject = server_dir / "pyproject.toml"
        pyproject.write_text('name = "blank"')

        _replace_placeholders(tmp_path, "my cool-env.v2!")

        content = pyproject.read_text()
        # Special characters should be replaced with underscores
        assert "my_cool_env_v2_" in content

    def test_no_changes_when_no_placeholder(self, tmp_path):
        """Test that files without placeholder are not modified."""
        server_dir = tmp_path / "server"
        server_dir.mkdir()

        pyproject = server_dir / "pyproject.toml"
        pyproject.write_text('name = "other-name"')

        modified = _replace_placeholders(tmp_path, "test")

        assert "server/pyproject.toml" not in modified

    def test_nested_directory_structure(self, tmp_path):
        """Test replacement in nested directory structure."""
        # Create nested structure
        server_dir = tmp_path / "server"
        server_dir.mkdir()
        (server_dir / "pyproject.toml").write_text('name = "blank"')

        env_dir = tmp_path / "environment"
        env_dir.mkdir()
        (env_dir / "pyproject.toml").write_text('name = "blank"')
        (env_dir / "README.md").write_text("# blank environment")

        modified = _replace_placeholders(tmp_path, "nested-test")

        # Normalize paths for cross-platform comparison
        modified_normalized = [p.replace("\\", "/") for p in modified]
        assert "server/pyproject.toml" in modified_normalized
        assert "environment/pyproject.toml" in modified_normalized
        assert "environment/README.md" in modified_normalized
