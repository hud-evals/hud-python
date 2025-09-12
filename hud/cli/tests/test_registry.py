"""Tests for registry.py - Local registry management for HUD environments."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import yaml

from hud.cli.utils.registry import (
    extract_digest_from_image,
    extract_name_and_tag,
    get_registry_dir,
    list_registry_entries,
    load_from_registry,
    save_to_registry,
)


class TestGetRegistryDir:
    """Test getting registry directory."""

    def test_get_registry_dir(self):
        """Test default registry directory."""
        with mock.patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path("/home/user")

            registry_dir = get_registry_dir()
            assert registry_dir == Path("/home/user/.hud/envs")


class TestExtractDigestFromImage:
    """Test extracting digest from Docker image reference."""

    def test_extract_from_full_digest(self):
        """Test extracting from full digest reference."""
        image = "myimage:tag@sha256:abc123def456789"
        digest = extract_digest_from_image(image)
        assert digest == "abc123def456"

    def test_extract_from_digest_only(self):
        """Test extracting from digest-only format."""
        image = "sha256:deadbeef1234567890"
        digest = extract_digest_from_image(image)
        assert digest == "deadbeef1234"

    def test_extract_from_tag(self):
        """Test extracting from tagged image."""
        image = "myimage:v1.2.3"
        digest = extract_digest_from_image(image)
        assert digest == "v1.2.3"

    def test_extract_from_long_tag(self):
        """Test extracting from long tag (truncated)."""
        image = "myimage:superlongtagname123456789"
        digest = extract_digest_from_image(image)
        assert digest == "superlongtag"  # Max 12 chars

    def test_extract_no_tag(self):
        """Test extracting from image without tag."""
        image = "myimage"
        digest = extract_digest_from_image(image)
        assert digest == "latest"

    def test_extract_with_registry(self):
        """Test extracting from image with registry."""
        image = "docker.io/library/ubuntu:20.04"
        digest = extract_digest_from_image(image)
        assert digest == "20.04"

    def test_extract_with_port(self):
        """Test extracting from image with port."""
        image = "localhost:5000/myimage"
        digest = extract_digest_from_image(image)
        assert digest == "latest"


class TestExtractNameAndTag:
    """Test extracting name and tag from Docker image reference."""

    def test_extract_simple(self):
        """Test extracting from simple image reference."""
        name, tag = extract_name_and_tag("myorg/myapp:v1.0")
        assert name == "myorg/myapp"
        assert tag == "v1.0"

    def test_extract_with_digest(self):
        """Test extracting from reference with digest."""
        name, tag = extract_name_and_tag("docker.io/hudpython/test:latest@sha256:abc123")
        assert name == "hudpython/test"
        assert tag == "latest"

    def test_extract_no_tag(self):
        """Test extracting from reference without tag."""
        name, tag = extract_name_and_tag("myorg/myapp")
        assert name == "myorg/myapp"
        assert tag == "latest"

    def test_extract_with_docker_registry(self):
        """Test extracting from reference with docker.io prefix."""
        name, tag = extract_name_and_tag("docker.io/library/ubuntu:20.04")
        assert name == "library/ubuntu"
        assert tag == "20.04"

    def test_extract_with_other_registry(self):
        """Test extracting from reference with custom registry."""
        name, tag = extract_name_and_tag("gcr.io/myproject/myapp:v2")
        assert name == "gcr.io/myproject/myapp"
        assert tag == "v2"

    def test_extract_single_name(self):
        """Test extracting from single name without org."""
        name, tag = extract_name_and_tag("ubuntu")
        assert name == "ubuntu"
        assert tag == "latest"


class TestSaveToRegistry:
    """Test saving to local registry."""

    @mock.patch("hud.cli.utils.registry.HUDConsole")
    def test_save_success(self, mock_hud_console_class, tmp_path):
        """Test successful save to registry."""
        mock_hud_console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        # Mock home directory
        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            lock_data = {"image": "test:latest@sha256:abc123", "tools": ["tool1", "tool2"]}

            result = save_to_registry(lock_data, "test:latest@sha256:abc123def456789")

            assert result is not None
            assert result.exists()
            assert result.name == "hud.lock.yaml"

            # Verify content
            with open(result) as f:
                saved_data = yaml.safe_load(f)
            assert saved_data == lock_data

            # Verify directory structure
            assert result.parent.name == "abc123def456"

            mock_hud_console.success.assert_called_once()

    @mock.patch("hud.cli.utils.registry.HUDConsole")
    def test_save_verbose(self, mock_hud_console_class, tmp_path):
        """Test save with verbose output."""
        mock_hud_console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            lock_data = {"image": "test:v1"}

            result = save_to_registry(lock_data, "test:v1", verbose=True)

            assert result is not None
            # Should show verbose info
            assert mock_hud_console.info.call_count >= 1

    @mock.patch("hud.cli.utils.registry.HUDConsole")
    def test_save_failure(self, mock_hud_console_class):
        """Test handling save failure."""
        mock_hud_console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        # Mock file operations to fail
        with (
            mock.patch("builtins.open", side_effect=OSError("Permission denied")),
            mock.patch("pathlib.Path.home", return_value=Path("/tmp")),
        ):
            lock_data = {"image": "test:latest"}

            result = save_to_registry(lock_data, "test:latest", verbose=True)

            assert result is None
            mock_hud_console.warning.assert_called_once()


class TestLoadFromRegistry:
    """Test loading from local registry."""

    def test_load_success(self, tmp_path):
        """Test successful load from registry."""
        # Create test registry structure
        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            registry_dir = get_registry_dir()
            digest_dir = registry_dir / "abc123"
            digest_dir.mkdir(parents=True)

            lock_data = {"image": "test:latest", "version": "1.0"}
            lock_file = digest_dir / "hud.lock.yaml"
            lock_file.write_text(yaml.dump(lock_data))

            # Load it back
            loaded = load_from_registry("abc123")
            assert loaded == lock_data

    def test_load_not_found(self, tmp_path):
        """Test loading non-existent entry."""
        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            loaded = load_from_registry("nonexistent")
            assert loaded is None

    def test_load_corrupted(self, tmp_path):
        """Test loading corrupted lock file."""
        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            registry_dir = get_registry_dir()
            digest_dir = registry_dir / "bad"
            digest_dir.mkdir(parents=True)

            lock_file = digest_dir / "hud.lock.yaml"
            lock_file.write_text("invalid: yaml: content:")

            loaded = load_from_registry("bad")
            assert loaded is None


class TestListRegistryEntries:
    """Test listing registry entries."""

    def test_list_empty(self, tmp_path):
        """Test listing empty registry."""
        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            entries = list_registry_entries()
            assert entries == []

    def test_list_entries(self, tmp_path):
        """Test listing multiple entries."""
        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            registry_dir = get_registry_dir()

            # Create several entries
            for digest in ["abc123", "def456", "ghi789"]:
                digest_dir = registry_dir / digest
                digest_dir.mkdir(parents=True)
                lock_file = digest_dir / "hud.lock.yaml"
                lock_file.write_text(f"image: test:{digest}")

            # Create a directory without lock file (should be ignored)
            (registry_dir / "nolockfile").mkdir(parents=True)

            # Create a file in registry dir (should be ignored)
            (registry_dir / "README.txt").write_text("info")

            entries = list_registry_entries()

            assert len(entries) == 3
            digests = [entry[0] for entry in entries]
            assert set(digests) == {"abc123", "def456", "ghi789"}

            # Verify all paths are lock files
            for _, lock_path in entries:
                assert lock_path.name == "hud.lock.yaml"
                assert lock_path.exists()

    def test_list_no_registry_dir(self, tmp_path):
        """Test listing when registry directory doesn't exist."""
        with mock.patch("pathlib.Path.home", return_value=tmp_path / "nonexistent"):
            entries = list_registry_entries()
            assert entries == []
