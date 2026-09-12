"""Tests for hud.__init__ module."""

from __future__ import annotations


class TestHudInit:
    """Tests for the hud package initialization."""

    def test_version_import_success(self):
        """Test that version is imported successfully."""
        import hud

        # Version should be available
        assert hasattr(hud, "__version__")
        assert isinstance(hud.__version__, str)

    def test_all_exports_available(self):
        """Test that all exported functions are available."""
        import hud

        expected_exports = [
            "Chat",
            "DockerRuntime",
            "Environment",
            "Grade",
            "Job",
            "HUDRuntime",
            "HostedRuntime",
            "Run",
            "Runtime",
            "RuntimeConfig",
            "RuntimeGPU",
            "RuntimeLimits",
            "RuntimeMount",
            "RuntimeResources",
            "RuntimeTPU",
            "LocalRuntime",
            "SubprocessRuntime",
            "SyncPlan",
            "Task",
            "Taskset",
            "connect",
            "instrument",
        ]

        for export in expected_exports:
            assert hasattr(hud, export), f"Missing export: {export}"
