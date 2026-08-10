"""Tests for hud/__init__.py module initialization."""

from __future__ import annotations


class TestInitModule:
    """Test the hud module initialization."""

    def test_version_exposed(self):
        """Test that the package exposes its version."""
        import hud

        assert hasattr(hud, "__version__")
        assert isinstance(hud.__version__, str)

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import hud

        expected = [
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
            "RuntimeResources",
            "RootfsProfile",
            "StorageConfig",
            "WorkspaceProfile",
            "LocalRuntime",
            "SubprocessRuntime",
            "SyncPlan",
            "Task",
            "Taskset",
            "Trace",
            "TrainingClient",
            "__version__",
            "connect",
            "instrument",
        ]

        assert set(hud.__all__) == set(expected)

        # Verify all exported items are actually available
        for item in hud.__all__:
            assert hasattr(hud, item)
