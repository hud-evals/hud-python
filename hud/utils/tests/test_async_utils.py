"""Tests for async utilities."""

from __future__ import annotations

import asyncio
import logging
import threading
from unittest.mock import patch

import pytest

from hud.utils.async_utils import fire_and_forget


class TestFireAndForget:
    """Test fire_and_forget function."""

    @pytest.mark.asyncio
    async def test_fire_and_forget_with_running_loop(self, caplog):
        """Test fire_and_forget when event loop is already running."""
        # Create a simple coroutine that sets a flag
        flag = []

        async def test_coro():
            flag.append(True)

        # Call fire_and_forget in async context
        fire_and_forget(test_coro(), description="test task")

        # Give it a moment to execute
        await asyncio.sleep(0.1)

        # Check that the coroutine was executed
        assert flag == [True]

    @pytest.mark.asyncio
    async def test_fire_and_forget_with_exception(self, caplog):
        """Test fire_and_forget handles exceptions gracefully."""

        async def failing_coro():
            raise ValueError("Test exception")

        # This should not raise
        fire_and_forget(failing_coro(), description="failing task")

        # Give it a moment to execute
        await asyncio.sleep(0.1)

        # The exception should be handled silently

    def test_fire_and_forget_no_event_loop(self):
        """Test fire_and_forget when no event loop is running."""
        # This test runs in sync context
        flag = threading.Event()

        async def test_coro():
            flag.set()

        # Call fire_and_forget in sync context
        fire_and_forget(test_coro(), description="sync test")

        # Wait for the thread to complete
        assert flag.wait(timeout=2.0), "Coroutine did not execute in thread"

    def test_fire_and_forget_thread_exception(self, caplog):
        """Test fire_and_forget handles thread exceptions."""

        async def failing_coro():
            raise ValueError("Thread exception")

        with caplog.at_level(logging.DEBUG, logger="hud.utils.async_utils"):
            fire_and_forget(failing_coro(), description="thread fail")

            # Give thread time to execute
            import time

            time.sleep(0.5)

            # Check that error was logged with correct format
            assert any(
                "Error in threaded thread fail:" in record.message
                and "Thread exception" in record.message
                for record in caplog.records
            )

    def test_fire_and_forget_interpreter_shutdown(self, caplog):
        """Test fire_and_forget handles interpreter shutdown gracefully."""

        async def test_coro():
            pass

        # Mock the scenario where we get interpreter shutdown error
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("no running event loop")

            with patch("threading.Thread") as mock_thread:
                mock_thread.side_effect = RuntimeError(
                    "cannot schedule new futures after interpreter shutdown"
                )

                with caplog.at_level(logging.DEBUG):
                    # This should not raise or log
                    fire_and_forget(test_coro(), description="shutdown test")

                    # No error should be logged for interpreter shutdown
                    assert not any(
                        "Could not shutdown test" in record.message for record in caplog.records
                    )

    def test_fire_and_forget_other_thread_error(self, caplog):
        """Test fire_and_forget logs non-shutdown thread errors."""

        async def test_coro():
            pass

        # Mock the scenario where we get a different error
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("no running event loop")

            with patch("threading.Thread") as mock_thread:
                mock_thread.side_effect = RuntimeError("Some other error")

                with caplog.at_level(logging.DEBUG, logger="hud.utils.async_utils"):
                    fire_and_forget(test_coro(), description="error test")

                    # This error should be logged with correct format
                    assert any(
                        "Could not error test - no event loop available:" in record.message
                        for record in caplog.records
                    )

    @pytest.mark.asyncio
    async def test_fire_and_forget_cancelled_task(self):
        """Test fire_and_forget handles cancelled tasks."""

        cancel_event = asyncio.Event()

        async def long_running_coro():
            await cancel_event.wait()

        # Get the current loop
        loop = asyncio.get_running_loop()

        # Patch create_task to capture the task
        created_task = None
        original_create_task = loop.create_task

        def mock_create_task(coro):
            nonlocal created_task
            created_task = original_create_task(coro)
            return created_task

        with patch.object(loop, "create_task", side_effect=mock_create_task):
            fire_and_forget(long_running_coro(), description="cancel test")

        # Give it a moment to start
        await asyncio.sleep(0.01)

        # Cancel the task
        assert created_task is not None
        created_task.cancel()

        # This should not raise any exceptions
        await asyncio.sleep(0.01)
