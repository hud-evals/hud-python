"""Tests for the progress tracking utilities."""

from __future__ import annotations

import pytest

from hud.utils.progress import StepProgressTracker


@pytest.fixture
def tracker():
    return StepProgressTracker(total_tasks=2, max_steps_per_task=10)


def test_invalid_inputs_init():
    with pytest.raises(ValueError, match="total_tasks must be positive"):
        StepProgressTracker(total_tasks=0, max_steps_per_task=10)

    with pytest.raises(ValueError, match="max_steps_per_task must be positive"):
        StepProgressTracker(total_tasks=5, max_steps_per_task=0)


def test_start_task(tracker):
    assert tracker.start_time is None
    assert tracker._tasks_started == 0

    tracker.start_task("task1")

    assert tracker.start_time is not None
    assert tracker._tasks_started == 1
    assert tracker._task_steps["task1"] == 0
    assert not tracker._finished_tasks["task1"]

    tracker.start_task("task2")
    assert tracker._tasks_started == 2
    assert tracker._task_steps["task2"] == 0
    assert not tracker._finished_tasks["task2"]


def test_increment_step(tracker):
    tracker.start_task("task1")
    assert tracker.current_total_steps == 0

    tracker.increment_step("task1")
    assert tracker._task_steps["task1"] == 1
    assert tracker.current_total_steps == 1

    tracker.increment_step("task1")
    tracker.increment_step("task1")
    assert tracker._task_steps["task1"] == 3
    assert tracker.current_total_steps == 3

    tracker.start_task("task2")
    tracker.increment_step("task2")
    assert tracker._task_steps["task2"] == 1
    assert tracker.current_total_steps == 4

    tracker.finish_task("task1")
    initial_steps = tracker.current_total_steps
    tracker.increment_step("task1")
    assert tracker.current_total_steps == initial_steps

    for _ in range(15):
        tracker.increment_step("task2")
    assert tracker._task_steps["task2"] <= tracker.max_steps_per_task


def test_finish_task(tracker):
    tracker.start_task("task1")
    tracker.start_task("task2")

    tracker.increment_step("task1")
    tracker.increment_step("task1")
    initial_steps = tracker._task_steps["task1"]

    tracker.finish_task("task1")

    assert tracker._finished_tasks["task1"]
    assert tracker._tasks_finished == 1
    assert tracker._task_steps["task1"] == tracker.max_steps_per_task
    assert tracker.current_total_steps > initial_steps

    current_steps = tracker.current_total_steps
    tracker.finish_task("task1")
    assert tracker._tasks_finished == 1
    assert tracker.current_total_steps == current_steps


def test_get_progress(tracker):
    steps, total, percentage = tracker.get_progress()
    assert steps == 0
    assert total == tracker.total_potential_steps
    assert percentage == 0.0

    tracker.start_task("task1")
    tracker.increment_step("task1")
    steps, total, percentage = tracker.get_progress()
    assert steps == 1
    assert total == tracker.total_potential_steps
    assert percentage == (1 / tracker.total_potential_steps) * 100

    tracker.finish_task("task1")
    steps, total, percentage = tracker.get_progress()
    assert steps == tracker.max_steps_per_task
    assert total == tracker.total_potential_steps
    assert percentage == (tracker.max_steps_per_task / tracker.total_potential_steps) * 100

    tracker.start_task("task2")
    tracker.finish_task("task2")
    steps, total, percentage = tracker.get_progress()
    assert steps == tracker.total_potential_steps
    assert percentage == 100.0


def test_get_stats_no_progress(tracker):
    from unittest.mock import patch

    rate, eta = tracker.get_stats()
    assert rate == 0.0
    assert eta is None

    with patch("time.monotonic", return_value=100.0):
        tracker.start_task("task1")

        rate, eta = tracker.get_stats()
        assert rate == 0.0
        assert eta is None


def test_get_stats_with_progress():
    from unittest.mock import patch

    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0

        tracker = StepProgressTracker(total_tasks=1, max_steps_per_task=10)
        tracker.start_task("task1")

        mock_time.return_value = 160.0
        for _ in range(5):
            tracker.increment_step("task1")

        rate, eta = tracker.get_stats()

        assert rate == pytest.approx(5.0)
        assert eta == pytest.approx(60.0)

        for _ in range(5):
            tracker.increment_step("task1")

        rate, eta = tracker.get_stats()
        assert rate == pytest.approx(10.0)
        assert eta == pytest.approx(0.0)


def test_is_finished(tracker):
    assert not tracker.is_finished()

    tracker.start_task("task1")
    tracker.finish_task("task1")
    assert not tracker.is_finished()

    tracker.start_task("task2")
    tracker.finish_task("task2")
    assert tracker.is_finished()


def test_display(tracker):
    from unittest.mock import patch

    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        tracker.start_task("task1")

        mock_time.return_value = 130.0
        tracker.increment_step("task1")
        tracker.increment_step("task1")

        display_str = tracker.display()

        assert "%" in display_str
        assert "2/20" in display_str
        assert "0:30" in display_str
        assert "steps/min" in display_str

        tracker.finish_task("task1")
        display_str = tracker.display()
        assert "10/20" in display_str

        tracker.start_task("task2")
        tracker.finish_task("task2")
        display_str = tracker.display()
        assert "100%" in display_str
        assert "20/20" in display_str


def test_complex_workflow():
    tracker = StepProgressTracker(total_tasks=5, max_steps_per_task=20)

    for i in range(5):
        tracker.start_task(f"task{i}")

    for _ in range(10):
        tracker.increment_step("task0")

    for _ in range(5):
        tracker.increment_step("task1")

    tracker.finish_task("task2")

    for _ in range(15):
        tracker.increment_step("task3")

    tracker.finish_task("task3")

    steps, total, percentage = tracker.get_progress()
    expected_steps = 10 + 5 + 20 + 20 + 0
    assert steps == expected_steps
    assert total == 5 * 20
    assert percentage == (expected_steps / total) * 100

    assert tracker._tasks_finished == 2
    assert not tracker.is_finished()

    tracker.finish_task("task0")
    tracker.finish_task("task1")
    tracker.finish_task("task4")

    assert tracker.is_finished()
    assert tracker.get_progress()[2] == 100.0


def test_display_eta_when_finished(tracker):
    from unittest.mock import patch

    """Test that ETA shows 0:00 when progress is finished."""

    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0

        # Start and complete all tasks
        tracker.start_task("task1")
        for _ in range(10):
            tracker.increment_step("task1")
        tracker.finish_task("task1")

        tracker.start_task("task2")
        for _ in range(10):
            tracker.increment_step("task2")
        tracker.finish_task("task2")

        # Some time has passed
        mock_time.return_value = 120.0

        display = tracker.display()

        # When finished, ETA should be 0:00 (not ??:??)
        assert tracker.is_finished()
        assert "0:00" in display
        assert "100%" in display
        assert "20/20" in display
