from datetime import date

import pytest

from data_sources.base import ActionLog, TaskContext


@pytest.fixture
def log() -> ActionLog:
    return ActionLog()


@pytest.fixture
def context(log: ActionLog) -> TaskContext:
    return TaskContext(today=date(2026, 3, 31), log=log)
