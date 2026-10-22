"""The Larkspur Outdoor Co. sample exports in samples/, loaded the way a task loads uploads."""

from pathlib import Path

from data_sources.base import DataSource, TaskContext

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def sample_files(source_id: str) -> list[Path]:
    return sorted(path for path in (SAMPLES / source_id).iterdir() if path.is_file())


def load[S: DataSource](source: type[S], context: TaskContext) -> S:
    return source.load(sample_files(source.id), context)
