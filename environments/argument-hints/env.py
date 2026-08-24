"""Minimal environment showing what argument hints do.

Three of this template's arguments carry an ``x-hud-hint`` in their JSON
Schema, which is all it takes for the console to edit them with purpose-built
controls instead of JSON boxes:

- ``prompt`` -> a text box
- ``attachments`` -> a picker over the team's data files, with upload
- ``criteria`` -> a table of grading rows

The environment then does the ordinary work: it pulls the referenced data files
into the solver's workspace, hands the agent the prompt, and passes the criteria
straight to ``LLMJudgeGrader``, whose input shape they are.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

import httpx
from hud.environment import Environment
from hud.graders import EvaluationResult, LLMJudgeGrader, combine
from hud.settings import settings
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

env = Environment(name="argument-hints")

if workspace_dir := os.environ.get("WORKSPACE_DIR"):
    WORKSPACE_ROOT = Path(workspace_dir)
    _LOCAL = False
else:
    WORKSPACE_ROOT = Path(tempfile.gettempdir()) / "hud-argument-hints" / str(os.getpid())
    _LOCAL = True
FILES_DIRNAME = "files"

_TIMEOUT = httpx.Timeout(30.0, read=300.0)
_CHUNK = 1 << 20

# The workspace serves the agent a shell and streams its file diffs.
env.workspace(WORKSPACE_ROOT)


@env.shutdown
async def _cleanup() -> None:
    if _LOCAL:
        shutil.rmtree(WORKSPACE_ROOT, ignore_errors=True)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataFileRef(StrictModel):
    """One uploaded data file to stage into the workspace."""

    file_id: str = Field(description="Id of the uploaded data file.")
    path: str | None = Field(
        default=None,
        description="Destination under the files directory; defaults to the uploaded name.",
    )


class Criterion(StrictModel):
    """One criterion, exactly as ``LLMJudgeGrader`` takes it."""

    requirement: str = Field(description="What the answer has to do.")
    weight: float = Field(
        default=1.0,
        description="Share of the score; negative marks an error to penalize.",
    )


Prompt = Annotated[str, Field(json_schema_extra={"x-hud-hint": "prompt"})]
Attachments = Annotated[list[DataFileRef], Field(json_schema_extra={"x-hud-hint": "data-files"})]
Criteria = Annotated[list[Criterion], Field(json_schema_extra={"x-hud-hint": "grading"})]

# Model-typed arguments arrive as plain JSON, so the models are applied here.
_ATTACHMENTS = TypeAdapter(list[DataFileRef])
_CRITERIA = TypeAdapter(list[Criterion])


class DataFileError(RuntimeError):
    """A referenced data file could not be staged."""


def _destination(root: Path, relative: str) -> Path:
    """Resolve a files-directory-relative path, refusing anything that escapes it."""
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
        raise DataFileError(f"unsafe data file path: {relative!r}")
    return root / candidate


async def _stage(
    refs: list[DataFileRef],
    root: Path,
    hud_api_key: str | None = None,
) -> list[dict[str, str]]:
    """Pull each referenced file into ``root``; returns the start-frame declarations."""
    if not refs:
        return []
    api_key = (hud_api_key or settings.api_key or "").strip()
    if not api_key:
        raise DataFileError("HUD_API_KEY is unset; the environment cannot read data files")

    base = settings.hud_api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    declared: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:

        async def read_json(path: str) -> dict[str, Any]:
            response = await client.get(f"{base}{path}", headers=headers)
            if response.is_error:
                raise DataFileError(f"GET {path} failed: {response.status_code}")
            return response.json()

        for ref in refs:
            metadata = await read_json(f"/v2/data/{ref.file_id}")
            filename = metadata.get("filename")
            if not isinstance(filename, str) or not filename:
                raise DataFileError(f"data file {ref.file_id} has no filename")
            destination = _destination(root, ref.path or filename)
            destination.parent.mkdir(parents=True, exist_ok=True)

            url = (await read_json(f"/v2/data/{ref.file_id}/download")).get("url")
            if not isinstance(url, str) or not url:
                raise DataFileError(f"data file {ref.file_id} has no download url")
            # The presigned URL carries its own credentials; ours would be rejected.
            async with client.stream("GET", url) as response:
                if response.is_error:
                    raise DataFileError(f"downloading data file {ref.file_id} failed: {response.status_code}")
                with destination.open("wb") as out:
                    async for chunk in response.aiter_bytes(_CHUNK):
                        out.write(chunk)

            declared.append(
                {
                    "path": str(destination.relative_to(WORKSPACE_ROOT)),
                    "file_id": ref.file_id,
                }
            )
    return declared


async def _grade(answer: str, question: str, criteria: list[Criterion]) -> EvaluationResult:
    """Judge the answer against the criteria, which are the grader's own input."""
    return await combine(
        LLMJudgeGrader.grade(
            weight=1.0,
            answer=answer,
            criteria=[(item.requirement, item.weight) for item in criteria],
            question=question,
        )
    )


@env.template(
    id="review_files",
    description="Read the attached files, answer the prompt, and grade against criteria",
)
async def review_files(
    prompt: Prompt,
    attachments: Attachments,
    criteria: Criteria,
    hud_api_key: str | None = None,
):
    """Answer a prompt about uploaded files, graded by weighted criteria."""
    refs = _ATTACHMENTS.validate_python(attachments)
    rows = _CRITERIA.validate_python(criteria)

    files_dir = WORKSPACE_ROOT / FILES_DIRNAME
    # Clear prior tasks' leftovers so staging never writes through a stale symlink.
    shutil.rmtree(files_dir, ignore_errors=True)
    files_dir.mkdir(parents=True)
    declared = await _stage(refs, files_dir, hud_api_key)

    listing = "\n".join(f"- {entry['path']}" for entry in declared) or "- (none)"
    # A template that declares no `returns` is sent the agent's final text.
    answer = yield {
        "prompt": (
            f"You have a shell in {WORKSPACE_ROOT}. These files are staged for you:\n"
            f"{listing}\n\n"
            "Read what you need, then reply with your answer as your final message.\n\n"
            f"{prompt}"
        ),
        # Declaring the staged files lets the trace viewer read back contents
        # file tracking cannot carry, such as a PDF's bytes.
        "data_files": declared,
    }

    yield await _grade(str(answer or ""), prompt, rows)
