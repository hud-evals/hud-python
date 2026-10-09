from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from hud.environment import (
    DataFileArg,
    DataFileRef,
    DataFilesArg,
    Environment,
    GradingArg,
    PromptArg,
)
from hud.eval import Run

from .conftest import served


class _Attachment(DataFileRef):
    expand: bool = False


class _Fixture(DataFileRef):
    channel: Literal["mail", "calendar"]


class _Criterion(BaseModel):
    requirement: str
    weight: float = 1.0
    guidance: str | None = None


def test_argument_types_publish_editor_hints_and_model_schemas() -> None:
    env = Environment("typed-args")

    @env.template()
    async def task(
        prompt: PromptArg,
        attachment: DataFileArg[_Attachment],
        fixtures: DataFilesArg[_Fixture],
        criteria: GradingArg[_Criterion],
    ):
        yield prompt
        yield 1.0

    schema = task.manifest_entry()["args"]
    properties = schema["properties"]

    assert properties["prompt"]["x-hud-hint"] == "prompt"
    assert properties["attachment"]["x-hud-hint"] == "data-file"
    assert properties["fixtures"]["x-hud-hint"] == "data-files"
    assert properties["criteria"]["x-hud-hint"] == "grading"
    assert properties["fixtures"]["items"]["$ref"] == "#/$defs/_Fixture"
    assert properties["criteria"]["items"]["$ref"] == "#/$defs/_Criterion"
    assert "channel" in schema["$defs"]["_Fixture"]["properties"]
    assert "guidance" in schema["$defs"]["_Criterion"]["properties"]


async def test_typed_argument_values_reach_task_as_models() -> None:
    env = Environment("typed-args")

    @env.template()
    async def task(
        attachments: DataFilesArg[_Attachment],
        criteria: GradingArg[_Criterion],
    ):
        assert isinstance(attachments[0], _Attachment)
        assert isinstance(criteria[0], _Criterion)
        yield f"{attachments[0].path}:{criteria[0].requirement}"
        yield 1.0

    async with (
        served(env) as client,
        Run(
            client,
            "task",
            {
                "attachments": [{"file_id": "file-1", "path": "brief.pdf"}],
                "criteria": [{"requirement": "Answer the question", "weight": 2.0}],
            },
        ) as run,
    ):
        assert run.prompt_text == "brief.pdf:Answer the question"
        run.trace.content = "done"
