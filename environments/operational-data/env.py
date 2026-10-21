"""Team operations over export-backed Slack, Jira and calendar tools."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import aclosing
from pathlib import PurePosixPath
from typing import Annotated

from hud.environment import DataFileRef, Environment, GradingArg, PromptArg
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from pydantic.fields import FieldInfo

from data_sources.base import DataSource
from data_sources.sources import Calendar, Jira, Slack
from workflow import (
    Attachments,
    BundledAttachments,
    Criterion,
    Deliverables,
    ScenarioDate,
    Workflow,
)

env = Environment(name="operational-data")
workflow = Workflow(env)
tool_server = workflow.tool_server


class BundledInputs(BaseModel):
    """Bundled sample paths, grouped by source independently of uploaded file ids."""

    model_config = ConfigDict(extra="forbid")

    slack: list[str] = Field(default_factory=list)
    jira: list[str] = Field(default_factory=list)
    calendar: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)

    @field_validator("slack", "jira", "calendar", "attachments")
    @classmethod
    def source_paths(cls, paths: list[str], info: ValidationInfo) -> list[str]:
        for path in paths:
            relative = PurePosixPath(path)
            if (
                len(relative.parts) < 2
                or relative.parts[0] != info.field_name
                or ".." in relative.parts
                or "\\" in path
            ):
                raise ValueError(
                    f"bundled {info.field_name} paths must stay in samples/{info.field_name}/"
                )
        return paths


def exports_of(source: type[DataSource]) -> FieldInfo:
    """Field metadata that makes an argument the upload slot of one data source."""
    return Field(
        title=source.title,
        description=source.summary,
        json_schema_extra={"x-hud-hint": "data-files", "x-hud-data-source": source.id},
    )


Exports = Sequence[DataFileRef]


@env.template(
    id="assist",
    description=(
        "Work a request against tools rebuilt from uploaded Slack, Jira and calendar exports"
        " and a workspace of attached files; graded by weighted criteria"
    ),
)
async def assist(
    prompt: PromptArg,
    *,
    slack: Annotated[Exports, exports_of(Slack)] = (),
    jira: Annotated[Exports, exports_of(Jira)] = (),
    calendar: Annotated[Exports, exports_of(Calendar)] = (),
    attachments: Attachments = (),
    bundled: BundledInputs | None = None,
    deliverables: Deliverables = (),
    criteria: GradingArg[Criterion],
    as_of: ScenarioDate = None,
    hud_api_key: str | None = None,
):
    """Work an operational request using uploaded or bundled source exports."""
    samples = BundledInputs.model_validate(bundled or {})
    bundled_sources = {
        source: paths
        for source, paths in (
            (Slack, samples.slack),
            (Jira, samples.jira),
            (Calendar, samples.calendar),
        )
        if paths
    }
    async with aclosing(
        workflow.run(
            prompt,
            uploads={Slack: slack, Jira: jira, Calendar: calendar},
            bundled_sources=bundled_sources,
            attachments=attachments,
            bundled_attachments=samples.attachments,
            deliverables=deliverables,
            criteria=criteria,
            as_of=as_of,
        )
    ) as session:
        answer = yield await anext(session)
        yield await session.asend(answer)


@env.template(
    id="review",
    description="Review supplier terms, invoices, credits and payments and save a reconciliation.",
)
async def review(
    prompt: PromptArg,
    *,
    contracts: Annotated[
        Sequence[DataFileRef],
        Field(
            description="Signed purchasing agreements and amendments (PDF or Word).",
            json_schema_extra={"x-hud-hint": "data-files"},
        ),
    ] = (),
    invoices: Annotated[
        Sequence[DataFileRef],
        Field(
            description="Invoice registers or invoice documents.",
            json_schema_extra={"x-hud-hint": "data-files"},
        ),
    ] = (),
    payments: Annotated[
        Sequence[DataFileRef],
        Field(
            description="Bank payment and reversal records.",
            json_schema_extra={"x-hud-hint": "data-files"},
        ),
    ] = (),
    credits: Annotated[
        Sequence[DataFileRef],
        Field(
            description="Supplier credit notes and their approval records.",
            json_schema_extra={"x-hud-hint": "data-files"},
        ),
    ] = (),
    bundled: BundledAttachments | None = None,
    deliverables: Deliverables = (),
    criteria: GradingArg[Criterion],
    as_of: ScenarioDate = None,
    hud_api_key: str | None = None,
):
    samples = BundledAttachments.model_validate(bundled or {})
    attachments = [ref for group in (contracts, invoices, payments, credits) for ref in group]
    async with aclosing(
        workflow.run(
            prompt,
            attachments=attachments,
            bundled_attachments=samples.attachments,
            deliverables=deliverables,
            criteria=criteria,
            as_of=as_of,
        )
    ) as session:
        answer = yield await anext(session)
        yield await session.asend(answer)


@env.template(
    id="plan",
    description=(
        "Build an executable dispatch plan from inventory, policy, briefing and email evidence."
    ),
)
async def plan(
    prompt: PromptArg,
    *,
    inventory: Annotated[
        Sequence[DataFileRef],
        Field(
            description="Order, inventory, reservation and event data (for example SQLite).",
            json_schema_extra={"x-hud-hint": "data-files"},
        ),
    ] = (),
    policy: Annotated[
        Sequence[DataFileRef],
        Field(
            description="Current dispatch, capacity and kit assembly policies.",
            json_schema_extra={"x-hud-hint": "data-files"},
        ),
    ] = (),
    briefing: Annotated[
        Sequence[DataFileRef],
        Field(
            description="Launch briefings and customer context.",
            json_schema_extra={"x-hud-hint": "data-files"},
        ),
    ] = (),
    emails: Annotated[
        Sequence[DataFileRef],
        Field(
            description="Sent approvals and draft proposals with their original email headers.",
            json_schema_extra={"x-hud-hint": "data-files"},
        ),
    ] = (),
    bundled: BundledAttachments | None = None,
    deliverables: Deliverables = (),
    criteria: GradingArg[Criterion],
    as_of: ScenarioDate = None,
    hud_api_key: str | None = None,
):
    samples = BundledAttachments.model_validate(bundled or {})
    attachments = [ref for group in (inventory, policy, briefing, emails) for ref in group]
    async with aclosing(
        workflow.run(
            prompt,
            attachments=attachments,
            bundled_attachments=samples.attachments,
            deliverables=deliverables,
            criteria=criteria,
            as_of=as_of,
        )
    ) as session:
        answer = yield await anext(session)
        yield await session.asend(answer)
