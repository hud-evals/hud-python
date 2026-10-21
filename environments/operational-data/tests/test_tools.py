"""The tool surface the agent sees: listing, JSON arguments, errors, and the action log."""

from collections.abc import AsyncIterator

import pytest
from exports import load
from fastmcp import Client, FastMCP
from fastmcp.client.transports import FastMCPTransport

from data_sources.base import ActionLog, DataSource, TaskContext
from data_sources.provider import MountedSources, TaskInProgress
from data_sources.sources import SOURCES, Slack

Agent = Client[FastMCPTransport]


@pytest.fixture
def sources(context: TaskContext) -> list[DataSource]:
    return [load(source, context) for source in SOURCES.values()]


@pytest.fixture
async def agent(sources: list[DataSource]) -> AsyncIterator[Agent]:
    mounted = MountedSources()
    mounted.mount(sources)
    async with Client(FastMCP("samples", providers=[mounted])) as client:
        yield client


async def test_the_agent_sees_every_loaded_tool(agent: Agent, sources: list[DataSource]):
    listed = {tool.name for tool in await agent.list_tools()}
    assert listed == {tool.name for source in sources for tool in source.tools()}


async def test_tools_take_json_arguments_and_return_pages(agent: Agent):
    result = await agent.call_tool("jira_search_issues", {"jql": "project = CLOSE", "limit": 2})
    page = result.structured_content
    assert page is not None
    assert page["total"] > 2
    assert len(page["items"]) == 2


async def test_unknown_records_come_back_as_tool_errors(agent: Agent):
    result = await agent.call_tool("jira_get_issue", {"key": "NOPE-1"}, raise_on_error=False)
    assert result.is_error


async def test_write_tools_record_what_the_agent_changed(agent: Agent, log: ActionLog):
    await agent.call_tool(
        "slack_post_message", {"channel": "finance-close", "text": "Flash is out."}
    )
    await agent.call_tool("jira_add_comment", {"key": "CLOSE-17", "body": "Flash sent to Dana."})
    assert [action.tool for action in log.actions] == ["slack_post_message", "jira_add_comment"]
    assert log.actions[0].arguments == {"channel": "finance-close", "text": "Flash is out."}


async def test_rejected_writes_change_nothing(agent: Agent, log: ActionLog):
    result = await agent.call_tool(
        "jira_transition_issue", {"key": "CLOSE-14", "status": "Shipped"}, raise_on_error=False
    )
    assert result.is_error
    assert log.actions == []
    issue = await agent.call_tool("jira_get_issue", {"key": "CLOSE-14"})
    assert issue.structured_content is not None
    assert issue.structured_content["status"] == "To Do"


def test_two_sources_cannot_define_the_same_tool(context: TaskContext):
    with pytest.raises(ValueError, match="slack_"):
        MountedSources().mount([load(Slack, context), load(Slack, context)])


async def test_a_task_holds_the_tools_until_it_ends(context: TaskContext):
    mounted = MountedSources()
    with mounted.occupy():
        mounted.mount([load(Slack, context)])
        with pytest.raises(TaskInProgress), mounted.occupy():
            pass
    async with Client(FastMCP("empty", providers=[mounted])) as client:
        assert await client.list_tools() == []
    with mounted.occupy():
        pass
