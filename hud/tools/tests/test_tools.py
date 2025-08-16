from __future__ import annotations

import sys

import pytest
from mcp.types import ImageContent, TextContent

from hud.tools.bash import BashTool
from hud.tools.computer.hud import HudComputerTool
from hud.tools.edit import EditTool

# from hud.tools.helper import register_instance_tool  # TODO: Function not found in codebase


@pytest.mark.asyncio
async def test_bash_tool_echo():
    tool = BashTool()

    # Monkey-patch the private _session methods so no subprocess is spawned
    from hud.tools.types import ContentResult

    class _FakeSession:
        async def run(self, cmd: str):
            return ContentResult(output=f"mocked: {cmd}")

        async def start(self):
            return None

    tool.session = _FakeSession()  # type: ignore[assignment]

    result = await tool(command="echo hello")
    assert len(result) > 0
    assert isinstance(result[0], TextContent)
    assert result[0].text == "mocked: echo hello"


@pytest.mark.asyncio
async def test_bash_tool_restart_and_no_command():
    from hud.tools.types import ToolError

    tool = BashTool()

    from hud.tools.types import ContentResult

    class _FakeSession:
        async def run(self, cmd: str):
            return ContentResult(output="ran")

        async def start(self):
            return None

        def stop(self):
            return None

    tool.session = _FakeSession()  # type: ignore[assignment]

    # Monkey-patch _BashSession.start to avoid launching a real shell
    async def _dummy_start(self):
        self._started = True
        from types import SimpleNamespace

        # minimal fake process attributes used later
        self._process = SimpleNamespace(returncode=None)

    import hud.tools.bash as bash_mod

    bash_mod._BashSession.start = _dummy_start  # type: ignore[assignment]

    # restart=True returns system message
    res = await tool(command="ignored", restart=True)
    # Check that we get content blocks with the restart message
    assert len(res) > 0
    text_blocks = [b for b in res if isinstance(b, TextContent)]
    assert any("restarted" in b.text for b in text_blocks)

    # Calling without command raises ToolError
    with pytest.raises(ToolError):
        await tool()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="EditTool uses Unix commands")
async def test_edit_tool_flow(tmp_path):
    file_path = tmp_path / "demo.txt"

    edit = EditTool()

    # create
    res = await edit(command="create", path=str(file_path), file_text="hello\nworld\n")
    # Check for success message in content blocks
    text_blocks = [b for b in res if isinstance(b, TextContent)]
    assert any("created" in b.text for b in text_blocks)

    # view
    res = await edit(command="view", path=str(file_path))
    # Check content blocks for file content
    text_blocks = [b for b in res if isinstance(b, TextContent)]
    combined_text = "".join(b.text for b in text_blocks)
    assert "hello" in combined_text

    # replace
    res = await edit(command="str_replace", path=str(file_path), old_str="world", new_str="earth")
    # Check for success message in content blocks
    text_blocks = [b for b in res if isinstance(b, TextContent)]
    combined_text = "".join(b.text for b in text_blocks)
    assert "has been edited" in combined_text

    # insert
    res = await edit(command="insert", path=str(file_path), insert_line=1, new_str="first line\n")
    assert res


@pytest.mark.asyncio
async def test_base_executor_simulation():
    from hud.tools.executors.base import BaseExecutor

    exec = BaseExecutor()
    res = await exec.execute("echo test")
    assert "SIMULATED" in (res.output or "")
    shot = await exec.screenshot()
    assert isinstance(shot, str) and len(shot) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="EditTool uses Unix commands")
async def test_edit_tool_view(tmp_path):
    # Create a temporary file
    p = tmp_path / "sample.txt"
    p.write_text("Sample content\n")

    tool = EditTool()
    result = await tool(command="view", path=str(p))
    # Check content blocks for file content
    text_blocks = [b for b in result if isinstance(b, TextContent)]
    combined_text = "".join(b.text for b in text_blocks)
    assert "Sample content" in combined_text


@pytest.mark.asyncio
async def test_computer_tool_screenshot():
    comp = HudComputerTool()
    blocks = await comp(action="screenshot")
    # Check that we got content blocks back
    assert blocks is not None
    assert len(blocks) > 0
    # Either ImageContent or TextContent is valid
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)
