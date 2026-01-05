"""Tests for AppendOnlyContext."""

import pytest

from hud.multi_agent.context import AppendOnlyContext
from hud.multi_agent.schemas import ContextEntryType


class TestAppendOnlyContext:
    """Test AppendOnlyContext functionality."""

    def test_init(self):
        """Test context initialization."""
        context = AppendOnlyContext(max_tokens=100_000)
        assert len(context) == 0
        assert context.token_count == 0
        assert context.frozen_prefix_length == 0

    def test_append_messages(self):
        """Test appending different message types."""
        context = AppendOnlyContext()

        # Add system message
        entry = context.append_system("You are a helpful assistant.")
        assert entry.type == ContextEntryType.SYSTEM
        assert len(context) == 1

        # Add user message
        entry = context.append_user("Hello!")
        assert entry.type == ContextEntryType.USER
        assert len(context) == 2

        # Add assistant message
        entry = context.append_assistant("Hi there!")
        assert entry.type == ContextEntryType.ASSISTANT
        assert len(context) == 3

    def test_freeze_prefix(self):
        """Test prefix freezing for KV cache optimization."""
        context = AppendOnlyContext()

        context.append_system("System prompt")
        context.append_user("Initial context")
        context.freeze_prefix()

        assert context.frozen_prefix_length == 2

        # Add more messages
        context.append_assistant("Response")
        context.append_user("Follow-up")

        # Frozen prefix should not change
        assert context.frozen_prefix_length == 2
        assert len(context.get_frozen_prefix()) == 2
        assert len(context.get_unfrozen_entries()) == 2

    def test_tool_calls(self):
        """Test tool call and result logging."""
        context = AppendOnlyContext()

        # Add tool call
        call_entry = context.append_tool_call(
            tool_name="search",
            tool_args={"query": "test"},
            agent_id="main",
        )
        assert call_entry.type == ContextEntryType.TOOL_CALL
        assert call_entry.tool_name == "search"
        assert call_entry.tool_args == {"query": "test"}

        # Add tool result
        result_entry = context.append_tool_result(
            content="Search results here",
            tool_name="search",
        )
        assert result_entry.type == ContextEntryType.TOOL_RESULT

    def test_file_content(self):
        """Test file content and reference handling."""
        context = AppendOnlyContext()

        # Add file content
        content_entry = context.append_file_content(
            content="def hello(): pass",
            path="/src/main.py",
            start_line=1,
            end_line=5,
        )
        assert content_entry.type == ContextEntryType.FILE_CONTENT
        assert content_entry.path == "/src/main.py"

        # Add file reference (compacted)
        ref_entry = context.append_file_ref(
            path="/src/main.py",
            start_line=1,
            end_line=5,
        )
        assert ref_entry.type == ContextEntryType.FILE_REF
        assert ref_entry.compacted is True

    def test_token_counting(self):
        """Test token count estimation."""
        context = AppendOnlyContext()

        context.append_user("Hello world!")  # ~3 tokens
        assert context.token_count > 0

        initial_count = context.token_count
        context.append_assistant("This is a longer response with more tokens.")
        assert context.token_count > initial_count

    def test_should_compact(self):
        """Test compaction threshold detection."""
        context = AppendOnlyContext(max_tokens=100)

        # Add enough content to trigger compaction threshold (80%)
        context.append_user("x" * 400)  # ~100 tokens

        assert context.should_compact() is True

    def test_render(self):
        """Test context rendering."""
        context = AppendOnlyContext()

        context.append_system("System prompt")
        context.append_user("Hello")
        context.append_assistant("Hi!")

        rendered = context.render()
        assert "System prompt" in rendered
        assert "Hello" in rendered
        assert "Hi!" in rendered

    def test_render_for_agent(self):
        """Test rendering for LLM API format."""
        context = AppendOnlyContext()

        context.append_system("System")
        context.append_user("User message")
        context.append_assistant("Assistant response")

        messages = context.render_for_agent()

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_snapshot_restore(self):
        """Test snapshot creation and restoration."""
        context = AppendOnlyContext()

        context.append_system("System")
        context.append_user("User")
        context.freeze_prefix()
        context.append_assistant("Assistant")

        # Create snapshot
        snapshot = context.snapshot()

        # Restore to new context
        restored = AppendOnlyContext.from_snapshot(snapshot)

        assert len(restored) == len(context)
        assert restored.frozen_prefix_length == context.frozen_prefix_length
        assert restored.token_count == context.token_count

    def test_replace_entry(self):
        """Test entry replacement for compaction."""
        context = AppendOnlyContext()

        context.append_system("System")
        context.freeze_prefix()

        entry = context.append_user("Long content that will be compacted")
        original_id = entry.id

        # Create compacted version
        from hud.multi_agent.schemas import ContextEntry

        compacted = ContextEntry(
            id=original_id,
            type=ContextEntryType.USER,
            content="[Compacted]",
            compacted=True,
        )

        # Replace (should work for unfrozen entries)
        result = context.replace_entry(original_id, compacted)
        assert result is True

        # Try to replace frozen entry (should fail)
        frozen_entry = context.get_frozen_prefix()[0]
        result = context.replace_entry(frozen_entry.id, compacted)
        assert result is False


class TestContextEntry:
    """Test ContextEntry functionality."""

    def test_render_file_ref(self):
        """Test file reference rendering."""
        from hud.multi_agent.schemas import ContextEntry

        entry = ContextEntry(
            id="test",
            type=ContextEntryType.FILE_REF,
            content="",
            path="/src/main.py",
            start_line=10,
            end_line=20,
        )

        rendered = entry.render()
        assert "/src/main.py" in rendered
        assert "10-20" in rendered

    def test_render_tool_call(self):
        """Test tool call rendering."""
        from hud.multi_agent.schemas import ContextEntry

        entry = ContextEntry(
            id="test",
            type=ContextEntryType.TOOL_CALL,
            content="",
            tool_name="search",
            tool_args={"query": "test"},
        )

        rendered = entry.render()
        assert "search" in rendered

