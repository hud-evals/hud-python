"""LangChain MCP Agent implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import mcp.types as types
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from mcp_use.adapters.langchain_adapter import LangChainAdapter

if TYPE_CHECKING:
    from langchain.schema.language_model import BaseLanguageModel
    from langchain_core.tools import BaseTool

from hud.agent import MCPAgent
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

logger = logging.getLogger(__name__)


class LangChainMCPAgent(MCPAgent):
    """
    LangChain agent that uses MCP servers for tool execution.

    This agent wraps any LangChain-compatible LLM and provides
    access to MCP tools through LangChain's tool-calling interface.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LangChain MCP agent.

        Args:
            llm: Any LangChain-compatible language model
            **kwargs: Additional arguments passed to BaseMCPAgent
        """
        super().__init__(**kwargs)

        self.llm = llm
        self.adapter = LangChainAdapter(disallowed_tools=self.disallowed_tools)
        self._langchain_tools: list[BaseTool] | None = None

        self.model_name = (
            "langchain-" + self.llm.model_name  # type: ignore
            if hasattr(self.llm, "model_name")
            else "unknown"
        )

    def _get_langchain_tools(self) -> list[BaseTool]:
        """Get or create LangChain tools from MCP tools."""
        if self._langchain_tools is not None:
            return self._langchain_tools

        # Create LangChain tools from MCP tools using the adapter
        self._langchain_tools = []

        # Convert available tools using the adapter; no server grouping
        langchain_tools = self.adapter._convert_tools(self._available_tools, "default")  # type: ignore[reportAttributeAccessIssue]
        self._langchain_tools.extend(langchain_tools)

        logger.info("Created %s LangChain tools from MCP tools", len(self._langchain_tools))
        return self._langchain_tools

    async def create_initial_messages(
        self, prompt: str, screenshot: str | None
    ) -> list[BaseMessage]:
        """Create initial messages for LangChain."""
        messages = []

        # Add system message
        system_prompt = self.get_system_prompt()
        messages.append(SystemMessage(content=system_prompt))

        # Add user message with prompt and optional screenshot
        if screenshot:
            # For multimodal models, include the image
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot}"}},
            ]
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=prompt))

        return messages

    async def get_model_response(self, messages: list[BaseMessage]) -> AgentResponse:
        """Get response from LangChain model including any tool calls."""
        # Get LangChain tools (created lazily)
        langchain_tools = self._get_langchain_tools()

        # Create a prompt template from current messages
        # Extract system message if present
        system_content = "You are a helpful assistant"
        non_system_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_content = str(msg.content)
            else:
                non_system_messages.append(msg)

        # Create prompt with placeholders
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_content),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create agent with tools
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=langchain_tools,
            prompt=prompt,
        )

        # Create executor
        executor = AgentExecutor(
            agent=agent,
            tools=langchain_tools,
            verbose=False,
        )

        # Format the last user message as input
        last_user_msg = None
        for msg in reversed(non_system_messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg
                break

        if not last_user_msg:
            return AgentResponse(content="No user message found", tool_calls=[], done=True)

        # Extract text from message content
        input_text = ""
        if isinstance(last_user_msg.content, str):
            input_text = last_user_msg.content
        elif isinstance(last_user_msg.content, list):
            # Extract text from multimodal content
            for item in last_user_msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    input_text = item.get("text", "")
                    break

        # Build chat history (exclude last user message and system)
        chat_history = []
        for _, msg in enumerate(non_system_messages[:-1]):
            if isinstance(msg, HumanMessage | AIMessage):
                chat_history.append(msg)

        # Execute the agent
        try:
            result = await executor.ainvoke(
                {
                    "input": input_text,
                    "chat_history": chat_history,
                }
            )

            # Process the result
            output = result.get("output", "")

            # Check if tools were called
            if result.get("intermediate_steps"):
                # Tools were called
                tool_calls = []
                for action, _ in result["intermediate_steps"]:
                    if hasattr(action, "tool") and hasattr(action, "tool_input"):
                        tool_calls.append(
                            MCPToolCall(
                                name=action.tool,
                                arguments=action.tool_input,
                            )
                        )

                return AgentResponse(content=output, tool_calls=tool_calls, done=False)
            else:
                # No tools called, just text response
                return AgentResponse(content=output, tool_calls=[], done=True)

        except Exception as e:
            logger.error("Agent execution failed: %s", e)
            return AgentResponse(content=f"Error: {e!s}", tool_calls=[], done=True)

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[BaseMessage]:
        """Format tool results into LangChain messages."""
        # Create an AI message with the tool calls and results
        messages = []

        # First add an AI message indicating tools were called
        tool_names = [tc.name for tc in tool_calls]
        ai_content = f"I'll use the following tools: {', '.join(tool_names)}"
        messages.append(AIMessage(content=ai_content))

        # Build result text from tool results
        text_parts = []
        latest_screenshot = None

        for tool_call, result in zip(tool_calls, tool_results, strict=False):
            if result.isError:
                error_text = "Tool execution failed"
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        error_text = content.text
                        break
                text_parts.append(f"Error - {tool_call.name}: {error_text}")
            else:
                # Process success content
                tool_output = []
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        tool_output.append(content.text)
                    elif isinstance(content, types.ImageContent):
                        latest_screenshot = content.data

                if tool_output:
                    text_parts.append(f"{tool_call.name}: " + " ".join(tool_output))

        result_text = "\n".join(text_parts) if text_parts else "No output from tools"

        # Then add a human message with the tool results
        if latest_screenshot:
            # Include screenshot in multimodal format
            content = [
                {"type": "text", "text": f"Tool results:\n{result_text}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{latest_screenshot}"},
                },
            ]
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=f"Tool results:\n{result_text}"))

        return messages
