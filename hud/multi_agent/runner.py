"""MultiAgentRunner: Orchestration of multi-agent system.

This module ties everything together:
- Loads configuration from YAML
- Creates main agent and sub-agents
- Manages context, memory, and logging
- Handles the agent-as-tool pattern
- Implements CodeAct execution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.multi_agent.agent_tool import AgentToolRegistry, agent_as_tool
from hud.multi_agent.codeact import CodeActExecutor
from hud.multi_agent.compaction import ContextCompactor, ContextOffloader
from hud.multi_agent.config import AgentConfig, ConfigLoader, MultiAgentConfig, load_config
from hud.multi_agent.context import AppendOnlyContext
from hud.multi_agent.logger import StepLogger
from hud.multi_agent.memory import FilesystemMemory
from hud.multi_agent.schemas import GenericResult
from hud.multi_agent.sub_agent import SubAgent, SubAgentConfig

if TYPE_CHECKING:
    from hud.eval.context import EvalContext

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a multi-agent run."""

    success: bool
    reward: float = 0.0
    output: Any = None
    files: list[str] = field(default_factory=list)
    error: str | None = None
    duration_ms: float | None = None
    steps: int = 0
    logs_dir: str | None = None


class MultiAgentRunner:
    """Orchestrate a multi-agent system.

    This is the main entry point for running multi-agent tasks:
    1. Loads configuration from YAML
    2. Sets up context, memory, and logging
    3. Creates main agent with sub-agents as tools
    4. Runs the task and returns structured result

    Example:
        async with hud.eval(task, name="multi-agent") as ctx:
            runner = MultiAgentRunner(
                config_dir=Path("agents/"),
                ctx=ctx,
                workspace=Path("./workspace"),
            )

            result = await runner.run(
                task="Build a REST API with authentication",
                max_steps=50,
            )

            print(f"Reward: {result.reward}")
            print(f"Files: {result.files}")
    """

    def __init__(
        self,
        config_dir: Path | str | None = None,
        config: MultiAgentConfig | None = None,
        ctx: EvalContext | None = None,
        workspace: Path | str | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            config_dir: Path to YAML config directory or file
            config: Pre-loaded configuration (alternative to config_dir)
            ctx: EvalContext for evaluation
            workspace: Working directory for filesystem memory
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_dir is not None:
            self.config = load_config(config_dir)
        else:
            # Default config with single agent
            self.config = MultiAgentConfig()

        # Store context
        self.ctx = ctx

        # Setup workspace
        self.workspace = Path(workspace or self.config.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.memory = FilesystemMemory(self.workspace)
        self.context = AppendOnlyContext(max_tokens=self.config.max_context_tokens)
        self.offloader = ContextOffloader(
            self.memory, threshold=self.config.offload_threshold
        )
        self.compactor = ContextCompactor(
            self.memory, rot_threshold=self.config.max_context_tokens
        )

        # Logger
        self.logger = StepLogger(
            log_dir=Path(self.config.log_dir),
        )

        # CodeAct executor (initialized lazily)
        self._codeact: CodeActExecutor | None = None

        # Agent registry
        self.agent_registry = AgentToolRegistry()

        # Track state
        self._initialized = False
        self._start_time: datetime | None = None
        
        # Progress callback for UI updates
        self.progress_callback: Any = None

    async def initialize(self) -> None:
        """Initialize the runner and all components."""
        if self._initialized:
            return

        # Create sub-agents from config
        for name, agent_config in self.config.agents.items():
            if agent_config.type == "specialist":
                self._create_sub_agent(name, agent_config)

        self._initialized = True
        logger.info(
            f"MultiAgentRunner initialized with {len(self.config.agents)} agents"
        )

    def _create_sub_agent(self, name: str, config: AgentConfig) -> None:
        """Create a sub-agent from configuration.

        Args:
            name: Agent name
            config: Agent configuration
        """
        # Create SubAgent class dynamically with embedded config
        sub_config = SubAgentConfig(
            name=name,
            model=config.model,
            max_steps=config.max_steps,
            timeout=config.timeout,
            isolation=config.isolation,
            max_context_tokens=config.max_context_tokens,
            system_prompt=config.system_prompt,
            tools=config.tools,
            return_schema=config.returns.schema_name if config.returns else None,
            use_computer_beta=config.use_computer_beta,
        )

        # Create agent class with decorator
        return_schema = config.get_return_schema()

        # Create a closure to capture sub_config
        captured_config = sub_config

        @agent_as_tool(name=name, description=config.system_prompt[:100] if config.system_prompt else f"Sub-agent: {name}", returns=return_schema)
        class ConfiguredSubAgent(SubAgent):
            """Dynamically configured sub-agent."""

            def __init__(self, **kwargs: Any) -> None:
                # Use the captured config, but allow overrides
                super().__init__(config=captured_config, **kwargs)

        # Register the agent
        self.agent_registry.register(ConfiguredSubAgent)

    async def run(
        self,
        task: str,
        max_steps: int = 50,
    ) -> RunResult:
        """Run a task with the multi-agent system.

        Args:
            task: The task description/prompt
            max_steps: Maximum steps for main agent

        Returns:
            RunResult with output, files, and metrics
        """
        import time

        self._start_time = datetime.now()
        start_time = time.time()

        # Initialize if needed
        if not self._initialized:
            await self.initialize()

        try:
            # Setup context
            main_config = self.config.agents.get(self.config.main)
            if main_config and main_config.system_prompt:
                self.context.append_system(main_config.system_prompt)

            self.context.append_user(task)
            self.context.freeze_prefix()

            # Log initial step
            await self.logger.log_step(
                agent_id=self.config.main,
                input_prompt=task,
                input_context_size=self.context.token_count,
                model=main_config.model if main_config else "unknown",
            )

            # Run main agent loop
            result = await self._run_main_agent(task, max_steps)

            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            files = await self.memory.list_files()

            # Finalize logging
            await self.logger.finalize(
                success=result.get("success", True),
                final_result=result,
            )

            return RunResult(
                success=result.get("success", True),
                reward=result.get("reward", 0.0),
                output=result.get("output"),
                files=files,
                error=result.get("error"),
                duration_ms=duration_ms,
                steps=self.logger._step_count,
                logs_dir=str(self.logger.run_dir),
            )

        except Exception as e:
            logger.exception("MultiAgentRunner failed:")
            duration_ms = (time.time() - start_time) * 1000

            await self.logger.finalize(
                success=False,
                error=str(e),
            )

            return RunResult(
                success=False,
                error=str(e),
                duration_ms=duration_ms,
                steps=self.logger._step_count,
                logs_dir=str(self.logger.run_dir),
            )

    async def _run_main_agent(
        self,
        task: str,
        max_steps: int,
    ) -> dict[str, Any]:
        """Run the main orchestrator agent.

        Args:
            task: Task prompt
            max_steps: Maximum steps

        Returns:
            Result dict
        """
        # If we have an EvalContext, use it
        if self.ctx is not None:
            return await self._run_with_ctx(task, max_steps)

        # Otherwise, run standalone
        return await self._run_standalone(task, max_steps)

    async def _run_with_ctx(
        self,
        task: str,
        max_steps: int,
    ) -> dict[str, Any]:
        """Run with EvalContext (full integration).

        This provides:
        - Tool access via ctx.call_tool()
        - Reward from ctx.reward
        - Full telemetry
        - Sub-agents as callable tools
        """
        from fastmcp.tools import Tool as FastMCPTool
        from fastmcp.tools.tool import ToolResult

        # Type guard - this method is only called when ctx is not None
        assert self.ctx is not None, "EvalContext is required for _run_with_ctx"
        ctx = self.ctx  # Local reference for type checker

        main_config = self.config.agents.get(self.config.main)
        model = main_config.model if main_config else "gpt-4o-mini"

        # Register sub-agent tools with the context so the LLM can call them
        agent_tool_schemas = self.agent_registry.list_tools()
        for tool_schema in agent_tool_schemas:
            tool_name = tool_schema["name"]
            tool_desc = tool_schema.get("description", f"Call sub-agent: {tool_name}")
            input_schema = tool_schema.get("inputSchema", {})
            logger.info(f"Registering sub-agent tool: {tool_name}")

            # Create a custom Tool subclass that accepts any arguments
            # This bypasses FastMCP's function signature validation
            class SubAgentTool(FastMCPTool):
                """A tool wrapper that calls a sub-agent with arbitrary arguments."""

                def __init__(
                    self,
                    agent_name: str,
                    parent_ctx: Any,
                    registry: AgentToolRegistry,
                    step_logger: StepLogger,
                    compactor: ContextCompactor,
                    context: AppendOnlyContext,
                    progressive_compaction: bool,
                    name: str,
                    description: str,
                    parameters: dict[str, Any],
                ) -> None:
                    super().__init__(
                        name=name,
                        description=description,
                        parameters=parameters,
                    )
                    self._agent_name = agent_name
                    self._parent_ctx = parent_ctx
                    self._registry = registry
                    self._step_logger = step_logger
                    self._compactor = compactor
                    self._context = context
                    self._progressive_compaction = progressive_compaction

                async def run(self, arguments: dict[str, Any]) -> ToolResult:
                    """Run the sub-agent with the provided arguments.

                    Returns MINIMAL result to parent (token optimized):
                    - output: Natural language summary
                    - success: bool
                    - error: Error message if failed (KEPT)
                    - artifacts: List of file paths
                    - summary: Brief action summary
                    - log_file: Path to detailed YAML log
                    """
                    logger.info(f"Calling sub-agent: {self._agent_name}")
                    
                    # Report progress
                    if hasattr(self._parent_ctx, 'progress_callback') and callable(getattr(self._parent_ctx, 'progress_callback', None)):
                        self._parent_ctx.progress_callback(f"🔧 Calling {self._agent_name} agent...")

                    # Get prompt from arguments
                    prompt = arguments.get("prompt", str(arguments))

                    try:
                        # Pass run_id so sub-agent logs to same directory as parent
                        result = await self._registry.call(
                            self._agent_name,
                            self._parent_ctx,
                            run_id=self._step_logger.run_id,
                            **arguments,
                        )

                        # Result is already in minimal SubAgentResult format
                        if hasattr(result, "model_dump"):
                            result_dict: dict[str, Any] = result.model_dump()
                        elif isinstance(result, dict):
                            result_dict = result
                        else:
                            result_dict = {"output": str(result), "success": True}

                        # Log sub-agent call in main YAML log
                        await self._step_logger.log_subagent_call(
                            subagent_name=self._agent_name,
                            prompt=prompt,
                            result=result_dict,
                        )
                        
                        # Report completion
                        if hasattr(self._parent_ctx, 'progress_callback') and callable(getattr(self._parent_ctx, 'progress_callback', None)):
                            summary = result_dict.get('summary', 'completed')
                            self._parent_ctx.progress_callback(f"✅ {self._agent_name}: {summary}")

                        # Progressive compaction after each sub-agent call
                        # Per the following principle: keep context lean
                        if self._progressive_compaction and self._compactor.should_compact(
                            self._context
                        ):
                            compacted_count = self._compactor.compact_context(self._context)
                            if compacted_count > 0:
                                logger.debug(
                                    f"Progressive compaction: compacted {compacted_count} entries"
                                )
                                await self._step_logger.log_context_event(
                                    "progressive_compaction",
                                    {
                                        "after_subagent": self._agent_name,
                                        "entries_compacted": compacted_count,
                                        "context_tokens": self._context.token_count,
                                    },
                                )

                        logger.debug(
                            f"Sub-agent {self._agent_name} completed: "
                            f"{result_dict.get('summary', 'no summary')}"
                        )

                        # Return minimal result to parent context
                        # NO tool_calls or tool_results arrays - they're in the log file
                        return ToolResult(content=result_dict)

                    except Exception as e:
                        logger.exception(f"Sub-agent {self._agent_name} failed:")
                        
                        # Report error
                        if hasattr(self._parent_ctx, 'progress_callback') and callable(getattr(self._parent_ctx, 'progress_callback', None)):
                            self._parent_ctx.progress_callback(f"❌ {self._agent_name} failed: {str(e)[:50]}")

                        error_result = {
                            "output": "",
                            "success": False,
                            "error": str(e),  # KEEP errors visible per Manus principle
                            "artifacts": [],
                            "summary": "Failed with error",
                        }

                        # Log error
                        await self._step_logger.log_subagent_call(
                            subagent_name=self._agent_name,
                            prompt=prompt,
                            result=error_result,
                        )

                        return ToolResult(content=error_result)

            # Create and register the sub-agent tool
            sub_agent_tool = SubAgentTool(
                agent_name=tool_name,
                parent_ctx=ctx,
                registry=self.agent_registry,
                step_logger=self.logger,
                compactor=self.compactor,
                context=self.context,
                progressive_compaction=self.config.progressive_compaction,
                name=tool_name,
                description=tool_desc,
                parameters=input_schema or {"type": "object", "properties": {}},
            )
            
            # Attach progress callback to context so tools can access it (dynamic attribute)
            if self.progress_callback:
                setattr(ctx, 'progress_callback', self.progress_callback)
            
            ctx._tool_manager.add_tool(sub_agent_tool)

        # Rebuild routing to include new tools
        await ctx._build_routing()

        # Log available tools
        all_tools = await ctx.list_tools()
        tool_names = [t.name for t in all_tools]
        logger.info(f"All available tools: {tool_names}")

        # Build system prompt with conversation history injected directly
        base_system_prompt = main_config.system_prompt if main_config else ""
        
        # Read and inject previous conversation history from logs
        conversation_history = self._get_conversation_history()
        if conversation_history:
            history_section = f"\n\nPREVIOUS CONVERSATION:\n{conversation_history}\n\n---\nCurrent message follows:"
            system_prompt = base_system_prompt + history_section
        else:
            system_prompt = base_system_prompt

        # Choose agent based on model prefix
        if model.startswith("anthropic/") or "claude" in model.lower():
            from hud.agents.claude import ClaudeAgent
            agent = ClaudeAgent.create(
                ctx=ctx,
                model=model.replace("anthropic/", ""),
                system_prompt=system_prompt,
            )
        else:
            # Default to OpenAI
            from hud.agents.openai import OpenAIAgent
            agent = OpenAIAgent.create(
                ctx=ctx,
                model=model,
                system_prompt=system_prompt,
            )

        # Run agent
        trace = await agent.run(ctx, max_steps=max_steps)

        return {
            "success": not trace.isError,
            "output": trace.content,
            "reward": trace.reward,
            "error": trace.info.get("error") if trace.info else None,
        }

    async def _run_standalone(
        self,
        task: str,
        max_steps: int,
    ) -> dict[str, Any]:
        """Run without EvalContext (standalone mode).

        Useful for testing or when full HUD integration not needed.
        """
        main_config = self.config.agents.get(self.config.main)

        # Simple execution loop
        step = 0
        while step < max_steps:
            step += 1

            # Check if we need to compact
            if self.compactor.should_compact(self.context):
                compacted = self.compactor.compact_context(self.context)
                await self.logger.log_context_event(
                    "compaction",
                    {"entries_compacted": compacted},
                )

            # For standalone mode, we just return after one step
            # Full implementation would loop with LLM calls
            break

        return {
            "success": True,
            "output": f"Standalone execution completed: {task}",
            "reward": 0.0,
        }

    async def call_agent_tool(
        self,
        name: str,
        **kwargs: Any,
    ) -> Any:
        """Call a sub-agent as a tool.

        Args:
            name: Agent tool name
            **kwargs: Arguments for the agent

        Returns:
            Structured result from agent
        """
        # Log the call
        step_id = await self.logger.log_step(
            agent_id=self.config.main,
            input_prompt=f"Calling agent tool: {name}",
            input_context_size=self.context.token_count,
        )

        tool_call_id = await self.logger.log_tool_call(
            step_id=step_id,
            tool_name=name,
            tool_args=kwargs,
            agent_id=self.config.main,
        )

        try:
            # Execute via registry
            result = await self.agent_registry.call(name, self.ctx, **kwargs)

            # Process result for context
            result_str = str(result)
            processed = await self.offloader.process_tool_result(result_str, name)

            # Add to context
            self.context.append_tool_result(processed, tool_name=name)

            # Log result
            await self.logger.log_tool_result(
                call_id=tool_call_id,
                result=result_str,
                success=True,
            )

            return result

        except Exception as e:
            error = str(e)
            self.context.append_error(error)

            await self.logger.log_tool_result(
                call_id=tool_call_id,
                result=error,
                success=False,
            )

            raise

    async def execute_code(self, code: str) -> dict[str, Any]:
        """Execute Python code via CodeAct.

        Args:
            code: Python code to execute

        Returns:
            Execution result
        """
        # Initialize CodeAct if needed
        if self._codeact is None:
            self._codeact = CodeActExecutor(
                workspace=str(self.workspace),
                keep_errors=True,
            )

        result = await self._codeact.execute(code)

        # Log the execution
        await self.logger.log_step(
            agent_id=self.config.main,
            input_prompt=f"CodeAct execution:\n{code}",
            output_response=result.to_context(),
            model="codeact",
            error=result.error if not result.success else None,
        )

        # Add result to context
        context_str = result.to_context()
        processed = await self.offloader.process_tool_result(context_str, "python")
        self.context.append_tool_result(processed, tool_name="python")

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "return_value": result.return_value,
        }

    def _get_conversation_history(self) -> str:
        """Read previous conversation from logs and format for injection.
        
        Returns:
            Formatted conversation history string, or empty string if no history.
        """
        try:
            if not self.logger.main_log_file.exists():
                return ""
            
            import yaml
            with open(self.logger.main_log_file) as f:
                log_data = yaml.safe_load(f)
            
            if not log_data or "steps" not in log_data:
                return ""
            
            # Extract conversation from steps
            history_parts = []
            for step in log_data["steps"]:
                # User messages
                if step.get("prompt"):
                    history_parts.append(f"User: {step['prompt']}")
                
                # Sub-agent results (assistant responses)
                if step.get("action") == "call_subagent" and step.get("result"):
                    result = step["result"]
                    output = result.get("output", "")
                    if output:
                        # Truncate very long outputs
                        if len(output) > 500:
                            output = output[:500] + "..."
                        history_parts.append(f"Assistant (via {step.get('subagent', 'agent')}): {output}")
            
            return "\n".join(history_parts) if history_parts else ""
            
        except Exception as e:
            logger.warning(f"Failed to read conversation history: {e}")
            return ""

    def get_context(self) -> AppendOnlyContext:
        """Get the current context."""
        return self.context

    def get_memory(self) -> FilesystemMemory:
        """Get the filesystem memory."""
        return self.memory

    async def shutdown(self) -> None:
        """Shutdown the runner and release resources."""
        if self._codeact is not None:
            await self._codeact.shutdown()


__all__ = ["MultiAgentRunner", "RunResult"]

