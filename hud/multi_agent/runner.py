"""MultiAgentRunner: Orchestration of multi-agent system.

This module ties everything together:
- Loads configuration from YAML
- Creates main agent and sub-agents
- Manages logging
- Handles the agent-as-tool pattern
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.multi_agent.config import MultiAgentConfig, load_config
from hud.multi_agent.logger import StepLogger
from hud.multi_agent.sub_agent import SubAgentConfig, SubAgentResult, create_sub_agent

if TYPE_CHECKING:
    from hud.eval.context import EvalContext

logger = logging.getLogger(__name__)


def _build_result(
    schema: type | None,
    *,
    output: str,
    success: bool,
    error: str | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    """Build a result dict using the specified schema.
    
    First tries to parse JSON from the agent's output. If successful,
    uses that data to construct the schema. Otherwise, falls back to
    putting the output in the schema's primary field.
    
    Falls back to SubAgentResult if schema is None or construction fails.
    """
    from hud.multi_agent.sub_agent import parse_json_from_output
    
    # Fall back to SubAgentResult if no schema specified
    if schema is None:
        return SubAgentResult(
            output=output,
            success=success,
            error=error,
            duration_ms=duration_ms,
        ).model_dump()
    
    schema_name = schema.__name__

    def _primary_field_for(schema_type: type) -> str:
        """Pick the primary content field for a schema, with inference fallback."""
        explicit = {
            "SubAgentResult": "output",
            "GenericResult": "output",
            "CodeResult": "explanation",
            "ResearchResult": "summary",
            "ReviewResult": "summary",
            "PlanResult": "goal",
        }.get(schema_type.__name__)
        if explicit:
            return explicit

        metadata_fields = {"success", "error", "duration_ms", "timestamp", "tool_calls", "tool_results"}
        fields = getattr(schema_type, "model_fields", None) or getattr(schema_type, "__fields__", {}) or {}
        for name in fields:
            if name not in metadata_fields:
                return name
        return "output"

    output_field = _primary_field_for(schema)

    # Try to parse JSON from the output
    parsed_json = parse_json_from_output(output)
    
    if parsed_json is not None:
        try:
            parsed_json["success"] = success
            parsed_json["duration_ms"] = duration_ms
            if error is not None:
                parsed_json["error"] = error
            
            result = schema(**parsed_json)
            result_dict = result.model_dump()
            
            if "output" not in result_dict:
                result_dict["output"] = result_dict.get(output_field, "")
            
            logger.debug(f"Successfully parsed structured {schema_name} from agent output")
            return result_dict
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Failed to parse JSON for {schema_name}: {e}")
    
    try:
        kwargs: dict[str, Any] = {
            output_field: output,
            "success": success,
            "duration_ms": duration_ms,
        }
        if error is not None:
            kwargs["error"] = error
            
        result = schema(**kwargs)
        result_dict = result.model_dump()
        
        if "output" not in result_dict:
            result_dict["output"] = result_dict.get(output_field, "")
        
        return result_dict
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to construct {schema_name}, falling back to SubAgentResult: {e}")
        return SubAgentResult(
            output=output,
            success=success,
            error=error,
            duration_ms=duration_ms,
        ).model_dump()


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
    2. Sets up logging
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
            ctx: EvalContext for evaluation (required for running tasks)
            workspace: Working directory for file operations
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

        # Logger
        self.logger = StepLogger(
            log_dir=Path(self.config.log_dir),
        )

        # Track state
        self._initialized = False
        self._start_time: datetime | None = None

    async def initialize(self) -> None:
        """Initialize the runner and all components."""
        if self._initialized:
            return

        self._initialized = True
        logger.info(
            f"MultiAgentRunner initialized with {len(self.config.agents)} agents"
        )

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
        self._start_time = datetime.now()
        start_time = time.time()

        # Initialize if needed
        if not self._initialized:
            await self.initialize()

        try:
            # Log initial step
            main_config = self.config.agents.get(self.config.main)
            await self.logger.log_step(
                agent_id=self.config.main,
                input_prompt=task,
                model=main_config.model if main_config else "unknown",
            )

            # Run main agent loop
            result = await self._run_main_agent(task, max_steps)

            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            
            # List files in workspace
            files = [str(p) for p in self.workspace.rglob("*") if p.is_file()]

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
            
        Raises:
            RuntimeError: If EvalContext is not provided
        """
        if self.ctx is None:
            raise RuntimeError(
                "EvalContext is required. MultiAgentRunner must be initialized with ctx parameter."
            )
        
        return await self._run_with_ctx(task, max_steps)

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

        # Explicit check - this method requires EvalContext
        if self.ctx is None:
            raise RuntimeError("EvalContext is required for _run_with_ctx")
        ctx = self.ctx  # Local reference for type checker

        main_config = self.config.agents.get(self.config.main)
        model = main_config.model if main_config else "gpt-4o-mini"

        # Register sub-agent tools with the context
        for agent_name, agent_config in self.config.agents.items():
            if agent_config.type != "specialist":
                continue  # Skip orchestrator

            tool_desc = agent_config.system_prompt[:100] if agent_config.system_prompt else f"Call sub-agent: {agent_name}"
            logger.info(f"Registering sub-agent tool: {agent_name}")

            # Create SubAgentConfig for this agent
            return_schema = agent_config.get_return_schema()
            sub_config = SubAgentConfig(
                name=agent_name,
                model=agent_config.model,
                system_prompt=agent_config.system_prompt,
                max_steps=agent_config.max_steps,
                tools=agent_config.tools,
                return_schema=return_schema,
            )

            # Create a custom Tool subclass for sub-agent calls
            class SubAgentTool(FastMCPTool):
                """A tool wrapper that calls a sub-agent with logging."""

                def __init__(
                    self,
                    agent_name: str,
                    agent_config: SubAgentConfig,
                    parent_ctx: Any,
                    step_logger: StepLogger,
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
                    self._agent_config = agent_config
                    self._parent_ctx = parent_ctx
                    self._step_logger = step_logger

                async def run(self, arguments: dict[str, Any]) -> ToolResult:
                    """Run the sub-agent with the provided arguments."""
                    print(f"🔄 Calling sub-agent: {self._agent_name}")

                    # Report progress
                    progress_cb = getattr(self._parent_ctx, 'progress_callback', None)
                    if callable(progress_cb):
                        progress_cb(f"🔧 Calling {self._agent_name} agent...")

                    # Get prompt from arguments
                    prompt = arguments.get("prompt", str(arguments))
                    start_time = time.time()

                    try:
                        # Create and run the sub-agent directly
                        agent = create_sub_agent(self._agent_config, self._parent_ctx)

                        # Set prompt on context temporarily
                        original_prompt = getattr(self._parent_ctx, "prompt", None)
                        self._parent_ctx.prompt = prompt

                        try:
                            trace = await agent.run(self._parent_ctx, max_steps=self._agent_config.max_steps)
                        finally:
                            self._parent_ctx.prompt = original_prompt

                        duration_ms = (time.time() - start_time) * 1000
                        
                        # Write sub-agent log file with full trace
                        subagent_log_file = self._step_logger.run_dir / f"{self._agent_name}.yaml"
                        subagent_log_data = {
                            "agent": self._agent_name,
                            "model": self._agent_config.model,
                            "prompt": prompt,
                            "duration_ms": duration_ms,
                            "success": not trace.isError,
                            "messages": trace.messages or [],
                            "content": trace.content,
                        }
                        if trace.isError and trace.info:
                            subagent_log_data["error"] = trace.info.get("error")
                        
                        import yaml
                        with open(subagent_log_file, "w") as f:
                            yaml.dump(subagent_log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                        
                        logger.debug(f"Sub-agent {self._agent_name} log written to {subagent_log_file}")

                        result_dict = _build_result(
                            self._agent_config.return_schema,
                            output=trace.content or "",
                            success=not trace.isError,
                            error=trace.info.get("error") if trace.isError and trace.info else None,
                            duration_ms=duration_ms,
                        )
                        
                        if trace.isError:
                            error_msg = trace.info.get("error") if trace.info else "unknown error"
                            print(f"⚠️  Sub-agent '{self._agent_name}' returned error: {error_msg}")
                        
                        # Set log_file to point to the sub-agent's detailed log
                        result_dict["log_file"] = str(subagent_log_file)
                        
                        # Set artifacts from files_created if present (CodeResult)
                        if "files_created" in result_dict and not result_dict.get("artifacts"):
                            result_dict["artifacts"] = [
                                f.get("path", f) if isinstance(f, dict) else str(f)
                                for f in result_dict["files_created"]
                            ]

                        # Log sub-agent call in main YAML log
                        await self._step_logger.log_subagent_call(
                            subagent_name=self._agent_name,
                            prompt=prompt,
                            result=result_dict,
                        )

                        # Report completion
                        progress_cb = getattr(self._parent_ctx, 'progress_callback', None)
                        if callable(progress_cb):
                            summary = result_dict.get('summary', 'completed')
                            progress_cb(f"✅ {self._agent_name}: {summary}")

                        logger.debug(
                            f"Sub-agent {self._agent_name} completed: "
                            f"{result_dict.get('summary', 'no summary')}"
                        )

                        return ToolResult(content=result_dict)

                    except Exception as e:
                        logger.exception(f"Sub-agent {self._agent_name} failed:")

                        duration_ms = (time.time() - start_time) * 1000

                        # Report error
                        progress_cb = getattr(self._parent_ctx, 'progress_callback', None)
                        if callable(progress_cb):
                            progress_cb(f"❌ {self._agent_name} failed: {str(e)[:50]}")

                        # Use the configured return schema or fall back to SubAgentResult
                        error_result = _build_result(
                            self._agent_config.return_schema,
                            output="",
                            success=False,
                            error=str(e),
                            duration_ms=duration_ms,
                        )

                        # Log error
                        await self._step_logger.log_subagent_call(
                            subagent_name=self._agent_name,
                            prompt=prompt,
                            result=error_result,
                        )

                        return ToolResult(content=error_result)

            # Create and register the sub-agent tool
            sub_agent_tool = SubAgentTool(
                agent_name=agent_name,
                agent_config=sub_config,
                parent_ctx=ctx,
                step_logger=self.logger,
                name=agent_name,
                description=tool_desc,
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The prompt for this agent"}
                    },
                    "required": ["prompt"],
                },
            )

            # Progress callback can be set directly on ctx by callers (e.g., playground)
            # No need to set it here - tools will check ctx.progress_callback directly

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

    async def shutdown(self) -> None:
        """Shutdown the runner and release resources."""
        pass  # Reserved for future cleanup


__all__ = ["MultiAgentRunner", "RunResult"]
