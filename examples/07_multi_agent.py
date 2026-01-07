"""Multi-Agent System Example.

This example demonstrates the multi-agent system with:
- YAML configuration: Define agents in YAML files
- Agent-as-Tool pattern: Sub-agents exposed as callable tools
- Structured returns: Type-safe sub-agent results with make_schema()

The module provides:
- MultiAgentRunner: Orchestrate multi-agent tasks
- make_schema(): Create custom return schemas in one line
- SubAgentConfig / create_sub_agent: Programmatic sub-agent creation
- Built-in schemas: CodeResult, ResearchResult, ReviewResult, PlanResult
"""

import asyncio
from pathlib import Path

import hud
from hud.multi_agent import (
    # Core runner
    MultiAgentRunner,
    RunResult,
    # Schema factory
    make_schema,
    # Built-in schemas
    CodeResult,
    ResearchResult,
    ReviewResult,
    GenericResult,
    # Sub-agent utilities
    SubAgentConfig,
    create_sub_agent,
    # Config
    load_config,
    register_schema,
)


# =============================================================================
# Example 1: Basic MultiAgentRunner with YAML Config
# =============================================================================


async def example_basic_runner():
    """Basic multi-agent setup with YAML configuration.
    
    This is the recommended way to use multi-agent:
    1. Define agents in YAML files (agents/*.yaml)
    2. Create an environment with tools
    3. Run with MultiAgentRunner
    
    YAML files needed:
    - agents/main.yaml (orchestrator)
    - agents/coder.yaml (specialist)
    """
    print("\n" + "=" * 60)
    print("Example 1: MultiAgentRunner with YAML Config")
    print("=" * 60)

    # Create an environment with your tools
    env = hud.Environment("coding-env")
    
    # Add any tools the agents need
    @env.tool()
    def run_python(code: str) -> str:
        """Execute Python code and return the output."""
        # In reality, you'd connect to a Jupyter/code execution environment
        return f"Executed: {code[:50]}..."

    @env.tool()
    def read_file(path: str) -> str:
        """Read a file from the workspace."""
        return f"Contents of {path}"

    # Run with multi-agent orchestration
    async with hud.eval(env(), name="multi-agent-demo") as ctx:
        runner = MultiAgentRunner(
            config_dir=Path("agents/"),  # Directory with YAML configs
            ctx=ctx,
            workspace=Path("./workspace"),
        )

        result = await runner.run(
            task="Write a Python function to calculate fibonacci numbers",
            max_steps=20,
        )

        print(f"Success: {result.success}")
        print(f"Steps: {result.steps}")
        print(f"Logs: {result.logs_dir}")
        if result.output:
            print(f"Output: {result.output[:500]}")


# =============================================================================
# Example 2: Custom Schemas with make_schema()
# =============================================================================


async def example_custom_schema():
    """Create custom return schemas for your sub-agents.
    
    Use make_schema() to define structured output in one line.
    The schema is auto-registered for use in YAML configs.
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom Schemas with make_schema()")
    print("=" * 60)

    # Create a custom schema - one line!
    DataAnalysisResult = make_schema(
        "DataAnalysisResult",
        insights=list[str],              # List defaults to []
        chart_path=(str | None, None),   # Optional field
        metrics=dict[str, float],        # Dict defaults to {}
        confidence=(float, 0.8),         # Explicit default
    )

    print(f"Created: {DataAnalysisResult.__name__}")
    print(f"Fields: {list(DataAnalysisResult.model_fields.keys())}")

    # The schema includes AgentResultBase fields automatically
    result = DataAnalysisResult(
        insights=["Revenue increased 20%", "Q4 was strongest quarter"],
        chart_path="/workspace/revenue_chart.png",
        metrics={"total_revenue": 1_000_000, "growth_rate": 0.2},
    )

    print(f"\nExample result:")
    print(f"  insights: {result.insights}")  # type: ignore[attr-defined]
    print(f"  success: {result.success}")  # type: ignore[attr-defined] (from AgentResultBase)

    print("\n📝 Use in YAML config:")
    print("""
    # agents/analyst.yaml
    name: analyst
    type: specialist
    model: claude-sonnet-4-5
    system_prompt: "You analyze data and provide insights..."
    returns:
      schema: DataAnalysisResult  # References your custom schema
    """)


# =============================================================================
# Example 3: Using env.agent_tool() Pattern
# =============================================================================


async def example_agent_tool():
    """Register agents as tools directly on the environment.
    
    This is useful when you want to dynamically create agent tools
    without YAML configuration.
    """
    print("\n" + "=" * 60)
    print("Example 3: env.agent_tool() Pattern")
    print("=" * 60)

    env = hud.Environment("agent-tools-demo")

    # Register agents as callable tools
    env.agent_tool(
        name="coder",
        model="claude-sonnet-4-5",
        system_prompt="You are a Python expert. Write clean, efficient code.",
        max_steps=10,
    )

    env.agent_tool(
        name="reviewer",
        model="gpt-4o",
        system_prompt="You review code for bugs, style issues, and security.",
        max_steps=5,
    )

    print("Registered agent tools: coder, reviewer")
    print("\n📝 Usage:")
    print("""
    async with hud.eval(env()) as ctx:
        # Call agents like any other tool
        code = await ctx.call_tool(name="coder", arguments={
            "prompt": "Write a fibonacci function"
        })
        
        review = await ctx.call_tool(name="reviewer", arguments={
            "prompt": f"Review this code: {code}"
        })
    """)


# =============================================================================
# Example 4: Programmatic Sub-Agent Creation
# =============================================================================


async def example_programmatic_subagent():
    """Create sub-agents programmatically without YAML.
    
    Use SubAgentConfig and create_sub_agent() for full control.
    """
    print("\n" + "=" * 60)
    print("Example 4: Programmatic Sub-Agent Creation")
    print("=" * 60)

    # Create sub-agent config
    config = SubAgentConfig(
        name="researcher",
        model="claude-sonnet-4-5",
        system_prompt="You research topics and provide summaries with sources.",
        max_steps=15,
        return_schema=ResearchResult,  # Built-in schema
    )

    print(f"Created config: {config.name}")
    print(f"  Model: {config.model}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Return schema: {config.return_schema.__name__ if config.return_schema else 'None'}")

    print("\n📝 Usage with EvalContext:")
    print("""
    async with hud.eval(env()) as ctx:
        agent = create_sub_agent(config, ctx)
        trace = await agent.run(ctx, max_steps=config.max_steps)
        print(trace.content)  # Agent's response
    """)


# =============================================================================
# Example 5: Built-in Schemas Reference
# =============================================================================


def example_builtin_schemas():
    """Reference for built-in return schemas.
    
    These are ready to use in your YAML configs.
    """
    print("\n" + "=" * 60)
    print("Example 5: Built-in Schemas Reference")
    print("=" * 60)

    schemas = {
        "CodeResult": CodeResult,
        "ResearchResult": ResearchResult,
        "ReviewResult": ReviewResult,
        "GenericResult": GenericResult,
    }

    for name, schema in schemas.items():
        fields = list(schema.model_fields.keys())
        # Filter out base fields for clarity
        base_fields = {"success", "error", "duration_ms", "timestamp", "tool_calls", "tool_results"}
        custom_fields = [f for f in fields if f not in base_fields]
        
        print(f"\n{name}:")
        print(f"  Custom fields: {', '.join(custom_fields)}")

    print("\n📝 Use in YAML:")
    print("""
    # For coding agents
    returns:
      schema: CodeResult  # explanation, files_created, tests_passed, etc.
    
    # For research agents  
    returns:
      schema: ResearchResult  # summary, sources, key_findings, etc.
    
    # For code review agents
    returns:
      schema: ReviewResult  # summary, issues, approved, score, etc.
    """)


# =============================================================================
# Example 6: Complete YAML Config Reference
# =============================================================================


def example_yaml_reference():
    """Reference for YAML configuration structure."""
    print("\n" + "=" * 60)
    print("Example 6: YAML Configuration Reference")
    print("=" * 60)

    print("""
📁 agents/
├── main.yaml       # Orchestrator agent
├── coder.yaml      # Coding specialist
└── reviewer.yaml   # Code review specialist

# ═══════════════════════════════════════════════════════════
# agents/main.yaml - Orchestrator
# ═══════════════════════════════════════════════════════════
name: main
type: orchestrator
model: anthropic/claude-sonnet-4-5
system_prompt: |
  You are a project manager that delegates tasks to specialists.
  
  Available specialists:
  - coder: For writing and modifying code
  - reviewer: For code review and quality checks
  
  Break down tasks and delegate appropriately.

# ═══════════════════════════════════════════════════════════
# agents/coder.yaml - Specialist
# ═══════════════════════════════════════════════════════════
name: coder
type: specialist
model: anthropic/claude-sonnet-4-5
system_prompt: |
  You are an expert Python developer.
  Write clean, well-documented, tested code.
tools:
  - bash
  - str_replace_based_edit_tool
  - read_file
returns:
  schema: CodeResult
max_steps: 20

# ═══════════════════════════════════════════════════════════
# agents/reviewer.yaml - Specialist
# ═══════════════════════════════════════════════════════════
name: reviewer
type: specialist
model: anthropic/claude-sonnet-4-5
system_prompt: |
  You review code for:
  - Bugs and logic errors
  - Security vulnerabilities
  - Style and best practices
  - Performance issues
returns:
  schema: ReviewResult
max_steps: 10
    """)


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run examples."""
    print("=" * 60)
    print("HUD Multi-Agent System Examples")
    print("=" * 60)

    # Standalone examples (no environment needed)
    await example_custom_schema()
    await example_agent_tool()
    await example_programmatic_subagent()
    example_builtin_schemas()
    example_yaml_reference()

    # Full runner example (needs YAML configs and environment)
    # Uncomment to run with actual environment:
    # await example_basic_runner()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
