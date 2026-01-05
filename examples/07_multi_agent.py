"""Multi-Agent System Example.

This example demonstrates a multi-agent system with:
- Agent-as-Tool pattern: Sub-agents exposed as tools
- CodeAct: Agent writes Python code instead of JSON tools
- Filesystem as Memory: grep/glob search for external memory
- Append-only Context: KV cache optimization
"""

import asyncio
import os
from pathlib import Path

import hud
from hud.multi_agent import (
    # Core runner
    MultiAgentRunner,
    # Schemas for structured returns
    ResearchResult,
    CodeResult,
    ReviewResult,
    # Agent-as-tool decorators
    agent_as_tool,
    register_agent_tool,
    # Sub-agent base class
    SubAgent,
    # Context management
    AppendOnlyContext,
    # Filesystem memory
    FilesystemMemory,
    # CodeAct execution
    CodeActExecutor,
)


# =============================================================================
# Example 1: Using MultiAgentRunner with YAML config
# =============================================================================

# Create a local environment with tools for the demo
demo_env = hud.Environment("multi-agent-demo")


@demo_env.tool()
def fibonacci(n: int) -> list[int]:
    """Calculate Fibonacci sequence up to n numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib


@demo_env.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@demo_env.tool()
def echo(message: str) -> str:
    """Echo a message back."""
    return message


async def example_with_config():
    """Run a multi-agent task with access to BOTH Jupyter AND Browser environments.

    Uses MultiAgentRunner with:
    - main.yaml: Orchestrator that delegates to sub-agents
    - coder.yaml: Python/Jupyter specialist for code execution
    - browser.yaml: Web automation specialist for browsing
    
    The environment connects to BOTH:
    - Local Jupyter container (for Python/data tasks)
    - Remote hud-online-mind2web (for web browsing tasks)
    
    This allows sub-agents to use whichever tools they need!
    """
    print("\n" + "=" * 60)
    print("Example: Multi-Agent with Jupyter + Browser")
    print("=" * 60)

    from hud.environment import Environment

    # Connect to locally running containers
    JUPYTER_URL = os.getenv("HUD_DEV_URL", "http://localhost:8765/mcp")
    BROWSER_URL = os.getenv("HUD_BROWSER_URL", "http://localhost:8766/mcp")

    # Create a unified environment that connects to BOTH sources
    env = Environment("multi-agent-env")
    
    # 1. Connect to LOCAL Jupyter (for coding/data tasks)
    env.connect_url(JUPYTER_URL, alias="jupyter")
    
    # 2. Connect to LOCAL browser (hud-online-mind2web running locally)
    #    This gives sub-agents access to browser tools: navigate, click, screenshot, etc.
    env.connect_url(BROWSER_URL, alias="browser")

    # Multi-phase task: Browser scraping + Jupyter analysis
    # Using books.toscrape.com - a fast, reliable demo site for web scraping
    task_prompt = """
Scrape book data from an online bookstore and create a price analysis.

PHASE 1 - BROWSER SCRAPING:
1. Navigate to https://books.toscrape.com/
2. Extract data for the first 10 books visible on the homepage:
   - Book title
   - Price (in £)
   - Star rating (One to Five stars)
3. Record all the data before proceeding to Phase 2.

PHASE 2 - JUPYTER DATA ANALYSIS:
Using the scraped data, create a Python analysis:

1. Create a pandas DataFrame with the 10 books
2. Convert prices from £ to USD (use rate: 1 GBP = 1.27 USD)
3. Calculate:
   - Average book price in USD
   - Most expensive and cheapest books
   - Average rating (convert star text to numbers)
   - Price distribution stats
4. Create a bar chart showing prices by book title
5. Save the chart as 'book_analysis.png'
6. Print a summary table with all metrics

PHASE 3 - INSIGHTS:
Provide 3 insights about the book pricing patterns you observed.

Return the complete analysis with all findings.
"""

    env.prompt = task_prompt

    # Run with combined environment (has tools from BOTH Jupyter and Browser)
    async with hud.eval(env(), name="multi-agent-combined") as ctx:
        ctx.prompt = task_prompt

        print(f"Task: Book Store Scraping + Price Analysis")
        tools = await ctx.list_tools()
        print(f"Available tools ({len(tools)}): {[t.name for t in tools]}")

        runner = MultiAgentRunner(
            config_dir=Path("agents/"),
            ctx=ctx,
            workspace=Path("./workspace"),
        )

        result = await runner.run(
            task=ctx.prompt,
            max_steps=50,  # More steps needed for complex multi-phase task
        )

        print(f"\n{'='*60}")
        print("RESULT:")
        print(f"{'='*60}")
        print(f"  Success: {result.success}")
        print(f"  Steps: {result.steps}")
        print(f"  Files in workspace: {len(result.files)}")
        print(f"  Logs: {result.logs_dir}")
        print(f"\nAgent Output:")
        if result.output:
            print(f"  {result.output[:1500]}..." if len(result.output) > 1500 else f"  {result.output}")


# =============================================================================
# Example 2: Creating a custom sub-agent
# =============================================================================


@agent_as_tool(name="analyze", description="Analyze code complexity", returns=ReviewResult)
class AnalyzerAgent(SubAgent):
    """Custom sub-agent for code analysis."""

    async def run_isolated(self, prompt: str, **kwargs):
        """Analyze code and return structured result."""
        # In a real implementation, this would use an LLM
        # Here we demonstrate the pattern

        return {
            "summary": f"Analysis of: {prompt}",
            "issues": [],
            "approved": True,
            "score": 0.85,
            "suggestions": ["Consider adding type hints", "Add docstrings"],
            "security_concerns": [],
        }


async def example_custom_agent():
    """Demonstrate creating a custom sub-agent."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Sub-Agent")
    print("=" * 60)

    # Register the agent
    register_agent_tool(AnalyzerAgent)

    # Create and run the agent directly
    agent = AnalyzerAgent(isolation=True)
    result = await agent.run_isolated("def hello(): print('world')")

    print(f"Result: {result}")


# =============================================================================
# Example 3: Using AppendOnlyContext directly
# =============================================================================


async def example_context():
    """Demonstrate AppendOnlyContext usage."""
    print("\n" + "=" * 60)
    print("Example 3: AppendOnlyContext")
    print("=" * 60)

    context = AppendOnlyContext(max_tokens=128_000)

    # Add system message and freeze (stable prefix for KV cache)
    context.append_system("You are a helpful coding assistant.")
    context.freeze_prefix()

    # Add conversation
    context.append_user("Write a hello world function")
    context.append_assistant("Here's a simple function:\n```python\ndef hello():\n    print('Hello, World!')\n```")

    # Add tool interaction
    context.append_tool_call("python", {"code": "hello()"})
    context.append_tool_result("Hello, World!")

    print(f"Context entries: {len(context)}")
    print(f"Frozen prefix: {context.frozen_prefix_length}")
    print(f"Token count: {context.token_count}")
    print(f"Should compact: {context.should_compact()}")

    # Render for display
    print("\nRendered context:")
    print(context.render())


# =============================================================================
# Example 4: Using FilesystemMemory
# =============================================================================


async def example_memory():
    """Demonstrate FilesystemMemory usage."""
    print("\n" + "=" * 60)
    print("Example 4: FilesystemMemory")
    print("=" * 60)

    workspace = Path("./workspace_example")
    workspace.mkdir(exist_ok=True)

    memory = FilesystemMemory(workspace)

    # Store some content
    ref = await memory.store(
        "research_notes",
        "This is a long research document about Python async patterns...\n" * 50,
    )
    print(f"Stored: {ref}")

    # Store code
    await memory.store(
        "fibonacci",
        """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",
        extension="py",
    )

    # Search with grep
    results = await memory.search("fibonacci")
    print(f"Search results: {results}")

    # List files
    files = await memory.list_files()
    print(f"Files in workspace: {files}")


# =============================================================================
# Example 5: CodeAct Execution
# =============================================================================


async def example_codeact():
    """Demonstrate CodeAct execution."""
    print("\n" + "=" * 60)
    print("Example 5: CodeAct Execution")
    print("=" * 60)

    # Use SandboxExecutor for simple execution (no Jupyter needed)
    from hud.multi_agent import SandboxExecutor

    executor = SandboxExecutor(timeout=30)

    # Execute some Python code
    code = """
# Calculate first 10 Fibonacci numbers
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

result = list(fib(10))
print(f"Fibonacci: {result}")
"""

    result = await executor.execute(code)

    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    if result.error:
        print(f"Error: {result.error}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("Multi-Agent System Examples")
    print("=" * 60)

    # Example 1: Full MultiAgentRunner with YAML config
    await example_with_config()

    # Other examples (standalone, no HUD connection needed)
    # await example_custom_agent()
    # await example_context()
    # await example_memory()
    # await example_codeact()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

