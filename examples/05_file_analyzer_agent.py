#!/usr/bin/env python3
"""
File Analyzer Agent - A Beginner-Friendly Example

This example demonstrates the core concepts of the HUD SDK:
- Creating an environment with custom tools
- Defining evaluation scenarios
- Running agents with different models
- Comparing model performance

The agent can list files, read their contents, and analyze text statistics.

Usage:
  # Set your API key
  export HUD_API_KEY="sk-hud-..."

  # Run the example
  uv run python examples/05_file_analyzer_agent.py

  # Or with a specific model
  uv run python examples/05_file_analyzer_agent.py --model gpt-4o

Requirements:
  - HUD_API_KEY environment variable
  - uv sync (to install dependencies)
"""

import argparse
import asyncio
from pathlib import Path

from openai import AsyncOpenAI

import hud
from hud.agents import OpenAIAgent
from hud.settings import settings


# =============================================================================
# Environment Setup
# =============================================================================

env = hud.Environment("file-analyzer")


@env.tool()
def list_files(directory: str = ".") -> str:
    """List all files in a directory.

    Args:
        directory: Path to directory (default: current directory)

    Returns:
        List of files as a string
    """
    try:
        path = Path(directory)
        files = [f.name for f in path.iterdir() if f.is_file()]
        return f"Files in {directory}:\n" + "\n".join(f"- {f}" for f in files)
    except Exception as e:
        return f"Error: {e}"


@env.tool()
def read_file(filepath: str) -> str:
    """Read contents of a file.

    Args:
        filepath: Path to the file

    Returns:
        File contents (limited to first 1000 characters)
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:1000]  # Limit to prevent token overflow
    except Exception as e:
        return f"Error reading file: {e}"


@env.tool()
def count_words(text: str) -> str:
    """Count words in text and provide statistics.

    Args:
        text: Text to analyze

    Returns:
        Word count and statistics
    """
    words = text.split()
    lines = text.split("\n")
    chars = len(text)

    # Calculate average word length (excluding spaces/newlines)
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

    return f"""Statistics:
- Words: {len(words)}
- Lines: {len(lines)}
- Characters: {chars}
- Average word length: {avg_word_length:.1f}
"""


# =============================================================================
# Evaluation Scenario
# =============================================================================


@env.scenario("analyze-readme")
async def analyze_readme():
    """Scenario: Analyze the README.md file in the current directory.

    The agent should:
    1. List files to confirm README.md exists
    2. Read the README.md file
    3. Count words in it
    4. Report the word count
    """
    prompt = """Please analyze the README.md file in the current directory:

1. First, list the files to confirm README.md exists
2. Read the README.md file
3. Count the words in it
4. Tell me the word count

Use the available tools: list_files, read_file, and count_words
"""

    response = yield prompt

    # Evaluate: Did the agent mention a word count?
    response_lower = response.lower()
    if "word" in response_lower and any(char.isdigit() for char in response):
        yield 1.0  # Success - mentioned words and a number
    else:
        yield 0.3  # Partial - didn't complete the task


# =============================================================================
# Main Execution
# =============================================================================


async def run_example(model: str = "gpt-4o-mini", verbose: bool = False):
    """Run the file analyzer example.

    Args:
        model: Model to use (default: gpt-4o-mini)
        verbose: Enable verbose output
    """
    if not settings.api_key:
        print("❌ Error: HUD_API_KEY not found!")
        print("\nTo run this example:")
        print("1. Get your API key from https://hud.ai/settings/api-keys")
        print("2. Set it: export HUD_API_KEY='sk-hud-...'")
        print("3. Run again: uv run python examples/05_file_analyzer_agent.py")
        return

    print("=" * 70)
    print("FILE ANALYZER AGENT")
    print("=" * 70)
    print(f"\n🤖 Model: {model}")
    print("📋 Task: Analyze README.md file")
    print("🔧 Tools: list_files, read_file, count_words")
    print("\n" + "=" * 70)

    # Create agent
    client = AsyncOpenAI(
        base_url=settings.hud_gateway_url,
        api_key=settings.api_key,
    )

    agent = OpenAIAgent.create(
        model=model,
        model_client=client,
        validate_api_key=False,
        verbose=verbose,
    )

    # Run the scenario
    print("\n🚀 Running agent...\n")
    result = await env("analyze-readme").run(agent=agent, max_steps=10)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"✅ Task completed!")
    print(f"📊 Reward: {result.reward}")
    print(f"🔢 Tool calls: {result.num_messages - 1}")  # Subtract initial prompt

    if result.content:
        print(f"\n📝 Agent's response:")
        print(result.content)

    print("\n" + "=" * 70)
    print("💡 What happened:")
    print("=" * 70)
    print("""
1. The agent received the task to analyze README.md
2. It automatically figured out which tools to use
3. It called the tools in the right order
4. It synthesized the results into a response

This demonstrates:
- Environment creation with custom tools
- Scenario-based evaluation
- Automatic tool usage by the agent
- Reward-based success measurement

View the full trace at the URL shown above to see every tool call!
    """)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="File Analyzer Agent - Beginner-friendly HUD SDK example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    await run_example(model=args.model, verbose=args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
