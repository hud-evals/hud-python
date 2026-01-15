#!/usr/bin/env python3
"""Run full agent evaluation for PDFBench Computer Use."""

import asyncio
import logging
import hud
from hud.agents import ClaudeAgent
from hud.types import Task

# Enable verbose logging
logging.basicConfig(level=logging.INFO)


async def main():
    print("Starting PDFBench CU evaluation...", flush=True)

    # Define MCP config
    mcp_config = {
        "local": {
            "command": "docker",
            "args": [
                "run", "--rm", "-i",
                "-v", "/Users/erikq/Documents/bytedance/harbor-pdfbench/harbor_tasks/pdfbench_eyemed_x001/environment/pdfs:/app/pdfs:ro",
                "-v", "/Users/erikq/Documents/bytedance/harbor-pdfbench/harbor_tasks/pdfbench_eyemed_x001/environment/solution.json:/opt/tbench/solution.json:ro",
                "-e", "PDF_PATH=/app/pdfs/eyemed.pdf",
                "-e", "SOLUTION_PATH=/opt/tbench/solution.json",
                "hud-pdfbench-cu:latest"
            ]
        }
    }

    # Define prompt
    prompt = """You are looking at an EyeMed vision enrollment form in a browser. Fill out the form with the following information:

**Employer Information:**
- Company name: GREENFIELD MANUFACTURING CORP
- Group number: 78432
- Effective date: 01/01/2025

**Employee Information:**
- Last name: MITCHELL
- First name: ROBERT
- Middle initial: J
- Date of birth: 03/15/1988

Use the computer tool to:
1. First take a screenshot to see the current form state
2. Click on each form field and type the values
3. Use Tab key to move between fields if needed
4. Scroll down if needed to see more fields

Fill in as many fields as you can see in the form."""

    # Create Task object - this logs the prompt to the trace
    task = Task(
        prompt=prompt,
        mcp_config=mcp_config,
        evaluate_tool={
            "name": "evaluate",
            "arguments": {
                "name": "verify_fields",
                "arguments": {
                    "solution_path": "/opt/tbench/solution.json",
                    "fuzzy_match": True,
                    "partial_credit": True
                }
            }
        }
    )

    # Create agent with computer tool
    agent = ClaudeAgent(
        model="claude-sonnet-4-20250514",
        allowed_tools=["computer"],
        initial_screenshot=True,
    )

    print("Running agent with Task (max 10 steps)...", flush=True)
    print(f"Prompt being sent:\n{prompt[:200]}...", flush=True)

    try:
        result = await agent.run(task, max_steps=10)
    except Exception as e:
        print(f"Agent run error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    print(f"\n{'='*60}")
    print(f"Result: {result}")
    print(f"Reward: {result.reward}")
    print(f"{'='*60}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
