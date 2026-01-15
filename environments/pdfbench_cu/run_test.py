#!/usr/bin/env python3
"""Test script for PDFBench Computer Use environment."""

import asyncio
import sys
sys.path.insert(0, "/Users/erikq/Documents/bytedance/hud-python")

import hud
from hud.datasets import Task
from hud.agents import ClaudeAgent
from hud.clients import MCPClient


async def main():
    print("Starting PDFBench CU test...")

    async with hud.async_trace("pdfbench_cu_test"):
        # Create task from sample
        task = Task(
            prompt="""You are looking at an EyeMed vision enrollment form in a browser. Fill out the form with the following information:

**Employer Information:**
- Company name: GREENFIELD MANUFACTURING CORP
- Group number: 78432
- Effective date: 01/01/2025

**Employee Information:**
- Last name: MITCHELL
- First name: ROBERT
- Middle initial: J
- Date of birth: 03/15/1988
- Address: 4521 OAKWOOD DRIVE, APARTMENT 12B
- City: COLUMBUS
- State: OH
- ZIP: 43215
- Phone: 614-555-8823
- Last four of SSN: 4472
- Email: rob.mitchell88@gmail.com
- Check the male checkbox
- Check the "Add" option for change type

Use the computer tool to:
1. Take a screenshot to see the form
2. Click on each form field and type the values
3. Use Tab key to move between fields
4. Check checkboxes by clicking on them
5. Scroll if needed to see more of the form""",
            mcp_config={
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
            },
            setup_tool={
                "name": "setup",
                "arguments": {
                    "name": "load_pdf",
                    "arguments": {
                        "pdf_path": "/app/pdfs/eyemed.pdf",
                        "solution_path": "/opt/tbench/solution.json"
                    }
                }
            },
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

        # Create MCP client
        client = MCPClient(mcp_config=task.mcp_config)

        # Create agent
        agent = ClaudeAgent(
            mcp_client=client,
            model="claude-sonnet-4-20250514",
            allowed_tools=["computer"]
        )

        print("Running agent...")
        result = await agent.run(task, max_steps=20)

        print(f"\nResult: {result}")
        print(f"Reward: {result.reward if hasattr(result, 'reward') else 'N/A'}")

        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
