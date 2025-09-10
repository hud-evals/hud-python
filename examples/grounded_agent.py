#!/usr/bin/env python3
"""
Grounded Agent Example

This example demonstrates the GroundedOpenAIChatAgent that separates:
- Visual grounding (element detection) using a specialized vision model
- High-level reasoning using GPT-4o or similar

Prerequisites:
1. Start a vLLM server with a vision model:
   vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000

2. Set your API keys:
   export OPENAI_API_KEY=your_openai_key
   export HUD_API_KEY=your_hud_key  # If using HUD cloud
"""

import asyncio
import os

import hud
from hud.agents.grounded_openai import GroundedOpenAIChatAgent
from hud.settings import settings
from hud.tools.grounding import GrounderConfig
from openai import AsyncOpenAI


async def main():
    """Run the grounded agent example."""
    
    with hud.trace("Grounded Agent Demo"):
        # Configure the grounding model
        grounder_config = GrounderConfig(
            api_base="https://openrouter.ai/api/v1",  # OpenRouter API
            model="qwen/qwen-2.5-vl-7b-instruct",    # Vision model for grounding
            api_key="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",
            system_prompt=(
                "You are a visual grounding model. Given an image and a description, "
                "return ONLY the center pixel coordinates of the described element as a single point "
                "in parentheses format: (x, y). Do not return bounding boxes or multiple coordinates."
            ),
            output_format="pixels",               # Direct pixel coordinates
            parser_regex=r"\((\d+),\s*(\d+)\)",  # Parse (x, y) format
        )
        
        # MCP configuration for environment
        # Option 1: Use local Docker browser
        mcp_config = {
            "local": {
                "command": "docker",
                "args": ["run", "--rm", "-i", "-p", "8080:8080", "hudevals/hud-browser:0.1.3"],
            }
        }
        
        # Option 2: Use HUD cloud with remote browser (uncomment to use)
        # mcp_config = {
        #     "hud": {
        #         "url": "https://mcp.hud.so/v3/mcp",
        #         "headers": {
        #             "Authorization": f"Bearer {settings.api_key}",
        #             "Mcp-Image": "hudpython/hud-remote-browser:latest",
        #         },
        #     }
        # }
        
        # Create OpenAI client for planning
        openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", settings.openai_api_key)
        )
        
        agent = GroundedOpenAIChatAgent(
            grounder_config=grounder_config,
            openai_client=openai_client,
            model_name="gpt-4o-mini",  # Planning model
            allowed_tools=["computer"], 
            append_setup_output=False,
            system_prompt=(
                "You are a helpful AI assistant that can control the computer through visual interaction.\n\n"
                "IMPORTANT: Always explain your reasoning and observations before taking actions:\n"
                "1. First, describe what you see on the screen\n"
                "2. Explain what you plan to do and why\n"
                "3. Then use the computer tool with natural language descriptions\n\n"
                "For example:\n"
                "- 'I can see a form with several input fields. I'll start by clicking on the name field which appears to be at the top of the form.'\n"
                "- 'The submit button is likely at the bottom of the form, I'll click on it now.'\n\n"
                "Remember:\n"
                "- Click on text fields before typing to make them active\n"
                "- Use descriptive element descriptions like 'blue submit button', 'search box at top', 'menu icon in corner'\n"
                "- If an element can't be found, try describing it differently"
            ),
        )
        agent.metadata = {}
        
        try:
            # Create a task with MCP config
            from hud.datasets import Task
            
            form_url = "https://hb.cran.dev/forms/post"
            
            form_prompt = f"""
            Fill out the form:
            1. Enter "Grounded Test" in the customer name field
            2. Enter "555-9876" in the telephone field
            3. Type "Testing grounded agent with separated vision and reasoning" in comments
            4. Select medium pizza size
            5. Choose mushroom as a topping
            6. Submit the form
            """
            
            task = Task(
                prompt=form_prompt,
                mcp_config=mcp_config,
                setup_tool={
                    "name": "playwright",
                    "arguments": {"action": "navigate", "url": form_url}
                }
            )
            
            print(f"📋 Task: Form interaction")
            print(f"🚀 Running grounded agent...\n")
            
            result = await agent.run(task, max_steps=10)
            print(f"Result: {result.content}\n")

        except Exception as e:
            print(f"Error during agent execution: {e}")
            
    
    print("\n✨ Grounded agent demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
