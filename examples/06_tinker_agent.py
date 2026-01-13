#!/usr/bin/env python3
"""
Tinker Agent Example

This example demonstrates how to use the TinkerAgent for:
- Computer use tasks with Tinker's sampling API
- Capturing completions for RL training
- Integration with HUD evaluation context

Required Environment Variables:
-------------------------------
TINKER_API_KEY     - API key for Tinker service (required)
HUD_API_KEY        - API key for HUD platform (required for tracing)

Optional Environment Variables:
-------------------------------
TINKER_BASE_URL    - Custom Tinker service URL (default: Tinker cloud)

Display Settings (optional):
----------------------------
TINKER_COMPUTER_WIDTH   - Display width (default: 700)
TINKER_COMPUTER_HEIGHT  - Display height (default: 448)

Usage:
------
1. Set environment variables:
   export TINKER_API_KEY="your-tinker-api-key"
   export HUD_API_KEY="your-hud-api-key"

2. Run the example:
   python examples/06_tinker_agent.py
"""

from __future__ import annotations

import asyncio
import os

import hud
from hud.eval.task import Task


async def main() -> None:
    # Check required environment variables
    tinker_api_key = os.getenv("TINKER_API_KEY")
    hud_api_key = os.getenv("HUD_API_KEY")

    if not tinker_api_key:
        print("❌ Error: TINKER_API_KEY environment variable is required")
        print("   Get your API key from: https://tinker-console.thinkingmachines.ai")
        return

    if not hud_api_key:
        print("⚠️  Warning: HUD_API_KEY not set - tracing will be disabled")

    # Import Tinker dependencies (requires tinker and tinker-cookbook packages)
    try:
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.image_processing_utils import get_image_processor
    except ImportError as e:
        print(f"❌ Error: Missing Tinker dependencies: {e}")
        print("   Install with: pip install tinker tinker-cookbook")
        return

    # Import TinkerAgent
    from hud.agents.tinker import TinkerAgent, TinkerCompletionStore

    print("🚀 Tinker Agent Demo")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # Step 1: Initialize Tinker components
    # -------------------------------------------------------------------------
    print("\n📡 Connecting to Tinker service...")

    tinker_base_url = os.getenv("TINKER_BASE_URL")
    model_name = "Qwen/Qwen3-VL-235B-A22B-Instruct"  # Vision-capable model
    renderer_name = "qwen3_vl_instruct"  # Use VL renderer for vision models
    lora_rank = 32

    try:
        service_client = tinker.ServiceClient(base_url=tinker_base_url)
        training_client = await service_client.create_lora_training_client_async(
            model_name,
            rank=lora_rank,
        )
        print(f"   ✓ Connected to model: {model_name}")

        # Get tokenizer, image processor, and renderer
        tokenizer = training_client.get_tokenizer()
        image_processor = get_image_processor(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer, image_processor)
        print(f"   ✓ Loaded renderer: {renderer_name}")

        # Get sampling client (creates initial checkpoint)
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        print("   ✓ Sampling client ready : {sampling_client}")

    except Exception as e:
        print(f"❌ Error connecting to Tinker: {e}")
        return

    # -------------------------------------------------------------------------
    # Step 2: Create TinkerAgent with completion store
    # -------------------------------------------------------------------------
    print("\n🤖 Creating TinkerAgent...")

    # Create a shared completion store for RL training
    # This captures all completions with prompts, tokens, and logprobs
    completion_store = TinkerCompletionStore()

    agent = TinkerAgent.create(
        model=model_name,
        sampling_client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
        completion_store=completion_store,
        temperature=0.7,
        max_new_tokens=1024,
        max_context_tokens=200000,
        system_prompt="You are a helpful AI assistant that can control a computer.",
    )

    print(f"   ✓ Agent created with model: {model_name}")
    print(f"   ✓ Completion store ready for RL training")

    # -------------------------------------------------------------------------
    # Step 3: Run agent on a simple task
    # -------------------------------------------------------------------------
    print("\n📋 Running task...")

    # Create task with MCP configuration for browser-based spreadsheet task
    task = Task(
        id="6e4744c7-b2c9-4bb6-807e-2cc144a4e8c2",
        prompt=(
            "Calculate from the RawData tab the z-scores from the mean close price "
            "for each row. Return, starting in ANSWER!A1 and descending to ANSWER!A5, "
            "the 5 dates with the greatest absolute value of standard deviations from the mean"
        ),
        mcp_config={
            "hud": {
                "url": "https://mcp.hud.ai/v3/mcp",
                "headers": {
                    "Authorization": "Bearer ${HUD_API_KEY}",
                    "Mcp-Image": "hudevals/hud-remote-browser:0.1.3",
                },
            }
        },
        system_prompt=(
            "All solutions should be put in the sheet called 'ANSWER'. "
            "In the answer sheet, all dates should use the American standard format "
            "MM/DD/YYYY with no leading zero. All numbers should use the format and "
            "decimal place precision given in the input sheets (e.g., with or without "
            "a thousands separator should depend on the inputs), unless specified otherwise."
        ),
        setup_tool={
            "name": "setup",
            "arguments": {
                "name": "sheets_from_xlsx",
                "arguments": {
                    "file_url": "https://gahludmjcsmszgyufydt.supabase.co//storage/v1/object/public/sheetbench/c6ddeb9a-0c16-4f5e-8a06-f148ebb4be8a/setup_input_2.xlsx?"
                },
            },
        },
        evaluate_tool={
            "name": "evaluate",
            "arguments": {
                "name": "sheets_cell_values",
                "arguments": {
                    "args": {
                        "A1": "1/12/2024",
                        "A2": "1/10/2024",
                        "A3": "1/15/2024",
                        "A4": "1/11/2024",
                        "A5": "1/17/2024",
                    }
                },
            },
        },
        metadata={
            "partial": True,
            "gold_file_url": "https://gahludmjcsmszgyufydt.supabase.co//storage/v1/object/public/sheetbench/c6ddeb9a-0c16-4f5e-8a06-f148ebb4be8a/gold_solution_2.xlsx?",
        },
        agent_config={
            "allowed_tools": ["computer"],
            "disallowed_tools": ["setup", "evaluate"],
            "initial_screenshot": True,
            "append_setup_output": True,
        },
    )

    try:
        async with hud.eval(task, name="tinker-demo", trace=bool(hud_api_key)) as ctx:
            result = await agent.run(ctx, max_steps=30)

        print(f"\n✨ Task completed!")
        print(f"   Content: {result.content}")
        print(f"   Done: {result.done}")
        print(f"   Error: {result.isError}")

    except Exception as e:
        print(f"❌ Error running task: {e}")
        raise

    # -------------------------------------------------------------------------
    # Step 4: Access completions for RL training
    # -------------------------------------------------------------------------
    print("\n📊 Completion Store Stats:")
    print(f"   Stored completions: {completion_store.size()}")

    # Pop completions to see what was captured
    record = completion_store.pop_next()
    if record:
        print(f"\n   Sample completion:")
        print(f"   - ID: {record.completion_id}")
        print(f"   - Tokens: {len(record.tokens)} tokens")
        print(f"   - Has logprobs: {record.logprobs is not None}")
        print(f"   - Finish reason: {record.finish_reason}")

    print("\n" + "=" * 50)
    print("Demo complete! 🎉")


async def run_computer_use_demo() -> None:
    """
    Advanced demo showing computer use with browser.

    This requires a browser environment and is more complex.
    Uncomment and run if you have the browser hub set up.
    """
    import os

    try:
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.image_processing_utils import get_image_processor
    except ImportError:
        print("Missing Tinker dependencies")
        return

    from hud.agents.tinker import TinkerAgent, TinkerCompletionStore

    # Initialize Tinker
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    service_client = tinker.ServiceClient(base_url=os.getenv("TINKER_BASE_URL"))
    training_client = await service_client.create_lora_training_client_async(
        model_name,
        rank=32,
    )
    tokenizer = training_client.get_tokenizer()
    image_processor = get_image_processor(model_name)
    renderer = renderers.get_renderer("qwen3_vl_instruct", tokenizer, image_processor)
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    completion_store = TinkerCompletionStore()
    agent = TinkerAgent.create(
        sampling_client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
        completion_store=completion_store,
        temperature=0.7,
        max_new_tokens=1024,
    )

    # Browser task
    task = Task(
        env={"name": "browser"},
        scenario="navigation",
        args={"url": "https://example.com"},
        agent_config={"system_prompt": "Navigate to example.com and describe what you see."},
    )

    async with hud.eval(task, name="tinker-browser-demo") as ctx:
        result = await agent.run(ctx, max_steps=10)

    print(f"Result: {result.content}")
    print(f"Completions captured: {completion_store.size()}")


if __name__ == "__main__":
    asyncio.run(main())
