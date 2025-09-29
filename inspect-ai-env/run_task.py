#!/usr/bin/env python3
"""
Single Sample Processing with HUD Environment

This script processes ONE sample at a time through your custom HUD environment
with setup/solver/scorer pipeline. Each sample gets its own container instance
and the dataset is processed in parallel across multiple containers.
"""

from __future__ import annotations

import asyncio
import json
import hud
import sys
from pathlib import Path

from hud.clients import MCPClient
from hud.datasets import Task
from hud.agents import ClaudeAgent, OperatorAgent, GenericOpenAIChatAgent
from hud.agents.base import find_reward, find_content


def get_agent_from_config(task_data: dict, client: MCPClient):
    """Create the appropriate agent based on task configuration"""
    sample_processing = task_data.get('sample_processing', {})
    agent_config = sample_processing.get('agent_config', {})
    agent_type = agent_config.get('type', 'claude')

    if agent_type == 'claude':
        return ClaudeAgent(
            mcp_client=client,
            model=agent_config.get('model', 'claude-3-5-sonnet-20241022'),
            initial_screenshot=agent_config.get('initial_screenshot', False),
            allowed_tools=agent_config.get('allowed_tools'),
            disallowed_tools=agent_config.get('disallowed_tools'),
        )
    elif agent_type == 'openai':
        return OperatorAgent(
            mcp_client=client,
            model=agent_config.get('model', 'gpt-4'),
            initial_screenshot=agent_config.get('initial_screenshot', False),
            allowed_tools=agent_config.get('allowed_tools'),
            disallowed_tools=agent_config.get('disallowed_tools'),
        )
    elif agent_type == 'generic_openai':
        return GenericOpenAIChatAgent(
            mcp_client=client,
            model=agent_config.get('model', 'gpt-4'),
            allowed_tools=agent_config.get('allowed_tools'),
            disallowed_tools=agent_config.get('disallowed_tools'),
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def process_single_sample(sample_data: dict, task_data: dict) -> dict:
    """
    Process a single sample through the setup -> solver -> scorer pipeline.
    This is the core function that gets called once per container instance.
    """
    with hud.trace("Single Sample Processing"):
        task = Task(**task_data)

        # Create MCP client
        client = MCPClient(mcp_config=task.mcp_config)

        # Create agent based on configuration
        agent = get_agent_from_config(task_data, client)

        sample_id = sample_data.get('id', 'unknown_sample')

        try:
            print(f"🔧 Initializing agent for sample: {sample_id}")
            await agent.initialize(task)

            # Phase 1: Setup
            print("📋 Running setup...")
            setup_result = await agent.call_tools(task.setup_tool)
            setup_content = setup_result[0].content
            print(f"✅ Setup complete: {setup_content}")

            # Phase 2: Process the single sample
            sample_processing = task_data.get('sample_processing', {})
            task_config = sample_processing.get('task_config', {})
            eval_spec = sample_processing.get('eval_spec', {})

            print(f"\n🔄 Processing sample {sample_id}")
            prompt = sample_data.get('prompt', '')
            print(f"   Prompt: {str(prompt)[:100]}...")

            # Process the sample through your environment
            from hud.datasets import ToolCall
            tool_call = ToolCall(
                name="process_sample",
                arguments={
                    "sample_data": sample_data,
                    "task_config": task_config,
                    "eval_spec": eval_spec
                }
            )
            result = await agent.call_tools(tool_call)

            if result[0].isError:
                print(f"❌ Sample processing failed: {result[0].content}")
                return {
                    "sample_id": sample_id,
                    "success": False,
                    "error": result[0].content
                }

            # Parse the processing result
            sample_result = json.loads(result[0].content)
            success = sample_result.get('success', False)
            score = sample_result.get('score', {})
            processing_time = sample_result.get('processing_time', 0)

            print(f"✅ Sample processed successfully")
            print(f"   Success: {success}")
            print(f"   Score: {score}")
            print(f"   Processing time: {processing_time:.3f}s")

            return {
                "sample_id": sample_id,
                "success": success,
                "score": score,
                "processing_time": processing_time,
                "setup_output": sample_result.get('setup_output'),
                "solver_output": sample_result.get('solver_output'),
                "timestamp": sample_result.get('timestamp')
            }

        except Exception as e:
            print(f"❌ Exception processing sample {sample_id}: {e}")
            return {
                "sample_id": sample_id,
                "success": False,
                "error": str(e)
            }
        finally:
            print("🧹 Cleaning up...")
            await client.shutdown()


def load_sample_by_id(sample_id: str, samples_file: str = "samples.jsonl") -> dict:
    """Load a specific sample by ID from the JSONL file."""
    try:
        with open(samples_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    if str(sample.get('id')) == str(sample_id):
                        return sample
        raise ValueError(f"Sample with ID '{sample_id}' not found in {samples_file}")
    except FileNotFoundError:
        raise ValueError(f"Samples file '{samples_file}' not found")


async def main():
    """
    Main function for single sample processing.

    Usage:
    python run_task.py <sample_id>
    """
    import argparse

    parser = argparse.ArgumentParser(description="Process a single sample by ID")
    parser.add_argument("sample_id", help="Sample ID to process")
    parser.add_argument("--config", default="tasks.json", help="Task configuration file")
    parser.add_argument("--samples", default="samples.jsonl", help="Samples JSONL file")
    parser.add_argument("--output", help="Output file for results (default: stdout)")

    args = parser.parse_args()

    # Load task configuration
    with open(args.config) as f:
        tasks = json.load(f)

    if len(tasks) != 1:
        print("❌ Task configuration must contain exactly one task for single sample processing")
        sys.exit(1)

    task_data = tasks[0]

    # Load the specific sample by ID
    try:
        sample_data = load_sample_by_id(args.sample_id, args.samples)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"🎯 Processing single sample: {sample_data.get('id', 'unknown')}")
    print("=" * 60)

    # Process the sample
    result = await process_single_sample(sample_data, task_data)

    # Output result
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n📄 Results saved to {args.output}")
    else:
        print("\n📊 Final Result:")
        print(json.dumps(result, indent=2))

    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    print("🚀 Single Sample Processing with HUD Environment")
    print("=" * 50)
    asyncio.run(main())
