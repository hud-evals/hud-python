#!/usr/bin/env python3
"""
SheetBench Agent Example with Enhanced Telemetry
"""

import asyncio
import logging
import time
import json
import hud
from hud.mcp import ClaudeMCPAgent
from hud.mcp.client import MCPClient
from datasets import load_dataset
from hud.datasets import to_taskconfigs

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also log MCP client actions
logging.getLogger("hud.mcp.client").setLevel(logging.DEBUG)
logging.getLogger("hud.mcp.base").setLevel(logging.DEBUG)
logging.getLogger("hud.mcp.claude").setLevel(logging.DEBUG)


async def main():
    # Load the dataset
    logger.info("Loading SheetBench dataset...")
    dataset = load_dataset("hud-evals/sheetbench-taskconfigs")
    
    with hud.trace("SheetBench Agent with Telemetry"):
        tsx = to_taskconfigs(dataset["train"])
        task = tsx[0]
        
        # Log task details
        logger.info("=" * 60)
        logger.info("TASK CONFIGURATION:")
        logger.info(f"  Task ID: {task.id}")
        logger.info(f"  Prompt: '{task.prompt}'")
        logger.info(f"  Has setup_tool: {task.setup_tool is not None}")
        if task.setup_tool:
            # Handle non-serializable types
            try:
                if hasattr(task.setup_tool, '__dict__'):
                    setup_tool_dict = task.setup_tool.__dict__
                elif hasattr(task.setup_tool, 'model_dump'):
                    setup_tool_dict = task.setup_tool.model_dump()
                else:
                    setup_tool_dict = str(task.setup_tool)
                logger.info(f"  Setup tool: {setup_tool_dict}")
            except Exception as e:
                logger.info(f"  Setup tool: {task.setup_tool} (could not serialize: {e})")
        logger.info(f"  Has evaluate_tool: {task.evaluate_tool is not None}")
        if task.evaluate_tool:
            try:
                if hasattr(task.evaluate_tool, '__dict__'):
                    eval_tool_dict = task.evaluate_tool.__dict__
                elif hasattr(task.evaluate_tool, 'model_dump'):
                    eval_tool_dict = task.evaluate_tool.model_dump()
                else:
                    eval_tool_dict = str(task.evaluate_tool)
                logger.info(f"  Evaluate tool: {eval_tool_dict}")
            except Exception as e:
                logger.info(f"  Evaluate tool: {task.evaluate_tool} (could not serialize: {e})")
        logger.info("=" * 60)
        
        # Create client with verbose mode
        client = MCPClient(mcp_config=task.mcp_config, verbose=True)
        
        # Create agent
        agent = ClaudeMCPAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )
        
        try:
            # Initialize and check tools
            logger.info("\n🔧 Initializing agent...")
            start_time = time.time()
            await agent.initialize(task)
            init_time = time.time() - start_time
            logger.info(f"✅ Agent initialized in {init_time:.2f} seconds")
            
            # Log available tools
            logger.info("\n📋 Available tools after filtering:")
            for tool in agent._available_tools:
                logger.info(f"  - {tool.name}: {tool.description[:50]}...")
            
            # Check for anthropic_computer specifically
            has_computer = any(t.name == "anthropic_computer" for t in agent._available_tools)
            logger.info(f"\n🖥️ Has anthropic_computer tool: {has_computer}")
            
            # Fetch initial telemetry
            logger.info("\n📡 Fetching initial telemetry...")
            telemetry = await client.fetch_telemetry()
            if telemetry:
                logger.info("Telemetry data:")
                for server, data in telemetry.items():
                    logger.info(f"  Server '{server}': {json.dumps(data, indent=2)}")
            
            # Run the task with timing
            logger.info("\n🚀 Starting task execution...")
            logger.info(f"Task prompt being sent: '{task.prompt}'")
            logger.info(f"Max steps: 15")
            
            run_start = time.time()
            
            # Add detailed logging by monkey-patching
            original_get_model_response = agent.get_model_response
            async def logged_get_model_response(messages):
                logger.info(f"\n📤 Sending {len(messages)} messages to model...")
                logger.info(f"Last message type: {type(messages[-1]) if messages else 'No messages'}")
                
                start = time.time()
                try:
                    response = await original_get_model_response(messages)
                    elapsed = time.time() - start
                    logger.info(f"📥 Model responded in {elapsed:.2f}s")
                    logger.info(f"  Response content: {response.content[:200] if response.content else 'No content'}...")
                    logger.info(f"  Tool calls: {[tc.name for tc in response.tool_calls] if response.tool_calls else 'None'}")
                    logger.info(f"  Done flag: {response.done}")
                    return response
                except Exception as e:
                    elapsed = time.time() - start
                    logger.error(f"❌ Model response failed after {elapsed:.2f}s: {e}")
                    raise
            
            agent.get_model_response = logged_get_model_response
            
            # Also patch call_tool to monitor tool execution
            original_call_tool = agent.call_tool
            async def logged_call_tool(tool_call):
                tool_name = tool_call.name if hasattr(tool_call, 'name') else str(tool_call)
                logger.info(f"\n🔧 Executing tool: {tool_name}")
                start = time.time()
                try:
                    result = await original_call_tool(tool_call)
                    elapsed = time.time() - start
                    logger.info(f"✅ Tool {tool_name} completed in {elapsed:.2f}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start
                    logger.error(f"❌ Tool {tool_name} failed after {elapsed:.2f}s: {e}")
                    raise
                    
            agent.call_tool = logged_call_tool
            
            # Run the task
            result = await agent.run(task, max_steps=15)
            
            run_time = time.time() - run_start
            
            # Log results
            logger.info("\n📊 TASK RESULTS:")
            logger.info("=" * 60)
            logger.info(f"  Execution time: {run_time:.2f} seconds")
            logger.info(f"  Reward: {result.reward}")
            logger.info(f"  Done: {result.done}")
            logger.info(f"  Error: {result.error}")
            if result.content:
                logger.info(f"  Content preview: {result.content[:200]}...")
            if result.info:
                logger.info(f"  Info: {json.dumps(result.info, indent=2)}")
            logger.info("=" * 60)
            
            # Fetch final telemetry
            logger.info("\n📡 Fetching final telemetry...")
            final_telemetry = await client.fetch_telemetry()
            if final_telemetry:
                logger.info("Final telemetry data:")
                for server, data in final_telemetry.items():
                    logger.info(f"  Server '{server}': {json.dumps(data, indent=2)}")
            
        except Exception as e:
            logger.error(f"❌ Error during execution: {e}", exc_info=True)
            raise
        finally:
            logger.info("\n🔚 Closing client...")
            await client.close()
    
    logger.info("\n✨ SheetBench agent demo complete!")


if __name__ == "__main__":
    asyncio.run(main())