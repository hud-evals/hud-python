#!/usr/bin/env python3
"""
Test suite for the text_2048 MCP environment.

This test file:
1. Builds the Docker image
2. Tests MCP initialization
3. Tests game-specific tools with HUD MCPClient
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add parent directory to path to import hud
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hud.clients import MCPClient


class TestText2048Environment:
    """Test suite for text_2048 MCP environment."""

    IMAGE_NAME = "hud-text-2048-test:latest"
    BUILD_TIMEOUT = 300  # 5 minutes for build

    @classmethod
    def setup_class(cls):
        """Build the Docker image before running tests."""
        print("\n🔨 Building text_2048 environment Docker image...")
        start_time = time.time()

        # Get the directory containing this test file
        test_dir = Path(__file__).parent
        dockerfile_dir = test_dir

        # Build the image
        cmd = ["docker", "build", "-t", cls.IMAGE_NAME, str(dockerfile_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=cls.BUILD_TIMEOUT)

        build_time = time.time() - start_time

        if result.returncode != 0:
            print(f"❌ Docker build failed in {build_time:.1f}s")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"Failed to build Docker image: {result.stderr}")

        print(f"✅ Docker image built successfully in {build_time:.1f}s")

    def test_phase1_basic_startup(self):
        """Phase 1: Test that the Docker container can start."""
        print("\n📋 Phase 1: Basic Server Startup Test")

        # Test if docker command works
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)

        assert result.returncode == 0, f"Docker not available: {result.stderr}"
        print("✅ Docker command available")

        # Test if our image exists
        result = subprocess.run(
            ["docker", "images", "-q", self.IMAGE_NAME], capture_output=True, text=True, timeout=5
        )

        assert result.stdout.strip(), f"Docker image {self.IMAGE_NAME} not found"
        print(f"✅ Docker image {self.IMAGE_NAME} exists")

    @pytest.mark.asyncio
    async def test_phase2_3_mcp_initialize_and_tools(self):
        """Phase 2-3: Test MCP initialization and tool discovery."""
        print("\n📋 Phase 2-3: MCP Initialize and Tool Discovery")

        # Create MCP config
        mcp_config = {
            "text-2048-test": {"command": "docker", "args": ["run", "--rm", "-i", self.IMAGE_NAME]}
        }

        # Create and initialize client
        client = MCPClient(mcp_config=mcp_config, verbose=False)

        try:
            await client.initialize()
            print("✅ MCPClient connected")

            # Wait for server to be ready
            await asyncio.sleep(3)

            # Get tools
            tools = client.get_available_tools()
            tool_names = [t.name for t in tools]

            print(f"✅ Found {len(tools)} tools")
            print(f"   Tools: {', '.join(tool_names)}")

            # Check for expected tools
            expected_tools = ["setup", "evaluate", "move"]
            for tool in expected_tools:
                assert tool in tool_names, f"Missing expected tool: {tool}"

            # Get resources
            resources = await client.list_resources()
            resource_uris = [str(r.uri) for r in resources]
            print(f"✅ Found {len(resources)} resources")
            print(f"   Resources: {', '.join(resource_uris)}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_game_tools(self):
        """Test game-specific tools."""
        print("\n📋 Testing Game-Specific Tools")

        mcp_config = {
            "text-2048-test": {"command": "docker", "args": ["run", "--rm", "-i", self.IMAGE_NAME]}
        }

        client = MCPClient(mcp_config=mcp_config, verbose=False)

        try:
            await client.initialize()
            await asyncio.sleep(3)

            # Test 1: Setup board
            print("\n1. Testing setup board...")
            result = await client.call_tool(
                "setup", {"name": "board", "arguments": {"board_size": 4}}
            )

            # Check for args serialization issue
            if result.isError and "Parameter 'args' must be one of types" in str(result.content):
                print("⚠️  Known issue: Cursor MCP integration serializes args as string")
                pytest.skip("Skipping due to known Cursor MCP args serialization issue")

            assert not result.isError, f"Setup board failed: {result.content}"
            print("✅ Board setup successful")

            # Test 2: Make a move
            print("\n2. Testing move...")
            result = await client.call_tool("move", {"direction": "up"})
            assert not result.isError, f"Move failed: {result.content}"
            print("✅ Move successful")

            # Test 3: Evaluate max number
            print("\n3. Testing evaluate max_number...")
            result = await client.call_tool(
                "evaluate", {"name": "max_number", "arguments": {"target": 64}}
            )

            if result.isError and "Parameter 'args' must be one of types" in str(result.content):
                print("⚠️  Known issue: Cursor MCP integration serializes args as string")
            else:
                assert not result.isError, f"Evaluate failed: {result.content}"
                print("✅ Evaluate successful")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_game_flow(self):
        """Test a complete game flow."""
        print("\n📋 Testing Complete Game Flow")

        mcp_config = {
            "text-2048-test": {"command": "docker", "args": ["run", "--rm", "-i", self.IMAGE_NAME]}
        }

        client = MCPClient(mcp_config=mcp_config, verbose=False)

        try:
            await client.initialize()
            await asyncio.sleep(3)

            # Initialize board
            print("\n1. Initializing 4x4 board...")
            result = await client.call_tool(
                "setup", {"name": "board", "arguments": {"board_size": 4}}
            )

            if result.isError and "Parameter 'args' must be one of types" in str(result.content):
                print("⚠️  Skipping game flow test due to args serialization issue")
                return

            # Make several moves
            moves = ["up", "right", "down", "left"]
            for i, direction in enumerate(moves):
                print(f"\n2.{i + 1}. Moving {direction}...")
                result = await client.call_tool("move", {"direction": direction})
                assert not result.isError, f"Move {direction} failed: {result.content}"
                print(f"✅ Move {direction} successful")

            # Check efficiency
            print("\n3. Checking game efficiency...")
            result = await client.call_tool(
                "evaluate", {"name": "efficiency", "arguments": {"min_ratio": 10.0}}
            )

            if not result.isError:
                print("✅ Efficiency check successful")

        finally:
            await client.close()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
