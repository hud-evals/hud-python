#!/usr/bin/env python3
"""
Environment Analysis Example

This example demonstrates how to use the new analyze_environment()
method to get comprehensive information about an MCP environment.

You can also use the CLI directly:
  hud analyze environments/text_2048/Dockerfile
  hud analyze --format json config.json
"""

import asyncio
import json
from pathlib import Path
from hud.clients import MCPClient
from rich import print as rprint
from rich.tree import Tree
from rich.table import Table


async def main():
    # Example 1: Analyze Text 2048 environment
    print("\n🔍 Analyzing Text 2048 Environment\n")

    # Using direct Python execution (no Docker)
    text_2048 = Path(__file__).parent.parent / "environments" / "text_2048"
    mcp_config = {
        "local": {
            "command": "python",
            "args": ["-m", "hud_controller.server"],
            "env": {"PYTHONPATH": str(text_2048 / "src")},
            "cwd": str(text_2048),
        }
    }

    client = MCPClient(mcp_config=mcp_config)

    try:
        # Initialize and analyze
        await client.initialize()
        analysis = await client.analyze_environment()

        # Display results using Rich
        display_analysis(analysis)

        # Example 2: Get hub tools specifically
        print("\n📦 Hub Tools Detail\n")

        for hub_name in ["setup", "evaluate"]:
            functions = await client.get_hub_tools(hub_name)
            if functions:
                print(f"{hub_name} hub functions: {', '.join(functions)}")

    finally:
        await client.close()


def display_analysis(analysis: dict):
    """Display analysis results using Rich formatting."""

    # Create a tree for tools
    tools_tree = Tree("🔧 Available Tools")

    # Regular tools
    regular_branch = tools_tree.add("Regular Tools")
    for tool in analysis["tools"]:
        if tool["name"] not in analysis["hub_tools"]:
            tool_branch = regular_branch.add(f"[cyan]{tool['name']}[/cyan]")
            if tool.get("description"):
                tool_branch.add(f"[dim]{tool['description']}[/dim]")

    # Hub tools
    if analysis["hub_tools"]:
        hub_branch = tools_tree.add("Hub Tools")
        for hub_name, functions in analysis["hub_tools"].items():
            hub_node = hub_branch.add(f"[yellow]{hub_name}[/yellow]")
            for func in functions:
                hub_node.add(f"[green]{func}[/green]")

    rprint(tools_tree)

    # Resources table
    if analysis["resources"]:
        print("\n📚 Available Resources\n")

        table = Table()
        table.add_column("URI", style="cyan")
        table.add_column("Name")
        table.add_column("Type", style="dim")

        for resource in analysis["resources"][:5]:  # Show first 5
            table.add_row(resource["uri"], resource.get("name", ""), resource.get("mime_type", ""))

        rprint(table)

        if len(analysis["resources"]) > 5:
            print(f"\n[dim]... and {len(analysis['resources']) - 5} more resources[/dim]")

    # Telemetry info
    if analysis["telemetry"]:
        print("\n📡 Telemetry Data\n")
        for key, value in analysis["telemetry"].items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
