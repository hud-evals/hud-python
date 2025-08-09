#!/usr/bin/env python3
"""
MCP Resources Exploration

This example shows how to discover and explore MCP resources:
- List available resources in an environment
- Understand setup tools and evaluators
- Explore available problems/scenarios
- Access telemetry and debugging info

Resources provide introspection into what an MCP server offers
without having to call tools directly.
"""

import asyncio
import json
import hud
from hud.mcp.client import MCPClient


async def explore_resources(client):
    """Explore all resources available in the MCP server."""

    print("\n🔍 Discovering MCP Resources...")
    print("-" * 50)

    all_resources = []

    # Get resources from each session
    for server_name, session in client._sessions.items():
        try:
            if hasattr(session.connector, "client_session") and session.connector.client_session:
                resources = await session.connector.client_session.list_resources()
                print(f"\n📚 Server '{server_name}' resources: {len(resources.resources)}")

                # Group resources by type
                resource_groups = {}
                for resource in resources.resources:
                    # Extract resource type from URI
                    uri = str(resource.uri)
                    resource_type = uri.split("://")[0] if "://" in uri else "other"

                    if resource_type not in resource_groups:
                        resource_groups[resource_type] = []
                    resource_groups[resource_type].append(resource)

                # Display grouped resources
                for group_name, items in resource_groups.items():
                    print(f"\n   📁 {group_name.upper()} ({len(items)} items):")
                    for item in items[:5]:  # Show first 5
                        print(f"      - {item.name}: {item.uri}")
                    if len(items) > 5:
                        print(f"      ... and {len(items) - 5} more")

                all_resources.extend(resources.resources)

        except Exception as e:
            print(f"   ❌ Error listing resources from {server_name}: {e}")

    return all_resources


async def explore_specific_resource(client, resource_uri):
    """Get detailed information about a specific resource."""

    print(f"\n🔎 Exploring resource: {resource_uri}")
    print("-" * 40)

    try:
        from pydantic import AnyUrl

        # Read the resource content using the new client
        result = await client.read_resource(AnyUrl(resource_uri))

        if result and result.contents:
            # Parse the first content item
            content = result.contents[0]

            if hasattr(content, "text"):
                try:
                    # Try to parse as JSON for pretty printing
                    data = json.loads(content.text)
                    print(json.dumps(data, indent=2)[:500] + "...")
                except json.JSONDecodeError:
                    # If not JSON, just print the text
                    print(content.text[:500] + "...")
            else:
                print(f"Content type: {type(content)}")
                print(str(content)[:500] + "...")
        else:
            print("No content found for this resource")

    except Exception as e:
        print(f"❌ Error reading resource: {e}")


async def main():
    print("🚀 MCP Resources Exploration")
    print("=" * 50)

    with hud.trace("Resources Exploration"):
        # Configure for local Docker environment
        mcp_config = {
            "browser": {
                "command": "docker",
                "args": [
                    "run",
                    "--rm",
                    "-i",
                    "-p",
                    "8080:8080",
                    "-e",
                    "LAUNCH_APPS=todo",
                    "hud-browser",
                ],
            }
        }

        # Create client using the new pattern
        client = MCPClient(mcp_config=mcp_config)

        try:
            print("\n⏳ Starting MCP server...")
            print("   View browser at: http://localhost:8080/vnc.html")

            # Initialize the client (this connects to the MCP server)
            await client.initialize()

            # Give server time to start
            await asyncio.sleep(2)

            # Explore all available resources
            resources = await explore_resources(client)

            # Explore some specific resources
            interesting_resources = [
                "setup://registry",
                "evaluators://registry",
                "problems://registry",
                "telemetry://current",
            ]

            for uri in interesting_resources:
                # Check if resource exists in our list
                if any(str(r.uri) == uri for r in resources):
                    await explore_specific_resource(client, uri)
                else:
                    print(f"\n⚠️  Resource '{uri}' not found")

            # Interactive exploration
            print("\n💡 Interactive Mode")
            print("Enter a resource URI to explore (or 'quit' to exit):")

            while True:
                uri = input("> ").strip()
                if uri.lower() in ["quit", "exit", "q"]:
                    break

                if uri:
                    await explore_specific_resource(client, uri)

        finally:
            await client.close()

    print("\n✨ Resources exploration complete!")
    print("\n📚 What we learned:")
    print("   - Resources provide metadata about MCP capabilities")
    print("   - Setup tools prepare environments for tasks")
    print("   - Evaluators check task completion")
    print("   - Telemetry provides debugging information")


if __name__ == "__main__":
    asyncio.run(main())
