"""Interactive mode for testing MCP environments."""

from __future__ import annotations

import json
from typing import Any

import questionary
from mcp.types import ImageContent, TextContent
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.tree import Tree

from hud.clients import MCPClient
from hud.utils.hud_console import HUDConsole

console = Console()


class InteractiveMCPTester:
    """Interactive MCP environment tester."""

    def __init__(self, server_url: str, verbose: bool = False) -> None:
        """Initialize the interactive tester.

        Args:
            server_url: URL of the MCP server (e.g., http://localhost:8765/mcp)
            verbose: Enable verbose output
        """
        self.server_url = server_url
        self.verbose = verbose
        self.client: MCPClient | None = None
        self.tools: list[Any] = []
        self.console = HUDConsole()

    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            # Create MCP config for HTTP transport
            # Note: We explicitly set auth to None to prevent OAuth discovery attempts
            config = {"server": {"url": self.server_url, "auth": None}}

            self.client = MCPClient(
                mcp_config=config,
                verbose=self.verbose,
                auto_trace=False,  # Disable telemetry for interactive testing
            )
            await self.client.initialize()

            # Fetch available tools
            self.tools = await self.client.list_tools()

            return True
        except Exception as e:
            self.console.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.client:
            await self.client.shutdown()
            self.client = None

    def display_tools(self) -> None:
        """Display available tools in a nice format."""
        if not self.tools:
            console.print("[yellow]No tools available[/yellow]")
            return

        # Group tools by hub
        regular_tools = []
        hub_tools = {}

        for tool in self.tools:
            if "/" in tool.name:
                hub, _ = tool.name.split("/", 1)
                if hub not in hub_tools:
                    hub_tools[hub] = []
                hub_tools[hub].append(tool)
            else:
                regular_tools.append(tool)

        # Display tools tree
        tree = Tree("🔧 Available Tools")

        if regular_tools:
            regular_node = tree.add("[cyan]Regular Tools[/cyan]")
            for i, tool in enumerate(regular_tools, 1):
                tool_node = regular_node.add(f"{i}. [white]{tool.name}[/white]")
                if tool.description:
                    tool_node.add(f"[dim]{tool.description}[/dim]")

        # Add hub tools
        tool_index = len(regular_tools) + 1
        for hub_name, tools in hub_tools.items():
            hub_node = tree.add(f"[yellow]{hub_name} Hub[/yellow]")
            for tool in tools:
                tool_node = hub_node.add(f"{tool_index}. [white]{tool.name}[/white]")
                if tool.description:
                    tool_node.add(f"[dim]{tool.description}[/dim]")
                tool_index += 1

        console.print(tree)

    async def select_tool(self) -> Any | None:
        """Let user select a tool."""
        if not self.tools:
            return None

        # Build choices list
        choices = []
        tool_map = {}

        # Group tools by hub for better organization
        hub_groups = {}
        regular_tools = []

        for tool in self.tools:
            if "/" in tool.name:
                hub, name = tool.name.split("/", 1)
                if hub not in hub_groups:
                    hub_groups[hub] = []
                hub_groups[hub].append((name, tool))
            else:
                regular_tools.append(tool)

        # Add regular tools first
        if regular_tools:
            # Add a separator for regular tools section
            if len(hub_groups) > 0:
                choices.append("───── Regular Tools ─────")

            for tool in regular_tools:
                # Format: Bold tool name with color + dim description
                if tool.description:
                    display = f"• {tool.name} │ {tool.description}"
                else:
                    display = f"• {tool.name}"

                choices.append(display)
                tool_map[display] = tool

        # Add hub-grouped tools with visual separation
        for hub_name, tools in sorted(hub_groups.items()):
            # Add a visual separator for each hub
            choices.append(f"───── {hub_name} ─────")

            for name, tool in sorted(tools, key=lambda x: x[0]):
                # Format with hub indicator and better separation
                if tool.description:
                    # Remove redundant description text
                    desc = tool.description
                    # Truncate long descriptions
                    if len(desc) > 60:
                        desc = desc[:57] + "..."
                    display = f"• {name} │ {desc}"
                else:
                    display = f"• {name}"

                choices.append(display)
                tool_map[display] = tool

        # Add quit option with spacing
        choices.append("─────────────────────")
        choices.append("❌ Quit")

        # Show selection menu with arrow keys
        console.print("\n[cyan]Select a tool (use arrow keys):[/cyan]")

        try:
            # Create custom Choice objects for better formatting
            from questionary import Choice

            formatted_choices = []
            for choice in choices:
                if choice.startswith("─────"):
                    # Separator - make it unselectable and styled
                    formatted_choices.append(Choice(title=choice, disabled=True, shortcut_key=None))  # type: ignore[arg-type]
                elif choice == "❌ Quit":
                    formatted_choices.append(choice)
                else:
                    formatted_choices.append(choice)

            # Use questionary's async select with enhanced styling
            selected = await questionary.select(
                "",
                choices=formatted_choices,
                style=questionary.Style(
                    [
                        ("question", ""),
                        ("pointer", "fg:#ff9d00 bold"),  # Orange pointer
                        ("highlighted", "fg:#00d7ff bold"),  # Bright cyan for highlighted
                        ("selected", "fg:#00ff00 bold"),  # Green for selected
                        ("separator", "fg:#666666"),  # Gray for separators
                        ("instruction", "fg:#858585 italic"),  # Dim instructions
                        ("disabled", "fg:#666666"),  # Gray for disabled items
                        ("text", "fg:#ffffff"),  # White text
                    ]
                ),
                instruction="(Use ↑/↓ arrows, Enter to select, Esc to cancel)",
            ).unsafe_ask_async()

            if selected is None:
                console.print("[yellow]No selection made (ESC or Ctrl+C pressed)[/yellow]")
                return None

            if selected == "❌ Quit" or selected.startswith("─────"):
                return None

            return tool_map.get(selected)

        except KeyboardInterrupt:
            console.print("[yellow]Interrupted by user[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]Error in tool selection: {e}[/red]")
            return None

    async def get_tool_arguments(self, tool: Any) -> dict[str, Any] | None:
        """Prompt user for tool arguments."""
        if not hasattr(tool, "inputSchema") or not tool.inputSchema:
            return {}

        schema = tool.inputSchema

        # Show schema
        console.print("\n[yellow]Tool Parameters:[/yellow]")
        schema_str = json.dumps(schema, indent=2)
        syntax = Syntax(schema_str, "json", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, title=f"{tool.name} Schema", border_style="dim"))

        # Handle different schema types
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            if not properties:
                return {}

            # Prompt for each property
            args = {}
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type")
                if not prop_type and "anyOf" in prop_schema:
                    prop_type = next(
                        (
                            s.get("type")
                            for s in prop_schema.get("anyOf", [])
                            if s.get("type") != "null"
                        ),
                        None,
                    )
                if not prop_type and "oneOf" in prop_schema:
                    prop_type = next(
                        (
                            s.get("type")
                            for s in prop_schema.get("oneOf", [])
                            if s.get("type") != "null"
                        ),
                        None,
                    )
                prop_type = prop_type or "string"

                description = prop_schema.get("description", "")
                is_required = prop_name in required

                # Build prompt
                prompt = f"{prop_name}"
                if description:
                    prompt += f" ({description})"
                if not is_required:
                    prompt += " [optional]"

                # Get value based on type
                if prop_type == "boolean":
                    if is_required:
                        value = await questionary.confirm(prompt).unsafe_ask_async()
                    else:
                        # For optional booleans, offer a choice
                        choice = await questionary.select(
                            prompt, choices=["true", "false", "skip (leave unset)"]
                        ).unsafe_ask_async()
                        if choice == "skip (leave unset)":
                            continue
                        value = choice == "true"
                elif prop_type == "number" or prop_type == "integer":
                    value_str = await questionary.text(
                        prompt,
                        default="",
                        validate=lambda text, pt=prop_type, req=is_required: True
                        if not text and not req
                        else (
                            text.replace("-", "").replace(".", "").isdigit()
                            if pt == "number"
                            else text.replace("-", "").isdigit()
                        )
                        or f"Please enter a valid {pt}",
                    ).unsafe_ask_async()
                    if not value_str and not is_required:
                        continue
                    value = int(value_str) if prop_type == "integer" else float(value_str)
                elif prop_type == "array":
                    value_str = await questionary.text(
                        prompt + " (comma-separated)", default=""
                    ).unsafe_ask_async()
                    if not value_str and not is_required:
                        continue
                    value = [v.strip() for v in value_str.split(",")]
                elif prop_type == "object":
                    # For object types, allow JSON input
                    console.print(f"[dim]Enter JSON object for {prop_name}:[/dim]")
                    value_str = await questionary.text(
                        prompt + " (JSON format)", default="{}"
                    ).unsafe_ask_async()
                    if not value_str and not is_required:
                        continue
                    try:
                        value = json.loads(value_str)
                    except json.JSONDecodeError as e:
                        console.print(f"[red]Invalid JSON: {e}[/red]")
                        # Try again
                        value_str = await questionary.text(
                            prompt + " (JSON format, please fix the error)", default=value_str
                        ).unsafe_ask_async()
                        try:
                            value = json.loads(value_str)
                        except json.JSONDecodeError:
                            console.print("[red]Still invalid JSON, using empty object[/red]")
                            value = {}
                else:  # string or unknown
                    value = await questionary.text(prompt, default="").unsafe_ask_async()
                    if not value and not is_required:
                        continue

                args[prop_name] = value

            return args
        else:
            # For non-object schemas, just get a single value
            console.print("[yellow]Enter value (or press Enter to skip):[/yellow]")
            value = Prompt.ask("Value", default="")
            return {"value": value} if value else {}

    async def call_tool(self, tool: Any, arguments: dict[str, Any]) -> None:
        """Call a tool and display results."""
        if not self.client:
            return

        try:
            # Show what we're calling
            console.print(f"\n[cyan]Calling {tool.name}...[/cyan]")
            if arguments:
                console.print(f"[dim]Arguments: {json.dumps(arguments, indent=2)}[/dim]")

            # Make the call
            result = await self.client.call_tool(name=tool.name, arguments=arguments)

            # Display results
            console.print("\n[green]✓ Tool executed successfully[/green]")

            if result.isError:
                console.print("[red]Error result:[/red]")

            # Display content blocks
            for content in result.content:
                if isinstance(content, TextContent):
                    console.print(
                        Panel(
                            content.text,
                            title="Result",
                            border_style="green" if not result.isError else "red",
                        )
                    )
                elif isinstance(content, ImageContent):
                    mime_type = getattr(content, "mimeType", "image/png")
                    data_length = len(content.data) if hasattr(content, "data") else 0
                    console.print(
                        Panel(
                            f"📷 Image ({mime_type})\nSize: {data_length:,} bytes (base64 encoded)",
                            title="Result",
                            border_style="green" if not result.isError else "red",
                        )
                    )
                else:
                    # Handle other content types
                    console.print(json.dumps(content, indent=2))

        except Exception as e:
            console.print(f"[red]✗ Tool execution failed: {e}[/red]")

    async def run(self) -> None:
        """Run the interactive testing loop."""
        self.console.header("Interactive MCP Tester")

        # Connect to server
        console.print(f"[cyan]Connecting to {self.server_url}...[/cyan]")
        if not await self.connect():
            return

        console.print("[green]✓ Connected successfully[/green]")
        console.print(f"[dim]Found {len(self.tools)} tools[/dim]\n")

        try:
            while True:
                # Select tool
                tool = await self.select_tool()
                if not tool:
                    break

                # Get arguments
                console.print(f"\n[cyan]Selected: {tool.name}[/cyan]")
                arguments = await self.get_tool_arguments(tool)
                if arguments is None:
                    console.print("[yellow]Skipping tool call[/yellow]")
                    continue

                # Call tool
                await self.call_tool(tool, arguments)

                # Just add a separator and continue to tool selection
                console.print("\n" + "─" * 50)

        finally:
            # Disconnect
            console.print("\n[cyan]Disconnecting...[/cyan]")
            await self.disconnect()

            # Show next steps tutorial
            self.console.section_title("Next Steps")
            self.console.info("🏗️  Ready to test with real agents? Run:")
            self.console.info("    [cyan]hud build[/cyan]")
            self.console.info("")
            self.console.info("This will:")
            self.console.info("  1. Build your environment image")
            self.console.info("  2. Generate a hud.lock.yaml file")
            self.console.info("  3. Prepare it for testing with agents")
            self.console.info("")
            self.console.info("Then you can:")
            self.console.info("  • Test locally: [cyan]hud run <image>[/cyan]")
            self.console.info("  • Push to registry: [cyan]hud push --image <registry/name>[/cyan]")
            self.console.info("  • Use with agents via the lock file")

            console.print("\n[dim]Happy testing! 🎉[/dim]")


async def run_interactive_mode(server_url: str, verbose: bool = False) -> None:
    """Run interactive MCP testing mode.

    Args:
        server_url: URL of the MCP server
        verbose: Enable verbose output
    """
    tester = InteractiveMCPTester(server_url, verbose)
    await tester.run()
