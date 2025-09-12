"""Analyze command implementation for MCP environments."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from hud.clients import MCPClient
from hud.utils.hud_console import HUDConsole

console = Console()
hud_console = HUDConsole()


def parse_docker_command(docker_cmd: list[str]) -> dict:
    """Convert Docker command to MCP config."""
    return {
        "local": {"command": docker_cmd[0], "args": docker_cmd[1:] if len(docker_cmd) > 1 else []}
    }


async def analyze_environment(docker_cmd: list[str], output_format: str, verbose: bool) -> None:
    """Analyze MCP environment and display results."""
    hud_console.header("MCP Environment Analysis", icon="🔍")

    # Convert Docker command to MCP config
    mcp_config = parse_docker_command(docker_cmd)

    # Display command being analyzed
    hud_console.dim_info("Command:", " ".join(docker_cmd))
    hud_console.info("")  # Empty line

    # Create client
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing MCP client...", total=None)

        client = MCPClient(mcp_config=mcp_config, verbose=verbose, auto_trace=False)

        try:
            await client.initialize()
            progress.update(task, description="[green]✓ Client initialized[/green]")

            # Analyze environment
            progress.update(task, description="Analyzing environment...")
            analysis = await client.analyze_environment()
            progress.update(task, description="[green]✓ Analysis complete[/green]")

        except Exception as e:
            progress.update(task, description=f"[red]✗ Failed: {e}[/red]")

            # On Windows, Docker stderr might not propagate properly
            import platform

            if platform.system() == "Windows" and "docker" in docker_cmd[0].lower():
                console.print("\n[yellow]💡 Tip: Docker logs may not show on Windows.[/yellow]")
                console.print(f"[yellow]   Try: hud debug {' '.join(docker_cmd[3:])}[/yellow]")
                console.print("[yellow]   This will show more detailed error information.[/yellow]")
            elif verbose:
                console.print("\n[dim]For more details, try running with 'hud debug'[/dim]")

            return
        finally:
            await client.shutdown()

    # Display results based on format
    if output_format == "json":
        console.print_json(json.dumps(analysis, indent=2))
    elif output_format == "markdown":
        display_markdown(analysis)
    else:  # interactive
        display_interactive(analysis)


def display_interactive(analysis: dict) -> None:
    """Display analysis results in interactive format."""
    # Server metadata
    hud_console.section_title("📊 Environment Overview")
    meta_table = Table(show_header=False, box=None)
    meta_table.add_column("Property", style="bright_black")
    meta_table.add_column("Value")

    # Check if this is a live analysis (has metadata) or metadata-only analysis
    if "metadata" in analysis:
        # Live analysis format
        for server in analysis["metadata"]["servers"]:
            meta_table.add_row("Server", f"[green]{server}[/green]")
        meta_table.add_row(
            "Initialized",
            "[green]✓[/green]" if analysis["metadata"]["initialized"] else "[red]✗[/red]",
        )
    else:
        # Metadata-only format
        if "image" in analysis:
            # Show simple name in table
            image = analysis["image"]
            display_ref = image.split("@")[0] if ":" in image and "@" in image else image
            meta_table.add_row("Image", f"[green]{display_ref}[/green]")

        if "status" in analysis:
            meta_table.add_row("Source", analysis.get("source", analysis["status"]).title())

        if "build_info" in analysis:
            meta_table.add_row("Built", analysis["build_info"].get("generatedAt", "Unknown"))
            meta_table.add_row("HUD Version", analysis["build_info"].get("hudVersion", "Unknown"))

        if "push_info" in analysis:
            meta_table.add_row("Pushed", analysis["push_info"].get("pushedAt", "Unknown"))

        if "init_time" in analysis:
            meta_table.add_row("Init Time", f"{analysis['init_time']} ms")

        if "tool_count" in analysis:
            meta_table.add_row("Tools", str(analysis["tool_count"]))

    console.print(meta_table)

    # Tools
    hud_console.section_title("🔧 Available Tools")
    tools_tree = Tree("[bold bright_white]Tools[/bold bright_white]")

    # Check if we have hub_tools info (live analysis) or not (metadata-only)
    if "hub_tools" in analysis:
        # Live analysis format - separate regular and hub tools
        # Regular tools
        regular_tools = tools_tree.add("[bright_white]Regular Tools[/bright_white]")
        for tool in analysis["tools"]:
            if tool["name"] not in analysis["hub_tools"]:
                tool_node = regular_tools.add(f"[bright_white]{tool['name']}[/bright_white]")
                if tool["description"]:
                    tool_node.add(f"[bright_black]{tool['description']}[/bright_black]")

                # Show input schema if verbose
                if analysis.get("verbose") and tool.get("input_schema"):
                    schema_str = json.dumps(tool["input_schema"], indent=2)
                    syntax = Syntax(schema_str, "json", theme="monokai", line_numbers=False)
                    tool_node.add(syntax)

        # Hub tools
        if analysis["hub_tools"]:
            hub_tools = tools_tree.add("[bright_white]Hub Tools[/bright_white]")
            for hub_name, functions in analysis["hub_tools"].items():
                hub_node = hub_tools.add(f"[rgb(181,137,0)]{hub_name}[/rgb(181,137,0)]")
                for func in functions:
                    hub_node.add(f"[bright_white]{func}[/bright_white]")
    else:
        # Metadata-only format - just list all tools
        for tool in analysis["tools"]:
            tool_node = tools_tree.add(f"[bright_white]{tool['name']}[/bright_white]")
            if tool.get("description"):
                tool_node.add(f"[bright_black]{tool['description']}[/bright_black]")

            # Show input schema if verbose
            if tool.get("inputSchema"):
                schema_str = json.dumps(tool["inputSchema"], indent=2)
                syntax = Syntax(schema_str, "json", theme="monokai", line_numbers=False)
                tool_node.add(syntax)

    console.print(tools_tree)

    # Resources
    if analysis["resources"]:
        hud_console.section_title("📚 Available Resources")
        resources_table = Table()
        resources_table.add_column("URI", style="bright_white")
        resources_table.add_column("Name", style="bright_white")
        resources_table.add_column("Type", style="bright_black")

        for resource in analysis["resources"][:10]:
            resources_table.add_row(
                resource["uri"], resource.get("name", ""), resource.get("mime_type", "")
            )

        console.print(resources_table)

        if len(analysis["resources"]) > 10:
            remaining = len(analysis["resources"]) - 10
            console.print(f"[bright_black]... and {remaining} more resources[/bright_black]")

    # Telemetry (only for live analysis)
    if analysis.get("telemetry"):
        hud_console.section_title("📡 Telemetry Data")
        telemetry_table = Table(show_header=False, box=None)
        telemetry_table.add_column("Key", style="dim")
        telemetry_table.add_column("Value")

        if "live_url" in analysis["telemetry"]:
            telemetry_table.add_row("Live URL", f"[link]{analysis['telemetry']['live_url']}[/link]")
        if "status" in analysis["telemetry"]:
            telemetry_table.add_row("Status", f"[green]{analysis['telemetry']['status']}[/green]")
        if "services" in analysis["telemetry"]:
            services = analysis["telemetry"]["services"]
            running = sum(1 for s in services.values() if s == "running")
            telemetry_table.add_row("Services", f"{running}/{len(services)} running")

        console.print(telemetry_table)

    # Environment variables (for metadata-only analysis)
    if analysis.get("env_vars"):
        hud_console.section_title("🔑 Environment Variables")
        env_table = Table(show_header=False, box=None)
        env_table.add_column("Type", style="dim")
        env_table.add_column("Variables")

        if analysis["env_vars"].get("required"):
            env_table.add_row("Required", ", ".join(analysis["env_vars"]["required"]))
        if analysis["env_vars"].get("optional"):
            env_table.add_row("Optional", ", ".join(analysis["env_vars"]["optional"]))

        console.print(env_table)


def display_markdown(analysis: dict) -> None:
    """Display analysis results in markdown format."""
    md = []
    md.append("# MCP Environment Analysis\n")

    # Metadata
    md.append("## Environment Overview")

    # Check if this is live analysis or metadata-only
    if "metadata" in analysis:
        md.append(f"- **Servers**: {', '.join(analysis['metadata']['servers'])}")
        md.append(f"- **Initialized**: {'✓' if analysis['metadata']['initialized'] else '✗'}")
    else:
        # Metadata-only format
        if "image" in analysis:
            md.append(f"- **Image**: {analysis['image']}")
        if "source" in analysis:
            md.append(f"- **Source**: {analysis['source']}")
        if "build_info" in analysis:
            md.append(f"- **Built**: {analysis['build_info'].get('generatedAt', 'Unknown')}")
        if "tool_count" in analysis:
            md.append(f"- **Tools**: {analysis['tool_count']}")

    md.append("")

    # Tools
    md.append("## Available Tools\n")

    # Check if we have hub_tools info (live analysis) or not (metadata-only)
    if "hub_tools" in analysis:
        # Regular tools
        md.append("### Regular Tools")
        for tool in analysis["tools"]:
            if tool["name"] not in analysis["hub_tools"]:
                md.extend([f"- **{tool['name']}**: {tool.get('description', 'No description')}"])
        md.append("")

        # Hub tools
        if analysis["hub_tools"]:
            md.append("### Hub Tools")
            for hub_name, functions in analysis["hub_tools"].items():
                md.extend([f"- **{hub_name}**"])
                for func in functions:
                    md.extend([f"  - {func}"])
            md.append("")
    else:
        # Metadata-only format - just list all tools
        for tool in analysis["tools"]:
            md.extend([f"- **{tool['name']}**: {tool.get('description', 'No description')}"])
        md.append("")

    # Resources
    if analysis["resources"]:
        md.append("## Available Resources\n")
        md.append("| URI | Name | Type |")
        md.append("|-----|------|------|")
        for resource in analysis["resources"]:
            uri = resource["uri"]
            name = resource.get("name", "")
            mime_type = resource.get("mime_type", "")
            md.extend([f"| {uri} | {name} | {mime_type} |"])
        md.append("")

    # Telemetry (only for live analysis)
    if analysis.get("telemetry"):
        md.append("## Telemetry")
        if "live_url" in analysis["telemetry"]:
            md.extend([f"- **Live URL**: {analysis['telemetry']['live_url']}"])
        if "status" in analysis["telemetry"]:
            md.extend([f"- **Status**: {analysis['telemetry']['status']}"])
        if "services" in analysis["telemetry"]:
            md.extend([f"- **Services**: {analysis['telemetry']['services']}"])
        md.append("")

    # Environment variables (for metadata-only analysis)
    if analysis.get("env_vars"):
        md.append("## Environment Variables")
        if analysis["env_vars"].get("required"):
            md.extend([f"- **Required**: {', '.join(analysis['env_vars']['required'])}"])
        if analysis["env_vars"].get("optional"):
            md.extend([f"- **Optional**: {', '.join(analysis['env_vars']['optional'])}"])
        md.append("")

    console.print("\n".join(md))


async def analyze_environment_from_config(
    config_path: Path, output_format: str, verbose: bool
) -> None:
    """Analyze MCP environment from a JSON config file."""
    hud_console.header("MCP Environment Analysis", icon="🔍")

    # Load config from file
    try:
        with open(config_path) as f:  # noqa: ASYNC230
            mcp_config = json.load(f)
        console.print(f"[dim]Config: {config_path}[/dim]\n")
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        return

    await _analyze_with_config(mcp_config, output_format, verbose)


async def analyze_environment_from_mcp_config(
    mcp_config: dict[str, Any], output_format: str, verbose: bool
) -> None:
    """Analyze MCP environment from MCP config dict."""
    hud_console.header("MCP Environment Analysis", icon="🔍")
    await _analyze_with_config(mcp_config, output_format, verbose)


async def _analyze_with_config(
    mcp_config: dict[str, Any], output_format: str, verbose: bool
) -> None:
    """Internal helper to analyze with MCP config."""
    # Create client
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing MCP client...", total=None)

        client = MCPClient(mcp_config=mcp_config, verbose=verbose)

        try:
            await client.initialize()
            progress.update(task, description="[green]✓ Client initialized[/green]")

            # Analyze environment
            progress.update(task, description="Analyzing environment...")
            analysis = await client.analyze_environment()
            progress.update(task, description="[green]✓ Analysis complete[/green]")

        except Exception as e:
            progress.update(task, description=f"[red]✗ Failed: {e}[/red]")
            return
        finally:
            await client.shutdown()

    # Display results based on format
    if output_format == "json":
        console.print_json(json.dumps(analysis, indent=2))
    elif output_format == "markdown":
        display_markdown(analysis)
    else:  # interactive
        display_interactive(analysis)
