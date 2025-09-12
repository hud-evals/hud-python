"""Initialize RL configuration from environment analysis."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
import yaml

from hud.clients import MCPClient
from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def init_command_wrapper(directory: str, output: Path | None, force: bool, build: bool) -> None:
    """Wrapper to handle interactive prompts before entering async context."""
    hud_console.header("RL Config Generator", icon="🔧")

    # Determine if this is a directory or Docker image
    path = Path(directory)
    is_directory = path.exists() and path.is_dir()

    if is_directory:
        # Working with a directory - check for lock file
        lock_path = path / "hud.lock.yaml"

        if not lock_path.exists():
            if build:
                # Auto-build was requested
                hud_console.info("Building environment...")
                from hud.cli.build import build_command

                build_command(str(directory), None, False, False, {})
                # After build, lock file should exist
            else:
                # Try to get image from pyproject.toml or auto-generate
                from hud.cli.utils.environment import get_image_name, image_exists

                image, source = get_image_name(directory)

                if not (source == "cache" and image_exists(image)):
                    hud_console.warning(f"No hud.lock.yaml found in {directory}")
                    # Need to handle interactive prompt here, before async
                    action = hud_console.select(
                        "No lock file found. Would you like to:",
                        ["Build the environment", "Use Docker image directly", "Cancel"],
                    )

                    if action == "Build the environment":
                        hud_console.info("Building environment...")
                        from hud.cli.build import build_command

                        build_command(str(directory), None, False, False, {})
                        # After build, lock file should exist
                    elif action == "Use Docker image directly":
                        # Prompt for image name
                        image = typer.prompt("Enter Docker image name")
                        directory = image  # Override to use as Docker image
                        is_directory = False  # Treat as image, not directory
                    else:
                        raise typer.Exit(1)

    # Now run the async command with resolved parameters
    asyncio.run(init_command(directory, output, force, False))


async def init_command(directory: str, output: Path | None, force: bool, build: bool) -> None:
    """Generate hud-vf-gym config from environment."""
    # Determine if this is a directory or Docker image
    path = Path(directory)
    is_directory = path.exists() and path.is_dir()

    if is_directory:
        # Working with a directory - look for lock file
        lock_path = path / "hud.lock.yaml"

        if lock_path.exists():
            hud_console.info(f"Found lock file: {lock_path}")
            lock_data = read_lock_file_path(lock_path)

            if not lock_data:
                hud_console.error("Failed to read lock file")
                raise typer.Exit(1)

            # Get image and tools from lock file
            image = lock_data.get("image", "")
            tools = lock_data.get("tools", [])

            if not image:
                hud_console.error("No image found in lock file")
                hud_console.hint("Run 'hud build' to create a proper lock file")
                raise typer.Exit(1)

            if not tools:
                hud_console.error("No tools found in lock file")
                hud_console.hint("Lock file may be outdated. Run 'hud build' to regenerate")
                raise typer.Exit(1)

            # Use lock file data to generate config
            await generate_from_lock(image, tools, output, force)

        else:
            # No lock file - try to use cached image
            # Build should have been handled in the wrapper
            from hud.cli.utils.environment import get_image_name, image_exists

            image, source = get_image_name(directory)

            if source == "cache" and image_exists(image):
                # Found cached image in pyproject.toml
                hud_console.info(f"Using cached image: {image}")
                await analyze_and_generate(image, output, force)
            else:
                # This should have been handled in the wrapper
                hud_console.error("No valid image or lock file found")
                raise typer.Exit(1)

    else:
        # Working with a Docker image directly
        image = directory
        await analyze_and_generate(image, output, force)


def read_lock_file_path(lock_path: Path) -> dict[str, Any]:
    """Read lock file from specific path."""
    try:
        with open(lock_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        hud_console.error(f"Failed to read lock file: {e}")
        return {}


async def generate_from_lock(
    image: str, tools: list[dict], output: Path | None, force: bool
) -> None:
    """Generate config from lock file data."""
    # Determine output path
    if output is None:
        # Default to configs/{image_name}.yaml
        image_name = image.split("/")[-1].split(":")[0]
        if "/" in image_name:
            image_name = image_name.split("/")[-1]
        output = Path("configs") / f"{image_name}.yaml"

    # Check if file exists
    if output.exists() and not force:
        hud_console.error(f"Config file already exists: {output}")
        hud_console.info("Use --force to overwrite")
        raise typer.Exit(1)

    # Create output directory if needed
    output.parent.mkdir(parents=True, exist_ok=True)

    # Convert lock file tool format to full tool format
    # Lock file may have full or simplified format
    full_tools = []
    for tool in tools:
        full_tool = {
            "name": tool["name"],
            "description": tool.get("description", ""),
        }
        # Check if lock file has inputSchema (newer format)
        if "inputSchema" in tool:
            full_tool["inputSchema"] = tool["inputSchema"]
        else:
            # Old lock file format without schema
            full_tool["inputSchema"] = {"type": "object", "properties": {}, "required": []}
        full_tools.append(full_tool)

    # Generate config
    config = await generate_config(image, full_tools)

    # Write to file
    with open(output, "w") as f:  # noqa: ASYNC230
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    hud_console.success(f"Generated config: {output}")

    # Show summary
    hud_console.section_title("📋 Generated Configuration")
    hud_console.info("Source: hud.lock.yaml")
    hud_console.info(f"Image: {image}")
    hud_console.info(f"System prompt: {len(config['system_prompt'])} characters")
    hud_console.info(f"Action mappings: {len(config['action_mappings'])} tools")
    hud_console.info("")
    hud_console.info("Next steps:")
    hud_console.command_example("hud hf tasks.json --name my-tasks", "Create dataset")
    hud_console.command_example(f"hud rl --config {output}", "Start training")


async def analyze_and_generate(image: str, output: Path | None, force: bool) -> None:
    """Analyze Docker image and generate config."""
    # Determine output path
    if output is None:
        # Default to configs/{image_name}.yaml
        image_name = image.split("/")[-1].split(":")[0]
        output = Path("configs") / f"{image_name}.yaml"

    # Check if file exists
    if output.exists() and not force:
        hud_console.error(f"Config file already exists: {output}")
        hud_console.info("Use --force to overwrite")
        raise typer.Exit(1)

    # Create output directory if needed
    output.parent.mkdir(parents=True, exist_ok=True)

    hud_console.info(f"Analyzing environment: {image}")

    # Analyze the environment
    try:
        # Create MCP config for Docker
        mcp_config = {"local": {"command": "docker", "args": ["run", "--rm", "-i", image]}}

        # Initialize client and analyze
        client = MCPClient(mcp_config=mcp_config, auto_trace=False)
        await client.initialize()

        try:
            analysis = await client.analyze_environment()
            tools = analysis.get("tools", [])

            # Generate config
            config = await generate_config(image, tools)

            # Write to file
            with open(output, "w") as f:  # noqa: ASYNC230
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            hud_console.success(f"Generated config: {output}")

            # Show summary
            hud_console.section_title("📋 Generated Configuration")
            hud_console.info(f"System prompt: {len(config['system_prompt'])} characters")
            hud_console.info(f"Action mappings: {len(config['action_mappings'])} tools")
            hud_console.info("")
            hud_console.info("Next steps:")
            hud_console.command_example("hud hf tasks.json --name my-tasks", "Create dataset")
            hud_console.command_example(f"hud rl --config {output}", "Start training")

        finally:
            await client.shutdown()

    except Exception as e:
        hud_console.error(f"Failed to analyze environment: {e}")
        hud_console.hint("Make sure the Docker image exists and contains a valid MCP server")
        raise typer.Exit(1) from e


async def generate_config(image: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate hud-vf-gym configuration from tool analysis."""
    # Clean up image name for display
    display_name = image.split("@")[0] if "@" in image else image  # Remove SHA hash
    env_name = display_name.split("/")[-1].split(":")[0]  # Extract just the env name

    # Filter out setup/evaluate tools
    interaction_tools = [t for t in tools if t["name"] not in ["setup", "evaluate"]]

    # Generate system prompt
    tool_descriptions = []
    for tool in interaction_tools:
        # Check if we have schema (from direct analysis) or just name/description (from lock file)
        has_schema = "inputSchema" in tool and tool["inputSchema"].get("properties")

        if has_schema:
            params = tool.get("inputSchema", {}).get("properties", {})
            required = tool.get("inputSchema", {}).get("required", [])

            # Build parameter string
            param_parts = []
            for name, schema in params.items():
                param_type = schema.get("type", "any")
                if name in required:
                    param_parts.append(f"{name}: {param_type}")
                else:
                    param_parts.append(f"{name}?: {param_type}")

            param_str = ", ".join(param_parts) if param_parts else ""
        else:
            # No schema information
            param_str = "..."

        desc = tool.get("description", "No description")

        tool_descriptions.append(
            f"- {tool['name']}({param_str}): {desc}\n  Usage: <tool>{tool['name']}(...)</tool>"
        )

    # Add note if any tools are missing schema info
    if interaction_tools and any("inputSchema" not in t for t in interaction_tools):
        tool_descriptions.append(
            "\nNote: Some tools are missing parameter information. Update manually if needed."
        )

    system_prompt = f"""You are an AI agent in a HUD environment.

You have access to the following tools:

{chr(10).join(tool_descriptions)}

Always use the exact XML format shown above for tool calls.
Think step by step about what you need to do."""

    # Generate action mappings
    action_mappings = {}

    for tool in interaction_tools:
        # Check if we have inputSchema information
        has_input_schema = "inputSchema" in tool

        if has_input_schema:
            # We have schema info (even if no parameters)
            params = tool.get("inputSchema", {}).get("properties", {})
            required = tool.get("inputSchema", {}).get("required", [])

            # Simple 1:1 mapping by default
            mapping = {
                "_tool": tool["name"],
                "_parser": {
                    "positional": list(required)  # Use required params as positional
                },
            }

            # Add parameter mappings (only if there are params)
            for param_name in params:
                mapping[param_name] = {"from_arg": param_name}
        else:
            # No schema information at all
            mapping = {
                "_tool": tool["name"],
                "_parser": {
                    "positional": []  # No positional args without schema
                },
                "# TODO": "Update with actual parameters",
            }

        action_mappings[tool["name"]] = mapping

    # Add special "done" action if not present
    if "done" not in action_mappings:
        action_mappings["done"] = {
            "_tool": None,  # Special marker for task completion
            "_parser": {"positional": []},
        }

    # Build full config
    config = {
        "# Generated by hud rl init": f"for {env_name}",
        "job": {
            "name": f"RL Training - {env_name}",
            "metadata": {
                "environment": display_name,
                "full_image": image,
                "generated_by": "hud rl init",
            },
        },
        "system_prompt": system_prompt,
        "parser": {"use_thinking": True, "xml_weight": 0.6, "action_weight": 0.4},
        "action_mappings": action_mappings,
        "rubric": {
            "weights": {"task_completion": 0.8, "tool_execution": 0.1, "format_compliance": 0.1}
        },
        "defaults": {"max_turns": 100},
    }

    return config
