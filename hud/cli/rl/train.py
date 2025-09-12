"""Main RL training command implementation."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import typer

from hud.settings import settings
from hud.utils.hud_console import HUDConsole

from .pod import run_prime_training
from .utils import (
    detect_image_name,
    get_primary_dataset,
    validate_dataset_name,
)

hud_console = HUDConsole()


def find_task_json_files() -> list[Path]:
    """Find JSON files containing tasks in the current directory."""
    json_files = []
    patterns = [
        "*task*.json",
        "*eval*.json",
        "*Task*.json",
        "*Eval*.json",
        "*TASK*.json",
        "*EVAL*.json",
        "tasks.json",  # Most common name
    ]

    # First check current directory
    for pattern in patterns:
        json_files.extend(Path(".").glob(pattern))

    # If no files found, search one level deep
    if not json_files:
        for pattern in patterns:
            json_files.extend(Path(".").glob(f"*/{pattern}"))

    # Remove duplicates and sort, prioritizing "tasks.json"
    json_files = sorted(set(json_files))

    # Put tasks.json first if it exists
    tasks_json = Path("tasks.json")
    if tasks_json in json_files:
        json_files.remove(tasks_json)
        json_files.insert(0, tasks_json)

    return json_files


def train_command_wrapper(
    model: str,
    dataset: str | None,
    config: Path | None,
    gpus: str,
    provider: str,
    output_dir: Path,
) -> None:
    """Wrapper to handle interactive prompts before entering async context."""
    # Pre-flight checks for required environment variables
    hud_console.section_title("🔍 Pre-flight Checks")

    missing_vars = []

    # Check HUD API key
    if not settings.api_key:
        missing_vars.append("HUD_API_KEY")
    else:
        hud_console.success("✓ HUD_API_KEY configured")

    # Check WANDB API key (optional but recommended)
    if not getattr(settings, "wandb_api_key", None):
        hud_console.warning(
            "⚠ WANDB_API_KEY not set (optional but recommended for training metrics)"
        )
    else:
        hud_console.success("✓ WANDB_API_KEY configured")

    # Check PRIME API key (required for remote training)
    if provider == "prime" and not getattr(settings, "prime_api_key", None):
        missing_vars.append("PRIME_API_KEY")
    elif provider == "prime":
        hud_console.success("✓ PRIME_API_KEY configured")

    if missing_vars:
        hud_console.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        hud_console.info("")
        hud_console.info("Set them using one of these methods:")
        hud_console.info("1. Environment variables:")
        for var in missing_vars:
            hud_console.command_example(f"export {var}=your-{var.lower().replace('_', '-')}")
        hud_console.info("")
        hud_console.info("2. Create a .env file in your project root:")
        hud_console.command_example(
            "\n".join([f"{var}=your-{var.lower().replace('_', '-')}" for var in missing_vars]),
            "env",
        )
        raise typer.Exit(1)

    # Check for required components
    missing = check_requirements(config, dataset)

    # Auto-detect config if not specified and exactly one exists
    if not config and "config" not in missing:
        config_dir = Path("configs")
        if config_dir.exists():
            yaml_files = list(config_dir.glob("*.yaml"))
            if len(yaml_files) == 1:
                config = yaml_files[0]
                hud_console.info(f"Using config: {config}")

    # Store user choice for pod creation
    auto_create_pod = None
    team_id = None

    if missing:
        # Handle interactive prompts here
        if "config" in missing:
            if missing["config"] == "multiple":
                # Select from multiple configs
                config_dir = Path("configs")
                yaml_files = list(config_dir.glob("*.yaml"))
                config_names = [f.name for f in yaml_files]
                selected_config = hud_console.select(
                    "Multiple config files found. Select one:", config_names
                )
                config = config_dir / selected_config
            else:
                # No config found, offer to generate
                generate_config = hud_console.select(
                    "No config file found. Would you like to generate one?",
                    ["Yes, generate config", "No, I'll create it manually"],
                )

                if generate_config == "Yes, generate config":
                    hud_console.info("Running 'hud rl init' to generate config...")
                    hud_console.info("")
                    # Import here to avoid circular imports
                    from .init import init_command_wrapper

                    init_command_wrapper(".", None, False, False)

                    # Look for generated config
                    config_dir = Path("configs")
                    if config_dir.exists():
                        yaml_files = list(config_dir.glob("*.yaml"))
                        if yaml_files:
                            config = yaml_files[0]
                            hud_console.success(f"Using generated config: {config}")
                        else:
                            hud_console.error("Config generation failed")
                            raise typer.Exit(1)
                else:
                    hud_console.info("Please create a config file and try again")
                    raise typer.Exit(1)

        if "dataset" in missing:
            if missing["dataset"] == "multiple_json":
                # Multiple JSON files found, let user choose
                json_files = find_task_json_files()
                hud_console.info("Multiple task files found:")
                file_choice = hud_console.select(
                    "Select a task file to use:",
                    choices=[str(f) for f in json_files],
                )
                dataset = file_choice
                hud_console.success(f"Selected: {dataset}")
            elif missing["dataset"] == "none":
                hud_console.error("No dataset specified and no task JSON files found")
                hud_console.info("Please use --dataset or create a tasks.json file")
                hud_console.hint(
                    "Example: hud hf --name my-org/my-tasks  # Generate tasks from HUD evaluation"
                )
                raise typer.Exit(1)

    # Ask about pod creation for Prime training
    if provider == "prime":
        # Check if team ID is globally configured
        team_check = subprocess.run(
            ["prime", "config", "view"],  # noqa: S607
            capture_output=True,
            text=True,
        )

        has_global_team = False
        if team_check.returncode == 0:
            # Parse the table output - look for Team ID row
            for line in team_check.stdout.split("\n"):
                if "team id" in line.lower():
                    # Check if there's a value after the | separator
                    parts = line.split("|")
                    if len(parts) >= 2:
                        # Get the value part and check if it's not empty
                        value = parts[1].strip()
                        if value and value != "None":
                            has_global_team = True
                            hud_console.info("Using globally configured team ID")
                            break

        if not has_global_team:
            # Only ask if no global team is configured
            auto_create_pod = hud_console.select(
                "How would you like to create the Prime Intellect pod?",
                ["Personal account (automated)", "Team account (enter team ID)"],
            )

            # If team account selected, get the team ID
            if auto_create_pod == "Team account (enter team ID)":
                team_id = typer.prompt("Enter your team ID (e.g., team_abc123def456)")

                # Save it globally automatically
                subprocess.run(["prime", "config", "set-team-id", team_id])  # noqa: S603, S607
                hud_console.success("Team ID saved globally")

                auto_create_pod = (
                    "Personal account (automated)"  # Treat as automated after getting team ID
                )

    # Now run the async command
    asyncio.run(
        train_command(
            model=model,
            dataset=dataset,
            config=config,
            gpus=gpus,
            provider=provider,
            output_dir=output_dir,
            auto_create_pod=auto_create_pod,
            team_id=team_id,
        )
    )


async def train_command(
    model: str,
    dataset: str | None,
    config: Path | None,
    gpus: str,
    provider: str,
    output_dir: Path,
    auto_create_pod: str | None = None,
    team_id: str | None = None,
) -> None:
    """Run RL training on HUD environments."""
    hud_console.header("🤖 HUD RL Training")

    # Get environment image
    image = detect_image_name()
    if not image:
        hud_console.error("No environment image found")
        hud_console.hint("Run 'hud build' first or specify with 'hud rl init <image>'")
        raise typer.Exit(1)

    # Handle dataset (JSON file or HuggingFace dataset)
    dataset_size = None
    is_json_file = False

    # Use dataset from command or look for JSON files
    if not dataset:
        # Check for JSON files if no dataset specified
        json_files = find_task_json_files()
        if json_files:
            if len(json_files) == 1:
                dataset = str(json_files[0])
                hud_console.info(f"Found task file: {dataset}")
                is_json_file = True
            else:
                # This case should have been handled in train_command_wrapper
                hud_console.error("Multiple task files found but none selected")
                raise typer.Exit(1)
        else:
            # Use dataset from lock file
            dataset = get_primary_dataset()
            if dataset:
                hud_console.info(f"Using dataset from lock file: {dataset}")

    # Check if dataset is a file path
    if dataset and Path(dataset).exists() and dataset.endswith(".json"):
        is_json_file = True

    # Validate dataset
    if dataset and is_json_file:
        # Load and validate JSON file
        hud_console.info(f"Validating task file: {dataset}")
        try:
            with open(dataset) as f:  # noqa: ASYNC230
                tasks_data = json.load(f)

            # Handle both single task and array of tasks
            if isinstance(tasks_data, dict):
                tasks = [tasks_data]
            elif isinstance(tasks_data, list):
                tasks = tasks_data
            else:
                hud_console.error("Invalid tasks file format")
                raise typer.Exit(1)

            dataset_size = len(tasks)
            if dataset_size < 4:
                hud_console.error(f"Task file has only {dataset_size} tasks")
                hud_console.info("RL training requires at least 4 tasks for proper batching")
                hud_console.hint("Consider adding more tasks to your JSON file")
                raise typer.Exit(1)

            hud_console.success(f"✓ Task file has {dataset_size} tasks")

            # Check and convert MCP configs to remote if needed
            if tasks:
                sample_task = tasks[0]
                sample_mcp_config = sample_task.get("mcp_config", {})

                # Check if using local MCP configs
                config_type = "unknown"
                for server_config in sample_mcp_config.values():
                    if isinstance(server_config, dict) and "url" in server_config:
                        url = server_config.get("url", "")
                        if "mcp.hud.so" in url:
                            config_type = "remote"
                            break
                        else:
                            config_type = "local"

                if config_type == "local":
                    hud_console.info("Converting local MCP configs to remote for training...")

                    # Get the image name from lock file or environment
                    from .utils import get_image_from_lock

                    env_image = image or get_image_from_lock()

                    if not env_image:
                        hud_console.error("No image found for remote MCP conversion")
                        hud_console.hint("Run 'hud build' first")
                        raise typer.Exit(1)

                    # Check if image needs to be pushed
                    if "/" not in env_image or env_image.startswith("local/"):
                        hud_console.warning(f"Image '{env_image}' appears to be local only")
                        hud_console.info("Running 'hud push' to make it publicly available...")
                        from hud.cli.push import push_command

                        push_command(directory=".", yes=True)
                        hud_console.success("Image pushed successfully")
                        # Re-read image name after push
                        env_image = get_image_from_lock()

                    # Convert all tasks to use remote MCP
                    for task in tasks:
                        remote_config = {
                            "hud": {
                                "url": "https://mcp.hud.so/v3/mcp",
                                "headers": {
                                    "Authorization": "Bearer $HUD_API_KEY",
                                    "Mcp-Image": env_image,
                                },
                            }
                        }
                        task["mcp_config"] = remote_config

                    hud_console.success("✓ Converted all tasks to use remote MCP configs")

                    # Save the modified tasks back to the file
                    with open(dataset, "w") as f:  # noqa: ASYNC230
                        json.dump(tasks, f, indent=2)
                    hud_console.info("Updated task file with remote configs")
        except json.JSONDecodeError as e:
            hud_console.error(f"Invalid JSON in task file: {e}")
            raise typer.Exit(1) from e
    elif dataset:
        # Validate HuggingFace dataset
        hud_console.info(f"Validating dataset: {dataset}")
        try:
            # Try to load dataset info from HuggingFace
            from datasets import load_dataset_builder

            ds_builder = load_dataset_builder(dataset)
            ds_info = ds_builder.info

            # Check split sizes
            train_size = ds_info.splits.get("train", None) if ds_info.splits else None
            if train_size and train_size.num_examples < 4:
                hud_console.error(f"Dataset '{dataset}' has only {train_size.num_examples} tasks")
                hud_console.info("RL training requires at least 4 tasks for proper batching")
                hud_console.hint("Consider adding more tasks or duplicating existing ones")
                raise typer.Exit(1)
            elif train_size:
                dataset_size = train_size.num_examples
                hud_console.success(f"✓ Dataset has {dataset_size} tasks")
        except Exception as e:
            # If we can't validate, warn but continue
            hud_console.warning(f"Could not validate dataset size: {e}")
            hud_console.info("Proceeding with training - ensure dataset has at least 4 tasks")

    # Display configuration
    hud_console.section_title("📋 Training Configuration")
    hud_console.json_config(
        json.dumps(
            {
                "Model": model,
                "Dataset": dataset,
                "Config": str(config) if config else None,
                "Environment": image,
                "GPUs": gpus,
                "Provider": provider,
                "Output": str(output_dir),
            },
            indent=2,
        )
    )

    if not config:
        hud_console.error("No config file found")
        hud_console.hint("Run 'hud rl init' to generate a config file")
        raise typer.Exit(1)

    if not dataset:
        hud_console.error("No dataset found")
        hud_console.hint("Run 'hud hf tasks.json' to create a dataset")
        raise typer.Exit(1)

    # Always run remote training
    await run_remote_training(
        model=model,
        dataset=dataset,
        config=config,
        gpus=gpus,
        provider=provider,
        output_dir=output_dir,
        image=image,
        auto_create_pod=auto_create_pod,
        team_id=team_id,
        dataset_size=dataset_size,
        is_json_file=is_json_file,
    )


def check_requirements(config: Path | None, dataset: str | None) -> dict[str, Any]:
    """Check if required components are present."""
    missing = {}

    # Check config
    if not config:
        config_dir = Path("configs")
        if config_dir.exists():
            yaml_files = list(config_dir.glob("*.yaml"))
            if not yaml_files:
                missing["config"] = "none"
            elif len(yaml_files) > 1:
                missing["config"] = "multiple"
            # If exactly one config, we'll use it
        else:
            missing["config"] = "none"

    # Check dataset
    if not dataset:
        # First check for JSON files (preferred method)
        json_files = find_task_json_files()
        if json_files:
            if len(json_files) == 1:
                # Will be auto-selected
                pass
            else:
                missing["dataset"] = "multiple_json"
        else:
            # Check lock file for HuggingFace dataset
            primary_dataset = get_primary_dataset()
            if not primary_dataset:
                missing["dataset"] = "none"

    return missing


def generate_config_interactive() -> Path | None:
    """Generate config interactively and return the path."""
    from .init import init_command

    # Run init command
    asyncio.run(init_command(".", None, False, False))

    # Look for generated config
    config_dir = Path("configs")
    if config_dir.exists():
        yaml_files = list(config_dir.glob("*.yaml"))
        if yaml_files:
            return yaml_files[0]

    return None


def create_dataset_interactive() -> str | None:
    """Create dataset interactively and return the name."""
    # Check if tasks.json exists
    tasks_file = Path("tasks.json")
    if not tasks_file.exists():
        hud_console.error("No tasks.json file found")
        return None

    # Prompt for dataset name
    dataset_name = typer.prompt("Enter HuggingFace dataset name (e.g., username/dataset-name)")

    if not validate_dataset_name(dataset_name):
        hud_console.error("Invalid dataset name format")
        return None

    # Run hf command
    result = subprocess.run(  # noqa: S603
        ["hud", "hf", "tasks.json", "--name", dataset_name],  # noqa: S607
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        return dataset_name
    else:
        hud_console.error("Failed to create dataset")
        if result.stderr:
            hud_console.error(result.stderr)
        return None


async def run_remote_training(
    model: str,
    dataset: str,
    config: Path,
    gpus: str,
    provider: str,
    output_dir: Path,
    image: str,
    auto_create_pod: str | None = None,
    team_id: str | None = None,
    dataset_size: int | None = None,
    is_json_file: bool = False,
) -> None:
    """Run training on remote infrastructure."""
    hud_console.section_title("🚀 Remote Training")

    if provider == "prime":
        await run_prime_training(
            model,
            dataset,
            config,
            gpus,
            output_dir,
            image,
            auto_create_pod,
            team_id,
            dataset_size,
            is_json_file,
        )
    else:
        hud_console.error(f"Provider '{provider}' not yet supported")
        hud_console.info("Currently supported: prime")
        raise typer.Exit(1)
