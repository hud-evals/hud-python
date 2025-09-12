"""Pod creation and management utilities for Prime Intellect."""

from __future__ import annotations

import random
import re
import string
import subprocess
import time
from pathlib import Path  # noqa: TC003

import typer
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from hud.settings import settings
from hud.utils.hud_console import HUDConsole

from .ssh import check_and_configure_ssh_key, connect_and_train

hud_console = HUDConsole()


def parse_gpu_config(gpus: str) -> tuple[int, str]:
    """Parse GPU configuration string like '2xA100' into count and type."""
    if "x" in gpus:
        count_str, gpu_type = gpus.split("x", 1)
        try:
            count = int(count_str)
        except ValueError as e:
            hud_console.error(f"Invalid GPU count: {count_str}")
            raise typer.Exit(1) from e
    else:
        # Default to 1 GPU if no count specified
        count = 1
        gpu_type = gpus

    # Map common GPU names to Prime's expected format
    gpu_type_map = {
        "A100": "A100_80GB",
        "A10": "A10_24GB",
        "H100": "H100_80GB",
        "V100": "V100_32GB",
        "RTX3090": "RTX_3090",
        "RTX4090": "RTX_4090",
    }

    gpu_type = gpu_type_map.get(gpu_type, gpu_type)

    return count, gpu_type


async def create_and_connect_prime_pod(
    pod_name: str,
    gpu_type: str,
    gpu_count: int,
    model: str,
    dataset: str,
    config: Path,
    output_dir: Path,
    image: str,
    team_id: str | None = None,
    dataset_size: int | None = None,
    is_json_file: bool = False,
) -> None:
    """Create a Prime Intellect pod and connect to it for training."""
    hud_console.section_title("🌐 Creating Prime Intellect Pod")

    create_cmd = [
        "prime",
        "pods",
        "create",
        "--gpu-type",
        gpu_type,
        "--gpu-count",
        str(gpu_count),
        "--name",
        pod_name,
    ]

    hud_console.info(f"Creating pod: {pod_name}")
    hud_console.info(f"GPU configuration: {gpu_count}x {gpu_type}")

    # Check for global team config first
    has_global_team = False
    if not team_id:  # Only check if not explicitly provided
        team_check = subprocess.run(  # noqa: ASYNC221
            ["prime", "config", "view"],  # noqa: S607
            capture_output=True,
            text=True,
        )
        if team_check.returncode == 0:
            # Parse the table output more carefully
            for line in team_check.stdout.split("\n"):
                # Look for "Team ID" in the table (case insensitive)
                if "team id" in line.lower():
                    # Check if there's a value after the | separator
                    parts = line.split("|")
                    if len(parts) >= 2:
                        # Get the value part and check if it's not empty
                        value = parts[1].strip()
                        if value and value != "None":
                            has_global_team = True
                            # Don't overwrite team_id parameter - that's for explicit user input
                            break

    # Display automated selections
    hud_console.info("")
    hud_console.info("Automated selections:")
    hud_console.info("  Provider: Will select from supported providers")
    hud_console.info("  Disk: Default size")
    hud_console.info("  Image: cuda_12_4_pytorch_2_5")
    if team_id:
        hud_console.info(f"  Team: {team_id}")
    elif has_global_team:
        hud_console.info("  Team: Using pre-configured team")
    else:
        hud_console.info("  Team: Personal Account")
    hud_console.info("")

    # First, get the provider list by running the command with minimal input
    hud_console.info("Checking available providers...")

    # Run command with just a newline to see provider list
    provider_check = subprocess.run(  # noqa: S603, ASYNC221
        create_cmd,
        input="\n",  # Just send newline to see providers
        text=True,
        capture_output=True,
    )

    # Parse provider list
    provider_lines = []
    provider_map = {}  # Maps provider name to number

    if provider_check.stdout:
        lines = provider_check.stdout.strip().split("\n")
        for line in lines:
            # Look for lines like "1. datacrunch (spot) ($0.65/hr)"
            if ". " in line and ("$" in line or "/hr" in line):
                # Extract provider number and name
                parts = line.strip().split(". ", 1)
                if len(parts) == 2:
                    num = parts[0].strip()
                    # Extract provider name (before parentheses or dollar sign)
                    provider_info = parts[1]
                    provider_name = provider_info.split("(")[0].split("$")[0].strip().lower()
                    provider_map[provider_name] = num
                    provider_lines.append(line.strip())

    # Select provider based on our supported list
    supported_providers = ["datacrunch", "hyperstack"]
    provider_choice = "1"  # Default fallback

    for provider in supported_providers:
        if provider in provider_map:
            provider_choice = provider_map[provider]
            hud_console.info(f"Selected provider: {provider} (option {provider_choice})")
            break

    # Build inputs step by step for clarity
    disk_size = ""  # Just press enter for default
    image_choice = "7"  # cuda_12_4_pytorch_2_5

    # Log what we're doing
    hud_console.debug("Pod creation configuration:")
    hud_console.debug(f"  Team ID provided: {team_id}")
    hud_console.debug(f"  Global team detected: {has_global_team}")

    if team_id:
        # Explicit team ID provided, select Custom Team ID (option 3)
        team_choice = "3"
        # Fixed: confirmation should be lowercase 'y'
        inputs = f"{provider_choice}\n{disk_size}\n{image_choice}\n{team_choice}\n{team_id}\ny\n"
        hud_console.debug(f"  Using explicit team ID: option {team_choice} with ID {team_id}")
    elif has_global_team:
        # When team is pre-configured, it shows as option 2 - select it
        team_choice = "2"
        # Fixed: confirmation should be lowercase 'y' and come after team selection
        inputs = f"{provider_choice}\n{disk_size}\n{image_choice}\n{team_choice}\ny\n"
        hud_console.debug(f"  Using pre-configured team: option {team_choice}")
    else:
        # Personal account (option 1) - just press enter to accept default [1]
        inputs = (
            f"{provider_choice}\n{disk_size}\n{image_choice}\n\ny\n"  # Empty line for default [1]
        )
        hud_console.debug("  Using personal account: default option [1]")

    hud_console.debug(
        f"  Input sequence: provider={provider_choice}, disk={disk_size or 'default'}, image={image_choice}, team={team_choice if 'team_choice' in locals() else 'default'}"  # noqa: E501
    )

    # Show found providers
    if provider_lines:
        hud_console.info("")
        hud_console.info("Found providers:")
        for pl in provider_lines[:5]:  # Show first 5
            hud_console.info(f"  {pl}")

    try:
        console = Console()

        with Live(
            Spinner("dots", text="[bold]Creating pod...[/bold]", style="gold"),
            console=console,
            refresh_per_second=10,
        ):
            result = subprocess.run(  # noqa: S603, ASYNC221
                create_cmd,
                input=inputs,
                text=True,
                capture_output=True,
            )

        if result.returncode != 0:
            hud_console.error("Failed to create pod")

            # Parse output for better error reporting
            output_lines = result.stdout.strip().split("\n") if result.stdout else []

            # Look for provider prices
            for line in output_lines:
                if "$" in line and "/hr" in line:
                    hud_console.info(f"Provider option: {line.strip()}")

            # Check for team selection error
            if "invalid selection" in result.stdout.lower():
                hud_console.error("Team selection failed")
                # Find and display the team selection section
                for i, line in enumerate(output_lines):
                    if "Select Team:" in line:
                        hud_console.info("Team selection options:")
                        # Show next few lines
                        for j in range(i, min(i + 6, len(output_lines))):
                            hud_console.info(f"  {output_lines[j]}")
                        break

                hud_console.info("")
                hud_console.hint(
                    "The Prime CLI interface may have changed. Try running the command manually:"
                )
                hud_console.command_example(
                    f"prime pods create --gpu-type {gpu_type} --gpu-count {gpu_count} --name {pod_name}"  # noqa: E501
                )

            # Show error details
            if result.stderr:
                hud_console.error("Error output:")
                for line in result.stderr.strip().split("\n"):
                    hud_console.error(f"  {line}")

            # Show last part of stdout for context
            if result.stdout:
                hud_console.info("Command output:")
                # Show last 15 lines for brevity
                for line in output_lines[-15:]:
                    hud_console.info(f"  {line}")

            if "max_price" in str(result.stderr) or "max_price" in str(result.stdout):
                hud_console.warning("")
                hud_console.warning("The selected provider requires a maximum price limit.")
                hud_console.info("This is a known limitation with some providers.")
                hud_console.info("")
                hud_console.hint("Workarounds:")
                hud_console.info("1. Run the command manually and select a different provider")
                hud_console.info(
                    "2. Try again later when datacrunch (usually cheapest) is available"
                )
                hud_console.info("3. Use the Prime web interface: https://app.primeintellect.ai")

            hud_console.info("")
            hud_console.info("Debug info:")
            hud_console.info(f"  Command: {' '.join(create_cmd)}")
            hud_console.info(f"  Pod name: {pod_name}")
            hud_console.info(f"  Team ID: {'Provided' if team_id else 'Not provided'}")
            hud_console.info(f"  Global team detected: {has_global_team}")

            # Show the exact inputs we sent
            hud_console.info("  Inputs sent (in order):")
            input_parts = inputs.strip().split("\n")
            input_labels = [
                "Provider selection",
                "Disk size",
                "Image selection",
                "Team selection",
                "Team ID (if custom)",
                "Confirmation",
            ]
            for i, (part, label) in enumerate(zip(input_parts, input_labels, strict=False)):
                if part:
                    hud_console.info(f"    {i + 1}. {label}: '{part}'")
                else:
                    hud_console.info(f"    {i + 1}. {label}: [Enter/default]")

            raise typer.Exit(1)

        # Extract pod ID from output
        output_lines = result.stdout.strip().split("\n")
        pod_id = None
        for line in output_lines:
            if "Successfully created pod" in line:
                # Extract just the pod ID (alphanumeric characters)
                match = re.search(r"pod\s+([a-f0-9]+)", line)
                pod_id = match.group(1) if match else line.split()[-1].strip()
                break

        if not pod_id:
            hud_console.error("Could not extract pod ID from output")
            hud_console.info(f"Output: {result.stdout}")
            raise typer.Exit(1)

        hud_console.success(f"Created pod: {pod_id}")

        # Poll for pod status
        ssh_info = await poll_pod_status(pod_id)

        if ssh_info:
            hud_console.success("Pod is ready!")
            hud_console.info(f"SSH: {ssh_info}")

            # Check if SSH key is configured globally
            ssh_key_configured = await check_and_configure_ssh_key()

            if ssh_key_configured:
                # Automatically connect and run training
                await connect_and_train(
                    pod_id=pod_id,
                    ssh_info=ssh_info,
                    model=model,
                    dataset=dataset,
                    config=config,
                    output_dir=output_dir,
                    image=image,
                    dataset_size=dataset_size,
                    is_json_file=is_json_file,
                )
            else:
                # Manual fallback
                hud_console.section_title("📋 Manual Connection Required")
                hud_console.info("SSH key configuration failed. Connect manually:")
                hud_console.info("")
                hud_console.info("1. Download the SSH key from:")
                hud_console.info("   https://app.primeintellect.ai/dashboard/profile")
                hud_console.info("")
                hud_console.info("2. Set permissions:")
                hud_console.command_example("chmod 400 /path/to/prime-key.pem", "")
                hud_console.info("")
                hud_console.info("3. Connect to your instance:")
                hud_console.command_example(f"ssh -i /path/to/prime-key.pem {ssh_info}", "")
                hud_console.info("")
                hud_console.info("4. Run these commands:")
                hud_console.command_example("pip install verifiers hud-vf-gym", "")
                hud_console.command_example(f"prime env install {image}", "")

                # Build training command with env vars
                if settings.wandb_api_key:
                    hud_console.command_example(
                        f"export WANDB_API_KEY={settings.wandb_api_key}", ""
                    )

                train_cmd = f"""vf-train hud-vf-gym \\
  --model {model} \\
  --env-args '{{"taskset": "{dataset}", "config_path": "/root/config.yaml"}}' \\
  --output-dir {output_dir} \\
  --run-name hud-rl-{pod_id[:8]} \\
  --wandb-project hud-rl"""

                hud_console.command_example(train_cmd, "")
                hud_console.info("")
                hud_console.warning(
                    f"Remember to terminate when done: prime pods terminate {pod_id}"
                )
        else:
            hud_console.error("Pod failed to become active")
            raise typer.Exit(1)

    except subprocess.CalledProcessError as e:
        hud_console.error(f"Failed to create pod: {e}")
        raise typer.Exit(1) from e


async def poll_pod_status(pod_id: str) -> str | None:
    """Poll pod status until SSH is available."""
    console = Console()
    max_attempts = 120  # 20 minutes with 10s intervals
    attempt = 0

    # Create spinner
    spinner = Spinner(
        "dots", text="Waiting for pod to become active (should take 5-20 min)...", style="gold"
    )

    with Live(spinner, console=console, refresh_per_second=10) as live:
        while attempt < max_attempts:
            try:
                # Update check frequency in spinner text every minute
                if attempt % 6 == 0:  # Every minute
                    pass  # Will update in spinner text below

                result = subprocess.run(  # noqa: S603, ASYNC221
                    ["prime", "pods", "status", pod_id],  # noqa: S607
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    output = result.stdout
                    elapsed_minutes = (attempt * 10) // 60

                    # Parse status - look for lines with Status and SSH
                    lines = output.split("\n")
                    status_value = None
                    ssh_value = None

                    for line in lines:
                        # Handle both regular pipes | and box-drawing chars │
                        if "|" in line or "│" in line:
                            # Split by either type of pipe
                            separator = "│" if "│" in line else "|"
                            parts = [p.strip() for p in line.split(separator)]

                            if len(parts) >= 3:
                                key = parts[1].strip()
                                value = parts[2].strip()

                                if key == "Status":
                                    status_value = value
                                elif key == "SSH":
                                    ssh_value = value

                    # Update spinner text with current status
                    if status_value:
                        # Include SSH status in spinner text
                        ssh_status = f" | SSH: {ssh_value}" if ssh_value else ""
                        spinner.text = f"Pod status: {status_value} ({elapsed_minutes}m elapsed, should take 5-20 min){ssh_status}"  # noqa: E501

                    # Check if SSH is available (and not N/A)
                    if ssh_value and ssh_value.strip() and ssh_value.strip() != "N/A":
                        # Stop the spinner before logging
                        live.stop()
                        hud_console.success(f"SSH is available: {ssh_value}")
                        return ssh_value

                time.sleep(10)  # Wait 10 seconds # noqa: ASYNC251
                attempt += 1

            except Exception as e:
                spinner.text = f"[bold red]Status check failed: {e}[/bold red]"
                time.sleep(10)  # noqa: ASYNC251
                attempt += 1

    # Spinner is done, now we can use hud_console.error
    hud_console.error("Timeout: Pod did not become ready within 20 minutes")
    return None


async def run_prime_training(
    model: str,
    dataset: str,
    config: Path,
    gpus: str,
    output_dir: Path,
    image: str,
    auto_create_pod: str | None = None,
    team_id: str | None = None,
    dataset_size: int | None = None,
    is_json_file: bool = False,
) -> None:
    """Run training on Prime Intellect infrastructure."""
    # Check API key
    if not settings.prime_api_key:
        hud_console.error("Prime API key not found")
        hud_console.info("Set your Prime API key:")
        hud_console.info("  export PRIME_API_KEY='your-api-key'")
        hud_console.info("  # or")
        hud_console.info("  prime auth")
        raise typer.Exit(1)

    # Parse GPU configuration
    gpu_count, gpu_type = parse_gpu_config(gpus)

    # Generate short pod name (no dots allowed)
    model_suffix = model.split("/")[-1].replace(".", "-").lower()
    short_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))  # noqa: S311
    pod_name = f"hud-rl-{model_suffix}-{short_id}"[:30]  # Keep it short

    # Always create pod automatically
    await create_and_connect_prime_pod(
        pod_name=pod_name,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        model=model,
        dataset=dataset,
        config=config,
        output_dir=output_dir,
        image=image,
        team_id=team_id,
        dataset_size=dataset_size,
        is_json_file=is_json_file,
    )
