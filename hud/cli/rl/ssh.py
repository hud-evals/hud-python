"""SSH key configuration and connection utilities for Prime Intellect."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer

from hud.settings import settings
from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


async def check_and_configure_ssh_key() -> bool:
    """Check if SSH key is configured, prompt for it if not."""
    # Check current SSH key configuration
    result = subprocess.run(  # noqa: ASYNC221
        ["prime", "config", "view"],  # noqa: S607
        capture_output=True,
        text=True,
    )

    ssh_key_path = None
    if result.returncode == 0:
        # Parse the output for SSH key path
        for line in result.stdout.split("\n"):
            if "SSH Key Path" in line:
                # Handle table format: "| SSH Key Path        | C:\\Users\\saecl\\.ssh\\private_key.pem |" # noqa: E501
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        path = parts[2].strip()
                        if path and path != "None":
                            ssh_key_path = path
                            break
                # Handle simple format: "SSH Key Path: /path/to/key"
                elif ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        path = parts[1].strip()
                        if path and path != "None":
                            ssh_key_path = path
                            break

    # If SSH key is configured, verify it exists
    if ssh_key_path:
        if Path(ssh_key_path).expanduser().exists():
            hud_console.info(f"Using configured SSH key: {ssh_key_path}")
            return True
        else:
            hud_console.warning(f"Configured SSH key not found: {ssh_key_path}")

    # Prompt for SSH key
    hud_console.section_title("🔑 SSH Key Configuration")
    hud_console.info("Prime Intellect requires an SSH key for pod access.")
    hud_console.info("")
    hud_console.info("If you don't have a key:")
    hud_console.info("1. Visit https://app.primeintellect.ai/dashboard/profile")
    hud_console.info("2. Generate or upload your SSH key")
    hud_console.info("3. Download the private key file")
    hud_console.info("")

    key_path = typer.prompt("Enter path to your Prime SSH private key (e.g., ~/.ssh/prime-key.pem)")
    key_path = Path(key_path).expanduser()

    if not key_path.exists():
        hud_console.error(f"File not found: {key_path}")
        return False

    # Set permissions if not Windows
    if os.name != "nt":
        subprocess.run(["chmod", "400", str(key_path)])  # noqa: S603, S607, ASYNC221
        hud_console.success("Set proper permissions on key file")

    # Configure the SSH key globally
    result = subprocess.run(  # noqa: S603, ASYNC221
        ["prime", "config", "set-ssh-key-path", str(key_path)],  # noqa: S607
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        hud_console.success("SSH key configured successfully")
        return True
    else:
        hud_console.error("Failed to configure SSH key")
        if result.stderr:
            hud_console.error(f"Error: {result.stderr}")
        return False


async def connect_and_train(
    pod_id: str,
    ssh_info: str,
    model: str,
    dataset: str,
    config: Path,
    output_dir: Path,
    image: str,
    dataset_size: int | None = None,
    is_json_file: bool = False,
) -> None:
    """Connect to the pod via SSH and run training commands."""
    hud_console.section_title("🚀 Starting Remote Training")

    # Parse SSH info to get host and port
    # Format is like "root@65.108.33.78 -p 1234"
    ssh_parts = ssh_info.split()
    ssh_user_host = ssh_parts[0]  # root@65.108.33.78
    ssh_port = ssh_parts[2] if len(ssh_parts) > 2 else "22"  # 1234 or default 22

    # Get SSH key path from Prime config
    result = subprocess.run(  # noqa: ASYNC221
        ["prime", "config", "view"],  # noqa: S607
        capture_output=True,
        text=True,
    )

    ssh_key_path = None
    if result.returncode == 0:
        for line in result.stdout.split("\n"):
            if "SSH Key Path" in line:
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        ssh_key_path = parts[2].strip()
                        break
                elif ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        ssh_key_path = parts[1].strip()
                        break

    if not ssh_key_path:
        hud_console.error("SSH key path not configured")
        raise typer.Exit(1)

    # Verify SSH key exists
    ssh_key_path = Path(ssh_key_path).expanduser()
    if not ssh_key_path.exists():
        hud_console.error(f"SSH key not found: {ssh_key_path}")
        raise typer.Exit(1)

    hud_console.info(f"Using SSH key: {ssh_key_path}")

    # First, copy the config file to the pod using scp
    hud_console.info("Copying config file to pod...")
    try:
        # On Windows, we need to ensure proper path formatting
        config_path = str(config).replace("\\", "/")
        scp_cmd = [
            "scp",
            "-i",
            str(ssh_key_path),
            "-P",
            ssh_port,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            config_path,
            f"{ssh_user_host}:/root/config.yaml",
        ]
        hud_console.debug(f"Running: {' '.join(scp_cmd)}")
        subprocess.run(scp_cmd, check=True)  # noqa: S603, ASYNC221
        hud_console.success("Config file copied")
    except subprocess.CalledProcessError as e:
        hud_console.error(f"Failed to copy config file: {e}")
        if os.name == "nt":  # Windows
            hud_console.info("Make sure OpenSSH is installed. On Windows 10+, it's built-in.")
            hud_console.info(
                "If using older Windows, install Git for Windows which includes SSH/SCP."
            )
        else:
            hud_console.info("Make sure scp is installed and in your PATH")
        raise typer.Exit(1) from e

    # If dataset is a JSON file, copy it too
    remote_dataset = dataset  # Default to unchanged
    if is_json_file:
        hud_console.info("Copying task file to pod...")
        try:
            # On Windows, we need to ensure proper path formatting
            dataset_path = str(dataset).replace("\\", "/")
            # Extract just the filename for the remote path
            dataset_filename = os.path.basename(dataset)
            remote_dataset = f"/root/{dataset_filename}"

            scp_cmd = [
                "scp",
                "-i",
                str(ssh_key_path),
                "-P",
                ssh_port,
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                dataset_path,
                f"{ssh_user_host}:{remote_dataset}",
            ]
            hud_console.debug(f"Running: {' '.join(scp_cmd)}")
            subprocess.run(scp_cmd, check=True)  # noqa: S603, ASYNC221
            hud_console.success(f"Task file copied to {remote_dataset}")
        except subprocess.CalledProcessError as e:
            hud_console.error(f"Failed to copy task file: {e}")
            raise typer.Exit(1) from e

    hud_console.info("Setting up environment and starting training...")
    hud_console.info("This will take a few minutes for initial setup, then training will begin.")
    hud_console.info("")

    # Build environment exports
    env_exports = []
    wandb_key = getattr(settings, "wandb_api_key", None)
    if wandb_key:
        env_exports.append(f"export WANDB_API_KEY={wandb_key}")
    if settings.api_key:  # HUD API key
        env_exports.append(f"export HUD_API_KEY={settings.api_key}")
    env_export_cmd = " && ".join(env_exports) + " && " if env_exports else ""

    # Create the training script content using echo commands
    # This is more reliable than heredoc through SSH
    training_script_lines = [
        "import verifiers as vf",
        "",
        "# Load environment",
        "env = vf.load_environment(",
        '    env_id="hud-vf-gym",',
        f'    taskset="{remote_dataset}",',
        '    config_path="/root/config.yaml",',
        f"    num_tasks={dataset_size},",
        ")",
        "",
        'print(f"Loaded environment with {len(env.dataset)} tasks")',
        "",
        "# Load model and tokenizer",
        f'model, tokenizer = vf.get_model_and_tokenizer("{model}")',
        "",
        "# Get default training args",
        f'args = vf.grpo_defaults(run_name="hud-rl-{pod_id[:8]}")',
        f'args.output_dir = "{output_dir}"',
        'args.wandb_project = "hud-rl"',
        "args.logging_steps = 1",
        "",
        "# Create trainer",
        "trainer = vf.GRPOTrainer(",
        "    model=model,",
        "    processing_class=tokenizer,",
        "    env=env,",
        "    args=args,",
        "    peft_config=vf.lora_defaults(),",
        ")",
        "",
        "# Train",
        'print("Starting training...")',
        "trainer.train()",
    ]

    # Create echo commands for each line
    # First remove any existing file, then create new one
    training_script = "rm -f /root/train_hud_rl.py && " + " && ".join(
        [f"echo {line!r} >> /root/train_hud_rl.py" for line in training_script_lines]
    )

    # Build the full setup and training command
    full_command = (
        # Install uv
        "curl -LsSf https://astral.sh/uv/install.sh | sh && "
        'source "$HOME/.local/bin/env" && '
        # Install prime CLI and create venv
        "uv tool install prime && "
        "uv venv --python 3.12 && "
        "source .venv/bin/activate && "
        # Install packages
        "prime env install hud/hud-vf-gym@0.1.1 && "
        "uv pip install 'verifiers[train]' && "
        "uv pip install flash-attn --no-build-isolation && "
        # Set environment variables
        f"{env_export_cmd}"
        # Create the training script
        f"{training_script} && "
        "echo '✓ Training script created' && "
        # Start vLLM server in tmux (on GPU 0)
        f"tmux new-session -d -s vllm-server 'CUDA_VISIBLE_DEVICES=0 vf-vllm --model {model} --enforce-eager --disable-log-requests' && "  # noqa: E501
        "echo '✓ vLLM server started in tmux' && "
        # Wait a bit for server to start
        "echo 'Waiting for vLLM server to initialize...' && "
        "sleep 10 && "
        # Run training on GPU 1
        "echo 'Starting training on GPU 1...' && "
        "CUDA_VISIBLE_DEVICES=1 python /root/train_hud_rl.py"
    )

    try:
        # Execute the full command via SSH
        ssh_cmd = [
            "ssh",
            "-i",
            str(ssh_key_path),
            "-p",
            ssh_port,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            ssh_user_host,
            full_command,
        ]
        subprocess.run(ssh_cmd, check=True)  # noqa: S603, ASYNC221

    except subprocess.CalledProcessError as e:
        hud_console.error(f"Training failed: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        hud_console.warning("Training interrupted by user")
        hud_console.info(f"To reconnect: prime pods ssh {pod_id}")
        hud_console.info(f"To check status: prime pods status {pod_id}")
        hud_console.info(f"To terminate: prime pods terminate {pod_id}")
