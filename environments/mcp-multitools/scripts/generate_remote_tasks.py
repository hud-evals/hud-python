#!/usr/bin/env python3
"""
Generate final_tasks_remote.json from final_tasks.json.

Converts local Docker mcp_config to remote HUD mcp_config for cloud execution.
Filters out tasks with non-HUD MCP servers (e.g., Supabase MCP).
"""

import json
from pathlib import Path

# Configuration
DOCKER_IMAGE = "reinissalaks/mcp-multitools:0.1.1"
INPUT_FILE = "final_tasks.json"
OUTPUT_FILE = "final_tasks_remote.json"

# Environment variables to forward to the container
# Pattern: ENV_VAR_NAME -> Env-Var-Name header
ENV_VARS_TO_FORWARD = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "EXA_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_API_KEY",
    "LINEAR_API_KEY",
]


def env_var_to_header(env_var: str) -> str:
    """Convert ENV_VAR_NAME to Env-Var-Name header format."""
    parts = env_var.lower().split("_")
    return "Env-" + "-".join(part.capitalize() for part in parts)


def has_non_hud_mcp_server(mcp_config: dict) -> tuple[bool, str | None]:
    """Check if task has MCP servers pointing to non-HUD URLs."""
    for server_name, config in mcp_config.items():
        if server_name == "local":
            continue  # local will be converted to hud
        if "url" in config:
            url = config["url"]
            if not (url.startswith("https://mcp.hud.ai") or url.startswith("https://mcp.hud.so")):
                return True, server_name
    return False, None


def convert_to_remote_config(task: dict) -> dict:
    """Convert a task's mcp_config from local to remote."""
    task = task.copy()
    mcp_config = task.get("mcp_config", {}).copy()
    
    # Remove the local config
    if "local" in mcp_config:
        del mcp_config["local"]
    
    # Build headers with env var forwarding
    headers = {
        "Authorization": "Bearer ${HUD_API_KEY}",
        "Mcp-Image": DOCKER_IMAGE,
    }
    
    # Add env var forwarding headers
    for env_var in ENV_VARS_TO_FORWARD:
        header_name = env_var_to_header(env_var)
        headers[header_name] = f"${{{env_var}}}"
    
    # Add the HUD remote config
    mcp_config["hud"] = {
        "url": "https://mcp.hud.ai/v3/mcp",
        "headers": headers
    }
    
    task["mcp_config"] = mcp_config
    return task


def main():
    # Paths relative to this script
    script_dir = Path(__file__).parent
    task_jsons_dir = script_dir.parent / "task_jsons"
    
    input_path = task_jsons_dir / INPUT_FILE
    output_path = task_jsons_dir / OUTPUT_FILE
    
    # Read input
    print(f"Reading: {input_path}")
    with open(input_path, "r") as f:
        tasks = json.load(f)
    
    print(f"Found {len(tasks)} tasks")
    
    # Convert all tasks (keep non-HUD MCP servers as-is, just add hud config for local)
    remote_tasks = []
    tasks_with_external_mcp = []
    
    for task in tasks:
        mcp_config = task.get("mcp_config", {})
        has_external, server_name = has_non_hud_mcp_server(mcp_config)
        
        if has_external:
            tasks_with_external_mcp.append((task["id"], server_name))
        
        remote_tasks.append(convert_to_remote_config(task))
    
    # Report tasks with external MCP servers (warning only, not filtered)
    if tasks_with_external_mcp:
        print(f"\n⚠️  Note: {len(tasks_with_external_mcp)} tasks use non-HUD MCP servers")
        print(f"   (These won't work with --remote flag, run locally instead)")
        for task_id, server in tasks_with_external_mcp[:5]:
            print(f"   - {task_id} (uses '{server}' MCP server)")
        if len(tasks_with_external_mcp) > 5:
            print(f"   ... and {len(tasks_with_external_mcp) - 5} more")
        print()
    
    # Write output
    print(f"Writing: {output_path}")
    with open(output_path, "w") as f:
        json.dump(remote_tasks, f, indent=2)
    
    print(f"✅ Generated {OUTPUT_FILE} with {len(remote_tasks)} tasks")
    print(f"   Image: {DOCKER_IMAGE}")
    print(f"\n   Env vars forwarded:")
    for env_var in ENV_VARS_TO_FORWARD:
        print(f"     - {env_var} → {env_var_to_header(env_var)}")


if __name__ == "__main__":
    main()

