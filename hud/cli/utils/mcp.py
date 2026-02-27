"""MCP utilities for CLI commands."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import Client

logger = logging.getLogger(__name__)


async def analyze_environment(
    client: Client, verbose: bool = False, server_name: str | None = None
) -> dict[str, Any]:
    """Analyze an MCP environment via a connected client.

    Args:
        client: An initialized fastmcp.Client
        verbose: Enable verbose logging
        server_name: Optional server name for display

    Returns:
        Dictionary containing tools, hub_tools, resources, prompts, scenarios, telemetry
    """
    servers = [server_name] if server_name else []
    analysis: dict[str, Any] = {
        "tools": [],
        "hub_tools": {},
        "resources": [],
        "prompts": [],
        "scenarios": [],
        "verbose": verbose,
        "metadata": {"initialized": True, "servers": servers},
    }

    # Get all tools with schemas
    tools = await client.list_tools()
    for tool in tools:
        tool_info = {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.inputSchema,
        }
        analysis["tools"].append(tool_info)

        # Check if this is a hub tool (like setup, evaluate)
        if (
            tool.description
            and "internal" in tool.description.lower()
            and "functions" in tool.description.lower()
        ):
            hub_functions = await _get_hub_tools(client, tool.name, verbose)
            if hub_functions:
                analysis["hub_tools"][tool.name] = hub_functions

    # Get all resources
    try:
        resources = await client.list_resources()
        for resource in resources:
            resource_info: dict[str, Any] = {
                "uri": str(resource.uri),
                "name": resource.name,
                "description": resource.description,
                "mime_type": getattr(resource, "mimeType", None),
            }
            meta = getattr(resource, "meta", None)
            if meta:
                resource_info["meta"] = meta
            analysis["resources"].append(resource_info)
    except Exception as e:
        if verbose:
            logger.debug("Could not list resources: %s", e)

    # Get all prompts
    try:
        prompts = await client.list_prompts()
        for prompt in prompts:
            raw_args = getattr(prompt, "arguments", []) or []
            args: list[dict[str, Any]] = [
                {
                    "name": getattr(a, "name", None),
                    "required": getattr(a, "required", None),
                    "description": getattr(a, "description", None),
                }
                for a in raw_args
            ]

            prompt_info: dict[str, Any] = {
                "name": prompt.name,
                "description": prompt.description,
                "arguments": args,
            }
            meta = getattr(prompt, "meta", None)
            if meta:
                prompt_info["meta"] = meta
                if isinstance(meta, dict) and "arguments" in meta:
                    meta_args = {a["name"]: a for a in meta["arguments"] if "name" in a}
                    for arg in args:
                        arg_name = arg.get("name")
                        if arg_name and arg_name in meta_args:
                            meta_arg = meta_args[arg_name]
                            if "default" in meta_arg:
                                arg["default"] = meta_arg["default"]
                            if "type" in meta_arg:
                                arg["type"] = meta_arg["type"]
                            if "inputSchema" in meta_arg:
                                arg["inputSchema"] = meta_arg["inputSchema"]
            analysis["prompts"].append(prompt_info)
    except Exception as e:
        if verbose:
            logger.debug("Could not list prompts: %s", e)

    # Derive scenarios from prompt/resource pairs
    analysis["scenarios"] = _derive_scenarios(analysis)

    return analysis


async def _get_hub_tools(client: Client, hub_name: str, verbose: bool) -> list[str]:
    """Get subtools for a hub (setup/evaluate)."""
    try:
        result = await client.read_resource(f"file:///{hub_name}/functions")
        if result:
            content = result[0] if result else None
            text = getattr(content, "text", None) if content else None
            if text:
                return json.loads(text)
    except Exception as e:
        if verbose:
            logger.debug("Could not read hub functions for '%s': %s", hub_name, e)
    return []


def _derive_scenarios(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive scenarios from prompt/resource pairs."""
    scenarios_by_id: dict[str, dict[str, Any]] = {}

    for p in analysis.get("prompts", []):
        desc = (p.get("description") or "").strip()
        if not desc.startswith("[Setup]"):
            continue
        scenario_id = p.get("name")
        if not scenario_id:
            continue
        env_name, scenario_name = ([*scenario_id.split(":", 1), ""])[:2]
        scenario_info: dict[str, Any] = {
            "id": scenario_id,
            "env": env_name,
            "name": scenario_name or scenario_id,
            "setup_description": desc,
            "arguments": p.get("arguments") or [],
            "has_setup_prompt": True,
            "has_evaluate_resource": False,
        }
        meta = p.get("meta")
        if meta and isinstance(meta, dict):
            if "code" in meta:
                scenario_info["code"] = meta["code"]
            et = meta.get("exclude_tools")
            if isinstance(et, list):
                scenario_info["exclude_tools"] = [x for x in et if isinstance(x, str)]
            es = meta.get("exclude_sources")
            if isinstance(es, list):
                scenario_info["exclude_sources"] = [x for x in es if isinstance(x, str)]
        scenarios_by_id[scenario_id] = scenario_info

    for r in analysis.get("resources", []):
        desc = (r.get("description") or "").strip()
        if not desc.startswith("[Evaluate]"):
            continue
        scenario_id = r.get("uri")
        if not scenario_id:
            continue
        env_name, scenario_name = ([*scenario_id.split(":", 1), ""])[:2]
        if scenario_id not in scenarios_by_id:
            scenarios_by_id[scenario_id] = {
                "id": scenario_id,
                "env": env_name,
                "name": scenario_name or scenario_id,
                "arguments": [],
                "has_setup_prompt": False,
                "has_evaluate_resource": True,
            }
        scenarios_by_id[scenario_id]["evaluate_description"] = desc
        scenarios_by_id[scenario_id]["has_evaluate_resource"] = True
        meta = r.get("meta")
        if (
            meta
            and isinstance(meta, dict)
            and "code" in meta
            and "code" not in scenarios_by_id[scenario_id]
        ):
            scenarios_by_id[scenario_id]["code"] = meta["code"]

    return sorted(
        scenarios_by_id.values(),
        key=lambda s: (str(s.get("env") or ""), str(s.get("name") or "")),
    )
