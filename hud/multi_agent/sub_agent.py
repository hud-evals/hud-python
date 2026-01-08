"""SubAgent utilities for multi-agent systems.

This module provides configuration and factory functions for creating
sub-agents from any MCPAgent implementation.

The core insight: Any MCPAgent already implements the sub-agent protocol
via `run(ctx, max_steps) -> Trace`. This module provides utilities to:
- Configure which agent type to create (Claude, OpenAI, etc.)
- Create agents from configuration
- Format results for parent context (token-optimized)
- Generate structured JSON output matching return schemas
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hud.multi_agent.schemas import SubAgentResult
from hud.types import BaseAgentConfig

if TYPE_CHECKING:
    from hud.agents.base import MCPAgent
    from hud.environment import Environment

logger = logging.getLogger(__name__)


class SubAgentConfig(BaseAgentConfig):
    """Configuration for creating a sub-agent from any MCPAgent.

    Extends BaseAgentConfig with sub-agent specific settings.
    The `model` and `system_prompt` fields are inherited.

    Example:
        config = SubAgentConfig(
            name="coder",
            model="claude-sonnet-4-5",
            system_prompt="You are a Python expert.",
            max_steps=10,
            return_schema=CodeResult,
        )
        agent = create_sub_agent(config, parent_ctx)
        trace = await agent.run(parent_ctx, max_steps=config.max_steps)
    """

    # Sub-agent identity
    name: str = "sub_agent"

    # Execution limits (model and system_prompt inherited from BaseAgentConfig)
    max_steps: int = 10

    # Tool filtering - only allow these tools for this sub-agent
    tools: list[str] = []

    # Return schema class for structured output (e.g., CodeResult, ResearchResult)
    # If None, uses GenericResult
    return_schema: type[BaseModel] | None = None


def generate_json_instructions(schema: type[BaseModel]) -> str:
    """Generate instructions for returning structured JSON output.
    
    Creates a prompt suffix that instructs the agent to return JSON
    matching the Pydantic schema.
    
    Args:
        schema: Pydantic model class
        
    Returns:
        Instruction string to append to system prompt
    """
    # Get schema JSON for the model
    schema_json = schema.model_json_schema()
    
    # Extract required and optional fields
    properties = schema_json.get("properties", {})
    required = set(schema_json.get("required", []))
    
    # Build field descriptions
    field_lines = []
    for name, prop in properties.items():
        field_type = prop.get("type", "any")
        description = prop.get("description", "")
        is_required = name in required
        req_marker = "(required)" if is_required else "(optional)"
        
        if description:
            field_lines.append(f"  - {name}: {field_type} {req_marker} - {description}")
        else:
            field_lines.append(f"  - {name}: {field_type} {req_marker}")
    
    fields_text = "\n".join(field_lines)
    
    return f"""

## Required Output Format

When you complete your task, you MUST end your response with a JSON block containing your structured result.

**Schema: {schema.__name__}**
Fields:
{fields_text}

**Format your final response like this:**

```json
{{
  "success": true,
  "your_main_output_field": "your result here",
  ... other fields ...
}}
```

The JSON block must be valid JSON and must be the last thing in your response."""


def parse_json_from_output(output: str) -> dict[str, Any] | None:
    """Extract JSON from agent output.
    
    Tries multiple strategies:
    1. Look for ```json code block
    2. Look for any JSON object at the end
    3. Try parsing the entire output as JSON
    
    Args:
        output: Agent's text output
        
    Returns:
        Parsed dict or None if no valid JSON found
    """
    if not output:
        return None
    
    # Strategy 1: Look for ```json code block
    json_block_pattern = r"```json\s*([\s\S]*?)\s*```"
    matches = re.findall(json_block_pattern, output)
    if matches:
        # Take the last JSON block (most likely the final result)
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Look for JSON object at the end of output
    # Find the last { and try to parse from there
    last_brace = output.rfind("{")
    if last_brace != -1:
        try:
            return json.loads(output[last_brace:])
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try parsing entire output as JSON
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError:
        pass
    
    return None


def create_sub_agent(
    config: SubAgentConfig,
    ctx: Environment | None = None,
    **overrides: Any,
) -> MCPAgent:
    """Factory to create any MCPAgent as a sub-agent.

    Creates the appropriate MCPAgent implementation based on the model
    specified in config. Any MCPAgent can be used as a sub-agent since
    they all implement `run(ctx, max_steps) -> Trace`.

    If a return_schema is specified, appends JSON output instructions
    to the system prompt so the agent returns structured data.

    Args:
        config: SubAgentConfig with model and settings
        ctx: Optional Environment (or EvalContext) to bind to the agent
        **overrides: Override any config fields

    Returns:
        MCPAgent instance ready to run

    Example:
        config = SubAgentConfig(name="coder", model="claude-sonnet-4-5")
        agent = create_sub_agent(config, ctx)
        trace = await agent.run(ctx, max_steps=10)
    """
    # Apply overrides
    model = overrides.get("model", config.model) or "claude-sonnet-4-5"
    system_prompt = overrides.get("system_prompt", config.system_prompt) or ""
    
    # Append JSON output instructions if return schema is specified
    return_schema = overrides.get("return_schema", config.return_schema)
    if return_schema is not None:
        json_instructions = generate_json_instructions(return_schema)
        system_prompt = system_prompt + json_instructions
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Added JSON instructions for agent '{config.name}' with schema '{return_schema.__name__}' ({len(json_instructions)} chars)")

    # Use create_agent() to route through HUD gateway
    from hud.agents import create_agent

    return create_agent(
        model=model,
        ctx=ctx,
        system_prompt=system_prompt,
        **{k: v for k, v in overrides.items() if k not in ("model", "system_prompt", "return_schema")},
    )


__all__ = [
    "SubAgentConfig",
    "SubAgentResult",
    "create_sub_agent",
    "generate_json_instructions",
    "parse_json_from_output",
]
