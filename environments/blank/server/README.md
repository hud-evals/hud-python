# MCP Server

MCP layer that wraps environment data in tools for agent interaction.

## Structure

- `main.py` - Server initialization, imports routers
- `tools.py` - MCP tools that call environment HTTP endpoints

## Development

```bash
# Start MCP server with hot-reload
uv run hud dev
```

## Key Principles

- Keep tools thin - call environment HTTP endpoints
- Use routers for organization
- All long-lived state lives in `environment/`, not here