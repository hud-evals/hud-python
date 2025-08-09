# HUD MCP Client Architecture

This directory contains the MCP client implementations for HUD SDK. The architecture is designed to be flexible and extensible, allowing different client implementations while maintaining a consistent interface for agents.

## Architecture Overview

```
hud/clients/
├── base.py          # Protocol definition and base class
├── mcp_use.py       # MCP-use based implementation (legacy)
├── fastmcp.py       # FastMCP based implementation (modern)
└── __init__.py      # Exports and default client
```

## Protocol Definition

All clients must implement the `AgentMCPClient` protocol:

```python
class AgentMCPClient(Protocol):
    async def initialize(self) -> None:
        """Initialize the client - connect and fetch telemetry."""
    
    async def list_tools(self) -> list[types.Tool]:
        """List all available tools."""
    
    async def call_tool(name: str, arguments: dict | None = None) -> types.CallToolResult:
        """Execute a tool by name."""
```

## Available Implementations

### 1. MCPUseHUDClient (Default)
- Based on the `mcp_use` library
- Supports multiple concurrent server connections
- Battle-tested and stable
- Good for complex multi-server setups

### 2. FastMCPHUDClient
- Based on the `fastmcp` library
- Modern, clean API with better error handling
- Supports various transports (HTTP, WebSocket, stdio, in-memory)
- Better type safety and structured data support

## Usage Examples

### Basic Usage

```python
from hud.clients import MCPUseHUDClient, FastMCPHUDClient

# Configuration works for both clients
mcp_config = {
    "server_name": {
        "command": "python",
        "args": ["server.py"]
    }
}

# Option 1: MCP-use client
client = MCPUseHUDClient(mcp_config)

# Option 2: FastMCP client
client = FastMCPHUDClient(mcp_config)

# Both use the same API
async with client:
    tools = await client.list_tools()
    result = await client.call_tool("tool_name", {"arg": "value"})
```

### With Agents

```python
from hud.agents import ClaudeMCPAgent

# Either client works with agents
client = FastMCPHUDClient(mcp_config)

agent = ClaudeMCPAgent(
    mcp_client=client,
    model="claude-3-7-sonnet-20250219"
)

# Agent works identically with either client
result = await agent.run("Your task here")
```

## Choosing a Client

### Use MCPUseHUDClient when:
- You need proven stability
- Working with multiple MCP servers
- Migrating existing code
- Need specific mcp_use features

### Use FastMCPHUDClient when:
- Starting a new project
- Want modern API features
- Need better error handling
- Want structured data support
- Using in-memory servers for testing

## Adding New Clients

To add a new client implementation:

1. Inherit from `BaseHUDClient`
2. Implement the required methods:
   - `_connect()` - Establish connection
   - `list_tools()` - List available tools
   - `call_tool()` - Execute tools
   - `_read_resource_internal()` - Read resources

3. The base class handles:
   - Initialization flow
   - Telemetry fetching
   - Verbose logging
   - Common HUD features

## Backward Compatibility

For backward compatibility, `hud.client.MCPClient` imports the default client:

```python
# Old code continues to work
from hud.client import MCPClient
```

This imports `MCPUseHUDClient` by default. To change the default, modify `hud/clients/__init__.py`.

## Future Extensions

The protocol-based design allows for:
- Mock clients for testing
- Remote clients (gRPC, REST)
- Cached/offline clients
- Client decorators (retry, logging, metrics)

All without changing agent code!
