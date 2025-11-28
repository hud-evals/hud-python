# HUD MCP Client Architecture

This directory contains the MCP client implementations for HUD SDK. The architecture is designed to be flexible and extensible, allowing different client implementations while maintaining a consistent interface for agents.

## Architecture Overview

```
hud/clients/
├── base.py          # Protocol definition and base class
├── mcp_use.py       # MCP-use based implementation (default)
├── fastmcp.py       # FastMCP based implementation (alternative)
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
- Alternative implementation with different transport handling
- Supports various transports (HTTP, WebSocket, stdio, in-memory)
- Better type safety and structured data support

## Usage Examples

### Basic Usage

```python
from hud.clients import MCPClient, FastMCPHUDClient

# Configuration works for both clients
mcp_config = {
    "server_name": {
        "command": "python",
        "args": ["server.py"]
    }
}

# Default client (MCPUseHUDClient)
client = MCPClient(mcp_config)

# Alternative: FastMCP client
client = FastMCPHUDClient(mcp_config)

# Both use the same API
async with client:
    tools = await client.list_tools()
    result = await client.call_tool("tool_name", {"arg": "value"})
```

### With Agents

```python
from hud.agents import ClaudeAgent
from hud.clients import MCPClient

# Either client works with agents
client = MCPClient(mcp_config)

agent = ClaudeAgent(
    mcp_client=client,
    model="claude-sonnet-4-5"
)

# Agent works identically with either client
result = await agent.run("Your task here")
```


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

## Tool Output Validation

Both client implementations support tool output validation through the MCP protocol:

```python
# Enable strict validation
client = MCPClient(mcp_config, strict_validation=True)

# With strict validation:
# - Tools must return structured content if they define an output schema
# - Content must match the schema or an error is raised
# - Helps catch tool implementation issues early

# Default behavior (lenient validation):
client = MCPClient(mcp_config, strict_validation=False)  # Default

# With lenient validation:
# - Schema violations are logged as warnings
# - Execution continues even if output doesn't match schema
# - Better for development and debugging
```

### Example with Validation

```python
from hud.clients import MCPClient

# Create client with strict validation
client = MCPClient(mcp_config, strict_validation=True)

try:
    # If tool has output schema but returns invalid data,
    # this will raise a RuntimeError
    result = await client.call_tool("some_tool", {"arg": "value"})
except RuntimeError as e:
    print(f"Validation error: {e}")
```
