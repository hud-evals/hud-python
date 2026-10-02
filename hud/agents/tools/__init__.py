"""Provider-facing agent tools.

``AgentTool`` is the abstract base, generic in its client type. Capability
bases — ``SSHTool``, ``MCPTool`` (later ``RFBTool``) — bind that generic and
add per-protocol helpers. Provider subclasses extend one of those bases.

``HostedTool`` is a separate kind: provider-built-in tools (Claude WebSearch,
Gemini CodeExecution, …) that aren't backed by any capability/client and are
declared by agent config.
"""
