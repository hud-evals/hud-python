# A2A Chat

Serve a HUD chat task over the [A2A protocol](https://github.com/google/a2a),
and talk to it from Python clients.

`hud.Chat` is protocol-agnostic — these scripts are the protocol layer, kept
outside the SDK on purpose. Copy and adapt them.

| File | What it does |
|------|--------------|
| `server.py` | A2A server: one `Chat` (conversation) per A2A context, agent card, citations artifact |
| `client.py` | Minimal A2A client: send messages, print replies |
| `llm_client.py` | LLM-fronted client: an OpenAI model decides when to call the A2A agent as a tool |
| `chat_env.py` | Sample chat environment with `messages`-style tasks to serve |

## Run

From this directory (uv resolves the dependencies on first run):

```bash
# Terminal 1: serve the bundled chat task (spawns chat_env.py per turn)
uv run server.py

# Terminal 2: talk to it
uv run client.py            # plain client
uv run llm_client.py        # LLM-fronted client
```

Configuration is via env vars: `HUD_MODEL` picks the agent's model (gateway,
needs `HUD_API_KEY`), `HUD_TASK`/`HUD_ENV` pick the task row, `HUD_SOURCE`
spawns a different env source, and `HUD_ENV_URL` attaches each turn to an
already-served control channel (e.g. `hud serve chat_env.py` →
`HUD_ENV_URL=tcp://127.0.0.1:8765`) instead of spawning.

The server publishes an agent card at `/.well-known/agent-card.json` and
accepts A2A messages at the root endpoint. The configured task should accept a
`messages` argument for multi-turn history (see `chat_env.py`).
