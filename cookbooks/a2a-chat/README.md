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
# Terminal 1: serve a chat task from a deployed environment
HUD_ENV=my-hud-environment HUD_TASK=analysis_chat \
    uv run server.py

# Terminal 2: talk to it
uv run client.py            # plain client
uv run llm_client.py        # LLM-fronted client
```

The server publishes an agent card at `/.well-known/agent-card.json` and
accepts A2A messages at the root endpoint. The configured task should accept a
`messages` argument for multi-turn history (see `chat_env.py`).
