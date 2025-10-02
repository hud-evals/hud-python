# Environment

Backend service: owns state and exposes HTTP APIs the controller calls.

Endpoints (FastAPI)
- `GET /health` → {status: ok}
- `POST /act` → increments counter and returns {count}
- `POST /reset` → resets counter
- `GET /state` → returns {count}

Run (dev)
```bash
uv run uvicorn server:app --reload --port 8005
```

Principle: treat like a backend. Keep long‑lived state here; add endpoints as tools need them.
