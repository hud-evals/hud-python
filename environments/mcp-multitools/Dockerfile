FROM python:3.11-slim

WORKDIR /app

# Install git for dependency installation
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install MCP server dependencies
COPY server/pyproject.toml ./server/
RUN pip install --no-cache-dir ./server

# Copy and install environment dependencies
COPY environment/pyproject.toml ./environment/
RUN pip install --no-cache-dir ./environment

# Copy source code after dependencies
COPY server/ ./server/
COPY environment/ ./environment/

# Copy documents folder for document store
COPY documents/ ./documents/

# Environment variables (can be overridden at runtime)
ENV ENV_SERVER_PORT=8000
ENV PYTHONPATH=/app

# API keys are passed at runtime via -e flags:
# docker run -e EXA_API_KEY=... -e SUPABASE_URL=... -e SUPABASE_API_KEY=... -e LINEAR_API_KEY=...

# Start environment server in background, then run MCP server with stdio
CMD ["sh", "-c", "uvicorn environment.server:app --host 0.0.0.0 --port $ENV_SERVER_PORT --log-level warning --reload >&2 & sleep 0.5 && cd /app/server && exec hud dev server.main --stdio"]

