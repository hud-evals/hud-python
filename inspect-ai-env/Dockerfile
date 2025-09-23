FROM python:3.11-slim

WORKDIR /app

# Install git for dependency installation
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# TODO: ideally, we have docker download dataset and, if required, local model weights
# that way we don't have to redo this if something gets changed downstream of this.
# Example: RUN entrypoint.sh

# Copy and install dependencies
COPY pyproject.toml ./
COPY controller/ ./controller/
COPY environment/ ./environment/
RUN pip install -U pip
RUN pip install uv
RUN uv sync
RUN uv pip install --no-cache-dir -e .

ENV ENV_SERVER_PORT=8005

# Start context server in background, then run controller with hot-reload
# Disable access logs to prevent stdout corruption
CMD ["sh", "-c", "uvicorn environment.server:app --host 0.0.0.0 --port $ENV_SERVER_PORT --reload --log-level warning --reload-dir environment & sleep 0.5 && exec hud run controller --reload"]
