FROM python:3.11-slim

WORKDIR /app

# Install git for dependency installation
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# TODO: ideally, we have docker download dataset and, if required, local model weights
# that way we don't have to redo this if something gets changed downstream of this.
# Example: RUN entrypoint.sh

# Copy and install dependencies
COPY pyproject.toml pyproject.toml
RUN pip install uv
# Create a virtual environment
RUN uv venv /opt/venv

# Set the PATH to include the venv's bin directory
ENV PATH="/opt/venv/bin:$PATH"

# Create inspect_evals directory (eval will be downloaded at runtime)
RUN mkdir -p inspect_evals

COPY controller/ ./controller/
COPY environment/ ./environment/
COPY download-eval.sh ./download-eval.sh
RUN chmod +x download-eval.sh

RUN uv pip install -e .
RUN uv pip list
RUN ls -a

ENV ENV_SERVER_PORT=8005
ENV COLUMNS=120

# Start context server in background, then run controller with hot-reload
# Disable access logs to prevent stdout corruption
CMD ["sh", "-c", "./download-eval.sh && uvicorn environment.server:app --host 0.0.0.0 --port $ENV_SERVER_PORT --reload --log-level warning --reload-dir environment & sleep 0.5 && exec hud run controller --reload"]
