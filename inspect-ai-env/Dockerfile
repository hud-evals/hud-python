FROM python:3.11-slim

WORKDIR /app

# Install git and other system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY docker_pyproject.toml pyproject.toml
RUN pip install uv

# Create a virtual environment
RUN uv venv /opt/venv

# Set the PATH and VIRTUAL_ENV BEFORE running uv commands
# This ensures uv installs packages into the correct venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies into the activated venv
RUN uv sync
RUN uv pip install -e .

# Copy application files
COPY controller/ ./controller/
COPY environment/ ./environment/
COPY inspect_loader.py ./inspect_loader.py
COPY task_converter.py ./task_converter.py

# Create directories for eval storage and downloaded evals
RUN mkdir -p inspect_evals custom_evals logs

# Copy eval download script if it exists
COPY download-eval.sh ./download-eval.sh 
RUN chmod +x download-eval.sh

# Verification: ensure hud command is available
RUN ls -l /opt/venv/bin && which hud

# Start sandbox server in background, then run MCP controller
# The sandbox server provides file/exec operations
# The controller exposes these as MCP tools to the agent
CMD ["sh", "-c", "uvicorn environment.server:app --host 0.0.0.0 --port 8000 --log-level warning & sleep 0.5 && exec hud run controller"]
