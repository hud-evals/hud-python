FROM public.ecr.aws/docker/library/python:3.11-bookworm

WORKDIR /app

# Install git for dependency installation
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY pyproject.toml ./
COPY controller/ ./controller/
COPY environment/ ./environment/
RUN pip install --no-cache-dir -e .
