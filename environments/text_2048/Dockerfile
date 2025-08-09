FROM python:3.11-slim

# For live reload
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install git for fetching hud-python from GitHub
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies in editable mode
RUN pip install --no-cache-dir -e .

# Run as a module to ensure proper package imports
CMD ["python", "-m", "hud_controller.server"]