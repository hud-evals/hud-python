# PDFBench MCP server
FROM python:3.11-slim

# For live reload
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install git for dependency installation
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

ENV HUD_LOG_STREAM=stderr

# Install dependencies in editable mode
RUN pip install --no-cache-dir -e .

# Create directories for PDFs and solutions
RUN mkdir -p /app/pdfs /opt/tbench /tmp

CMD ["python", "-m", "hud_pdfbench.server"]
