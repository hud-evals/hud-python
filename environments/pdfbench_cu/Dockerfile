# PDFBench Computer Use - Headed Chrome with Xvfb for PDF viewing
FROM hudevals/hud-browser-base:latest

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

ENV HUD_LOG_STREAM=stderr
ENV DISPLAY=":1"

# Install dependencies using uv (available in base image)
RUN uv pip install --system --break-system-packages -e .

# Install Firefox for Playwright
RUN playwright install firefox
RUN playwright install-deps firefox

# Create directories
RUN mkdir -p /app/pdfs /opt/tbench /tmp

CMD ["python3", "-m", "hud_pdfbench_cu.server"]
