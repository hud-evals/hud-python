# Use our HUD base browser image with Playwright and uv pre-installed
FROM hudpython/base-browser:latest

# Create app-specific working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package using the existing venv at /opt/venv
# The --python flag tells uv to use this specific Python instead of creating a new venv
RUN uv pip install --python /opt/venv -e .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data

ENV DISPLAY_WIDTH=1448
ENV DISPLAY_HEIGHT=944

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# Note: Environment variables for browser providers should be set at runtime:
# - BROWSER_PROVIDER: anchorbrowser, steel, browserbase, hyperbrowser, kernel
# - Provider-specific API keys: ANCHOR_API_KEY, STEEL_API_KEY, etc.
# - GCP_CREDENTIALS_JSON: For Google Sheets functionality (if needed)

# Run remote browser with persistent context
CMD ["sh", "-c", "\
    # Start context server in background \
    python3 -m hud_controller.context >&2 & \
    # Wait a bit for context server to start \
    sleep 2 && \
    # Run MCP server in foreground with exec \
    exec python3 -m hud_controller.server \
"]