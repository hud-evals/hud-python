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

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# Note: Environment variables for browser providers should be set at runtime:
# - BROWSER_PROVIDER: anchorbrowser, steel, browserbase, hyperbrowser, kernel
# - Provider-specific API keys: ANCHOR_API_KEY, STEEL_API_KEY, etc.
# - GCP_CREDENTIALS_JSON: For Google Sheets functionality (if needed)

# Run as a module to ensure proper package imports
CMD ["python", "-m", "hud_controller.server"]