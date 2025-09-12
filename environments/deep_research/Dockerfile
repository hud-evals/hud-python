# Use the HUD base browser image with Playwright and uv pre-installed
FROM hudpython/base-browser:latest

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

# Install the package using the existing venv at /opt/venv
RUN uv pip install --python /opt/venv -e .

ENV DISPLAY_WIDTH=1448
ENV DISPLAY_HEIGHT=944

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Initial URL defaults to Wikipedia if not provided
ENV INITIAL_URL="https://en.wikipedia.org/wiki/Main_Page"

CMD ["python", "-m", "hud_controller.server"]

