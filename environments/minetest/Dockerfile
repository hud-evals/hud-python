FROM hudpython/novnc-base:latest

# Workdir
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install system dependencies for GUI automation and Minetest
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        minetest \
        xdotool \
        scrot \
        libgl1 \
        libgl1-mesa-dri \
        libglu1-mesa \
        libxxf86vm1 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 && \
    rm -rf /var/lib/apt/lists/*

# Install python package (editable)
RUN uv pip install --system --break-system-packages -e .

# Environment defaults
ENV MCP_TRANSPORT="stdio"
ENV HUD_LOG_STREAM="stderr"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONWARNINGS="ignore::SyntaxWarning:pyautogui"

# Expose noVNC port
EXPOSE 8080

# Start X11 and context server in the background, then VNC/noVNC, then run MCP server
CMD ["sh", "-c", "\
    # Start X11 if not already running \
    if [ ! -e /tmp/.X11-unix/X1 ]; then \
        Xvfb :1 -screen 0 1920x1080x24 >/dev/null 2>&1 & \
        while [ ! -e /tmp/.X11-unix/X1 ]; do sleep 0.1; done; \
    fi && \
    export DISPLAY=:1 && \
    # Start context server \
    python3 -m hud_controller.context >&2 & \
    # Start VNC and websockify \
    x11vnc -display :1 -forever -shared -nopw >/dev/null 2>&1 & \
    websockify --web /usr/share/novnc 8080 localhost:5900 >/dev/null 2>&1 & \
    # Give services a moment \
    sleep 3 && \
    # Run MCP server in foreground \
    exec python3 -m hud_controller.server \
"]

