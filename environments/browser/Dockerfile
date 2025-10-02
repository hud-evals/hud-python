# syntax=docker/dockerfile:1
FROM ubuntu:24.04 AS setup

# Update and install core dependencies (including working Chromium browser)
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
  vim \
  openssl \
  ca-certificates \
  curl \
  wget \
  sudo \
  bash \
  net-tools \
  novnc \
  x11vnc \
  xvfb \
  xfce4 \
  locales \
  libpq5 \
  sqlite3 \
  dbus-x11 \
  xfce4-terminal \
  xfonts-base \
  xdotool \
  psmisc \
  scrot \
  pm-utils \
  build-essential \
  unzip \
  xauth \
  gnupg \
  gpg \
  jq \
  git \
  build-essential \
  nodejs \
  npm

RUN update-ca-certificates

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install git for dependency installation
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Playwright
RUN uv pip install --system --break-system-packages playwright
RUN python3 -m playwright install chromium --with-deps

# Layer 1: Install server dependencies
COPY server/pyproject.toml /app/server/
RUN cd /app/server && uv pip install --system --break-system-packages .

# Layer 2: Install environment dependencies
COPY environment/pyproject.toml /app/environment/
RUN cd /app/environment && uv pip install --system --break-system-packages .

# Layer 3: Copy source code (changes here don't invalidate dependency layers)
COPY server/ /app/server/
COPY environment/ /app/environment/

# Auto-discover and install/build all frontend apps
RUN set -e; \
    for pkg in $(find /app/environment -type f -path '*/frontend/package.json'); do \
        app_dir=$(dirname "$pkg"); \
        echo "Installing dependencies in $app_dir"; \
        if [ -f "$app_dir/package-lock.json" ]; then \
            (cd "$app_dir" && npm ci --no-audit --no-fund); \
        else \
            (cd "$app_dir" && npm install --no-audit --no-fund); \
        fi; \
    done && \
    for pkg in $(find /app/environment -type f -path '*/frontend/package.json'); do \
        app_dir=$(dirname "$pkg"); \
        if [ -f "$app_dir/next.config.js" ]; then \
            echo "Building Next.js app in $app_dir"; \
            (cd "$app_dir" && npm run build); \
        fi; \
    done

# Make scripts executable
RUN find /app/environment -name "*.py" -type f -exec chmod +x {} \;

# Environment configuration
ENV MCP_TRANSPORT="stdio"
ENV HUD_LOG_STREAM="stderr"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONWARNINGS="ignore::SyntaxWarning:pyautogui"
ENV DISPLAY=":1"
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8080 3000-3200 5000-5200

# Simple startup: HUD_DEV=1 enables hot-reload; otherwise run production
CMD ["sh", "-c", "\
    if [ \"${HUD_DEV:-0}\" = \"1\" ]; then \
      uvicorn environment.server:app --host 0.0.0.0 --port 8000 --reload --log-level warning >&2 & \
      sleep 5 && cd /app/server && exec hud dev server.main --stdio; \
    else \
      uvicorn environment.server:app --host 0.0.0.0 --port 8000 --log-level warning >&2 & \
      sleep 5 && cd /app/server && exec python3 -m server.main; \
    fi\
"]