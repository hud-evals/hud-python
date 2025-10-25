# Browser Base Image

Base Docker image for browser environments with Playwright, Chromium, and VNC support.

## Build

```bash
docker build -t browser-base:latest .
```

## Test with VNC Access

### 1. Start the container

```bash
docker run -it --rm \
  -p 6080:6080 \
  -p 5900:5900 \
  -e DISPLAY=:1 \
  browser-base:latest \
  bash
```

### 2. Inside the container, start display servers

```bash
Xvfb :1 -screen 0 1920x1080x24 > /dev/null 2>&1 &
x11vnc -display :1 -nopw -listen 0.0.0.0 -forever > /dev/null 2>&1 &
/usr/share/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 6080 > /dev/null 2>&1 &
```

### 3. Test Playwright

```bash
python3 -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto('https://example.com')
    print('Title:', page.title())
    input('Press Enter to close...')
    browser.close()
"
```

### 4. View in browser

Open `http://localhost:6080/vnc.html` to see Chromium running.

## What's Included

- Ubuntu 24.04
- Desktop environment (Xvfb, x11vnc, noVNC, xfce4)
- Node.js & npm
- Python 3 with uv package manager
- Playwright with Chromium
- Development tools (git, curl, wget, etc.)