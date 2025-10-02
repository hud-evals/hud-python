# Apps Directory

Launchable web applications for the HUD browser environment. Each app is a self-contained service that can be dynamically launched.

## App Specification

Each app must implement:

### Required Files
- `launch.py` - Entry point script with standardized arguments
- `backend/` - Backend service (required)
- `frontend/` - Frontend service (optional)

### Launch Script Interface

```python
# launch.py
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-port", type=int)
    parser.add_argument("--backend-port", type=int, required=True)
    args = parser.parse_args()
    
    # Start your services on the provided ports
    # Backend must run on args.backend_port
    # Frontend (if present) should run on args.frontend_port

if __name__ == "__main__":
    main()
```

### Service Requirements

**Backend**
- Must bind to the provided `--backend-port`
- Should implement health check endpoint (`/health`)
- Must handle graceful shutdown
- Should use production-ready server (uvicorn, gunicorn, etc.)

**Frontend** (Optional)
- Must bind to the provided `--frontend-port`
- Should be a static build or development server
- Common frameworks: Next.js, React, Vue, etc.

## App Lifecycle

1. **Discovery** - Apps are discovered by scanning subdirectories
2. **Launch** - Controller calls `python launch.py --backend-port=5000 --frontend-port=3000`
3. **Registration** - Ports are registered for API access
4. **Operation** - App services run independently
5. **Cleanup** - Processes terminated when environment shuts down

## Integration Patterns

### Basic Web App
```python
# Minimal FastAPI backend
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import sys
    port = int(sys.argv[sys.argv.index("--backend-port") + 1])
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Full-Stack App
```python
# launch.py for app with both frontend and backend
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-port", type=int)
    parser.add_argument("--backend-port", type=int, required=True)
    args = parser.parse_args()
    
    # Start backend
    backend_proc = subprocess.Popen([
        "uvicorn", "backend.main:app",
        "--host", "0.0.0.0",
        "--port", str(args.backend_port)
    ])
    
    # Start frontend (if port provided)
    if args.frontend_port:
        frontend_proc = subprocess.Popen([
            "npm", "run", "dev", "--", "--port", str(args.frontend_port)
        ], cwd="frontend")
    
    # Wait for processes
    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        backend_proc.terminate()
        if args.frontend_port:
            frontend_proc.terminate()
```

## Optional Integrations

### Evaluation APIs
Apps can optionally provide evaluation endpoints for testing:
- `GET /api/eval/health` - Health check
- `GET /api/eval/stats` - Application statistics
- Additional endpoints as needed

### Environment Access
Apps can access the browser environment through:
- Shared network (communicate with controller)
- File system (shared volumes)
- Environment variables

## Development Guidelines

- **Port Binding** - Always use provided ports, never hardcode
- **Health Checks** - Implement basic health endpoints
- **Logging** - Use structured logging for debugging
- **Dependencies** - Manage dependencies with lockfiles
- **Graceful Shutdown** - Handle SIGTERM properly
- **Error Handling** - Return meaningful error responses

## Examples

- `todo/` - Full-stack Next.js + FastAPI application with evaluation integration
- See individual app READMEs for specific implementation details 