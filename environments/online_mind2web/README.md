# HUD Online Mind2Web Taskset

Based on hud remote-browser, this MCP server provides environment for Online-Mind2Web task exacution and evaluation.

## Running with Docker

The Docker image supports both production and development modes using the same Dockerfile.

### Building the Image

```bash
# Production build (default)
docker build -t hud-om2w:latest .
```

### Running the Test Task
```bash
hud eval ./test_task.json 
```

### Running Whole Online-Mind2Web Dataset From HuggingFace
```bash
hud eval Genteki/Online-Mind2Web --full --max-concurrent=5
```
