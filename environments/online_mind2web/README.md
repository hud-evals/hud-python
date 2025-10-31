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

### Different Evaluation Method

To chosse different evaluation method, you can change different `task["evaluate_tool"]["evaluate"]["name"]` value in task json file. Here are the different evaluation method we support for you:

| Evaluation Method | Final Screenshot | Screenshot History | Action Histroy | 
|:---|:---:|:---:| :---: |
| `autonomous` | ✔ | ✗ | ✔ |
| `webjudge` | ✔ | ✔ | ✔ |
| `overall_judge`[^1] | - | - | - |

[^1]: `overall_judge` will execute all evaluation methods above and return the average of the rewards of them.
