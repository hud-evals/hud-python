# Jupyter Env (for SpreadSheetBench)

## QuickStart

### MCP Server from Dockerhub (Don't Have to Build Docker Image)

Run task by
```
hud eval Genteki/SpreadSheetBench
```

### Local MCP Server

First we build the docker image with
```
docker build -t <image/name> .
```
Then modify the docker image name in `test_task.json`. Finally, load all `api_key` needed into environment varible and run

```
hud eval
```

## File Structure

`environments/jupyter` file sturcture:
```
├── Dockerfile
├── server
│   ├── config.py
│   ├── evaluate
│   │   ├── compare.py
│   │   ├── dumb.py
│   │   ├── eval_all.py
│   │   ├── eval_single.py
│   │   ├── generalize.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── main.py
│   ├── pyproject.toml
│   ├── setup
│   │   ├── __init__.py
│   │   └── load_spreadsheet.py
│   └── tools
│       ├── __init__.py
│       └── jupyter_with_record.py
└── test_task.json
```
Here we introduce the main parts of the environments
* `main.py` start point of MCP server
* `tools/jupyter_with_record.py`: offer `execute_code` method to allow agent interacting with jupyter kernel and record the solution
* `setup/`: setup methods for eval task
* `evaluate/` evaluations method for eval task


## Related Linkd
### Hugginface:
* [Genteki/SpreadSheetBench-Tiny](https://huggingface.co/datasets/Genteki/SpreadSheetBench-Tiny) (Size: 10)
* [Genteki/SpreadSheetBench-200](https://huggingface.co/datasets/Genteki/SpreadSheetBench-200) (Size: 200)
* [Genteki/SpreadSheetBench](https://huggingface.co/datasets/Genteki/SpreadSheetBench) (Size: 912)

### Example Traces (May require permission)
* [Single Test Task](https://www.hud.ai/trace/d31de170-e70a-4abb-8f95-70512515dade)
* [Genteki/SpreadSheetBench-Tiny Test](https://www.hud.ai/jobs/2c426368-e352-4c79-af4a-aefb136e3f58)

### Github

* Feature Branch: [New-Env-Jupyter](https://github.com/Genteki/hud-python/tree/New-Env-Jupyter)