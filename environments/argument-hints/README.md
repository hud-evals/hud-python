# Argument Hints Environment

A minimal environment showing what `x-hud-hint` annotations do: `prompt`
renders as a text box, `attachments` as a data-file picker, and `criteria`
as a table of grading rows — so a task is fully creatable from the console
without reading this code.

```bash
uv sync
hud eval tasks.py claude --runtime local -y
```

`env.py` defines one `review_files` template: it stages the referenced data
files into `/workspace/files`, hands the agent a shell and the prompt, and
passes the criteria straight to `LLMJudgeGrader`. It also declares a
`hud_api_key` argument (hidden in the console) — that declaration is what
makes hosted rollouts inject the runner's key for staging and grading.
