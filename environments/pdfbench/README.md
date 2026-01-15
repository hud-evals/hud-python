# HUD PDFBench Environment

PDF Form Filling MCP Environment for evaluating agent ability to fill PDF forms.

## Overview

This environment provides MCP tools for:
- Loading blank PDF forms
- Listing available form fields
- Filling form fields by name or bounding box
- Saving filled PDFs
- Evaluating filled PDFs against expected values

## Quick Start

### 1. Build the Docker image

```bash
docker build -t hud-pdfbench:latest .
```

### 2. Convert harbor-pdfbench tasks to HUD format

```bash
python convert_harbor_tasks.py /path/to/harbor-pdfbench/harbor_tasks -o tasks.json --limit 10
```

### 3. Run evaluation with HUD

```bash
# Single task
hud eval tasks.json --agent claude --max-steps 20

# Full dataset
hud eval tasks.json --full --agent gemini --max-concurrent 10
```

## Available MCP Tools

### Setup Tools

#### `setup.load_pdf`
Load a blank PDF form for filling.

```json
{
  "name": "setup",
  "arguments": {
    "name": "load_pdf",
    "arguments": {
      "pdf_path": "/app/pdfs/form.pdf",
      "output_path": "/tmp/filled.pdf",
      "solution_path": "/opt/tbench/solution.json"
    }
  }
}
```

### Action Tools

#### `list_fields`
List all form fields in the loaded PDF.

```json
{"name": "list_fields", "arguments": {"page": 0}}
```

#### `fill_field`
Fill a form field by name or bounding box.

```json
{"name": "fill_field", "arguments": {"field_name": "Name", "value": "John Doe"}}
```

Or by bbox:
```json
{"name": "fill_field", "arguments": {"bbox": "0,45,105,417,117", "value": "COMPANY NAME"}}
```

#### `get_field`
Get the current value of a field.

```json
{"name": "get_field", "arguments": {"field_name": "Name"}}
```

#### `save_pdf`
Save the filled PDF.

```json
{"name": "save_pdf", "arguments": {"output_path": "/tmp/filled.pdf"}}
```

### Evaluate Tools

#### `evaluate.verify_fields`
Verify filled PDF against expected values.

```json
{
  "name": "evaluate",
  "arguments": {
    "name": "verify_fields",
    "arguments": {
      "solution_path": "/opt/tbench/solution.json",
      "fuzzy_match": true,
      "partial_credit": true,
      "strict_empty": true
    }
  }
}
```

## Task Format

HUD tasks for PDFBench look like:

```json
{
  "id": "pdfbench_eyemed_x001",
  "prompt": "Fill out the EyeMed vision enrollment form...",
  "mcp_config": {
    "local": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "-v", "...", "hud-pdfbench:latest"]
    }
  },
  "setup_tool": {
    "name": "setup",
    "arguments": {
      "name": "load_pdf",
      "arguments": {
        "pdf_path": "/app/pdfs/eyemed.pdf",
        "output_path": "/tmp/filled.pdf",
        "solution_path": "/opt/tbench/solution.json"
      }
    }
  },
  "evaluate_tool": {
    "name": "evaluate",
    "arguments": {
      "name": "verify_fields",
      "arguments": {
        "solution_path": "/opt/tbench/solution.json"
      }
    }
  }
}
```

## Verification

The verifier uses bounding box matching to compare filled PDF fields against expected values:

- **Bbox Key Format**: `"page,x0,y0,x1,y1"` (e.g., `"0,45,105,417,117"`)
- **Fuzzy Matching**: Case-insensitive substring matching for text fields
- **Checkbox Handling**: Normalizes "Yes/True/1/On" to checked state
- **Partial Credit**: Returns proportional score (0.0-1.0)
- **Strict Empty**: Optionally verifies unlisted fields are unchanged

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PDF_PATH` | Auto-load PDF on startup |
| `OUTPUT_PATH` | Default output path for filled PDF |
| `SOLUTION_PATH` | Path to solution.json for evaluation |
