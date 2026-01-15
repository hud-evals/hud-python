#!/usr/bin/env python3
"""Convert harbor-pdfbench tasks to HUD format."""

import json
import os
import sys
from pathlib import Path


TOOL_INSTRUCTIONS = """

Please use the available tools to:
1. First call list_fields to see all available form fields
2. Use fill_field to fill each field with the appropriate value
3. Call save_pdf when done
4. The evaluation will automatically verify your work"""


def convert_task(harbor_task_dir: Path, docker_image: str = "hud-pdfbench:latest") -> dict:
    """Convert a single harbor task to HUD format.

    Args:
        harbor_task_dir: Path to harbor task directory (e.g., harbor_tasks/pdfbench_eyemed_x001)
        docker_image: Docker image name for the MCP server

    Returns:
        HUD task dict
    """
    task_name = harbor_task_dir.name

    # Read instruction.md
    instruction_path = harbor_task_dir / "instruction.md"
    if instruction_path.exists():
        raw_prompt = instruction_path.read_text().strip()
        # Append tool usage instructions
        prompt = raw_prompt + TOOL_INSTRUCTIONS
    else:
        prompt = f"Fill out the PDF form for task {task_name}" + TOOL_INSTRUCTIONS

    # Read solution.json
    solution_path = harbor_task_dir / "environment" / "solution.json"
    if not solution_path.exists():
        solution_path = harbor_task_dir / "solution" / "solution.json"

    if solution_path.exists():
        with open(solution_path) as f:
            solution = json.load(f)
        expected_values = solution.get("fields", solution)
        pdf_source = solution.get("pdf_source", "form.pdf")
    else:
        expected_values = {}
        pdf_source = "form.pdf"

    # Find the PDF file
    pdfs_dir = harbor_task_dir / "environment" / "pdfs"
    if pdfs_dir.exists():
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        if pdf_files:
            pdf_source = pdf_files[0].name

    # Extract output path from instruction (usually mentioned in the instruction)
    # Default pattern: /tmp/{form_type}_{task_name}_filled.pdf
    output_path = f"/tmp/{task_name}_filled.pdf"

    # Build HUD task
    hud_task = {
        "id": task_name,
        "prompt": prompt,
        "mcp_config": {
            "local": {
                "command": "docker",
                "args": [
                    "run",
                    "--rm",
                    "-i",
                    "-v", f"{harbor_task_dir}/environment/pdfs:/app/pdfs:ro",
                    "-v", f"{harbor_task_dir}/environment/solution.json:/opt/tbench/solution.json:ro",
                    "-e", f"PDF_PATH=/app/pdfs/{pdf_source}",
                    "-e", f"OUTPUT_PATH={output_path}",
                    "-e", "SOLUTION_PATH=/opt/tbench/solution.json",
                    docker_image,
                ],
            }
        },
        "setup_tool": {
            "name": "setup",
            "arguments": {
                "name": "load_pdf",
                "arguments": {
                    "pdf_path": f"/app/pdfs/{pdf_source}",
                    "output_path": output_path,
                    "solution_path": "/opt/tbench/solution.json",
                },
            },
        },
        "evaluate_tool": {
            "name": "evaluate",
            "arguments": {
                "name": "verify_fields",
                "arguments": {
                    "solution_path": "/opt/tbench/solution.json",
                    "fuzzy_match": True,
                    "partial_credit": True,
                    "strict_empty": True,
                },
            },
        },
        "metadata": {
            "source": "harbor-pdfbench",
            "task_name": task_name,
            "pdf_source": pdf_source,
            "field_count": len(expected_values),
            "category": "pdf-form-filling",
        },
    }

    return hud_task


def convert_all_tasks(
    harbor_tasks_dir: Path,
    output_file: Path,
    docker_image: str = "hud-pdfbench:latest",
    limit: int | None = None,
) -> list[dict]:
    """Convert all harbor tasks to HUD format.

    Args:
        harbor_tasks_dir: Path to harbor_tasks directory
        output_file: Path to write output JSON
        docker_image: Docker image name
        limit: Maximum number of tasks to convert

    Returns:
        List of HUD task dicts
    """
    tasks = []

    # Find all task directories
    task_dirs = sorted(harbor_tasks_dir.glob("pdfbench_*"))

    if limit:
        task_dirs = task_dirs[:limit]

    for task_dir in task_dirs:
        if task_dir.is_dir():
            try:
                hud_task = convert_task(task_dir, docker_image)
                tasks.append(hud_task)
                print(f"Converted: {task_dir.name}")
            except Exception as e:
                print(f"Error converting {task_dir.name}: {e}", file=sys.stderr)

    # Write output
    with open(output_file, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"\nConverted {len(tasks)} tasks to {output_file}")
    return tasks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert harbor-pdfbench tasks to HUD format")
    parser.add_argument(
        "harbor_tasks_dir",
        type=Path,
        help="Path to harbor_tasks directory",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("tasks.json"),
        help="Output file path (default: tasks.json)",
    )
    parser.add_argument(
        "--docker-image",
        default="hud-pdfbench:latest",
        help="Docker image name (default: hud-pdfbench:latest)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of tasks to convert",
    )

    args = parser.parse_args()

    convert_all_tasks(
        args.harbor_tasks_dir,
        args.output,
        args.docker_image,
        args.limit,
    )
