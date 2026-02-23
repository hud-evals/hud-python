"""Get command for downloading HuggingFace datasets."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import typer
from datasets import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def get_command(
    dataset_name: str = typer.Argument(
        ..., help="HuggingFace dataset name (e.g., 'hud-evals/browser-2048-tasks')"
    ),
    split: str = typer.Option(
        "train", "--split", "-s", help="Dataset split to download (train/test/validation)"
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Output filename (defaults to dataset_name.jsonl)"
    ),
    format: str | None = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json (list) or jsonl (one task per line)",
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Limit number of examples to download"
    ),
) -> None:
    """📥 Download a HuggingFace dataset and save it as JSONL."""
    console.print(f"\n[cyan]📥 Downloading dataset: {dataset_name}[/cyan]")

    # Import datasets library
    try:
        from datasets import load_dataset
    except ImportError as e:
        console.print("[red]Error: datasets library not installed[/red]")
        console.print("[yellow]Install with: pip install datasets[/yellow]")
        raise typer.Exit(1) from e

    # Determine output filename
    if output is None:
        # Convert dataset name to filename (e.g., "hud-evals/browser-2048" -> "browser-2048.json|jsonl") # noqa: E501
        if format is None:
            format = "json"
        ext = ".json" if format.lower() == "json" else ".jsonl"
        dataset_filename = dataset_name.split("/")[-1] + ext
        output = Path(dataset_filename)

    # Download dataset with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Loading {dataset_name}...", total=None)

        try:
            dataset = load_dataset(dataset_name, split=split)
            progress.update(task, completed=100)
        except ValueError as e:
            if "Unknown split" in str(e):
                console.print(f"[red]Error: Split '{split}' not found in dataset[/red]")
                console.print("[yellow]Common splits: train, test, validation[/yellow]")
            else:
                console.print(f"[red]Error loading dataset: {e}[/red]")
            raise typer.Exit(1) from e
        except FileNotFoundError as e:
            console.print(f"[red]Error: Dataset '{dataset_name}' not found[/red]")
            console.print("[yellow]Check the dataset name on HuggingFace Hub[/yellow]")
            raise typer.Exit(1) from e
        except Exception as e:
            if "authentication" in str(e).lower() or "401" in str(e):
                console.print("[red]Error: Dataset requires authentication[/red]")
                console.print("[yellow]Login with: huggingface-cli login[/yellow]")
            else:
                console.print(f"[red]Error loading dataset: {e}[/red]")
            raise typer.Exit(1) from e

    if not isinstance(dataset, Dataset):
        raise typer.Exit(1)

    # Apply limit if specified
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        console.print(f"[yellow]Limited to {len(dataset)} examples[/yellow]")

    # Save as JSON or JSONL
    console.print(f"[cyan]Writing to {output}...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        transient=True,
    ) as progress:
        task = progress.add_task("Saving...", total=len(dataset))

        if format is None:
            format = "json"

        if format.lower() == "json":
            # Write a single JSON array
            data_list = []
            for _, example in enumerate(dataset):
                item = example.to_dict() if hasattr(example, "to_dict") else example  # type: ignore
                for key, value in item.items():  # type: ignore
                    with contextlib.suppress(json.JSONDecodeError):
                        item[key] = json.loads(value)  # type: ignore
                data_list.append(item)
                progress.update(task, advance=1)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
        else:
            # Write JSONL
            with open(output, "w", encoding="utf-8") as f:
                for _, example in enumerate(dataset):
                    # Convert to dict if needed
                    if hasattr(example, "to_dict"):
                        example = example.to_dict()  # type: ignore
                    for key, value in example.items():  # type: ignore
                        with contextlib.suppress(json.JSONDecodeError):
                            example[key] = json.loads(value)  # type: ignore
                    # Write as JSON line
                    f.write(json.dumps(example) + "\n")
                    progress.update(task, advance=1)

    # Show summary
    console.print(f"\n[green]✅ Downloaded {len(dataset)} examples to {output}[/green]")

    # Show sample of fields
    if len(dataset) > 0:
        first_example = dataset[0]
        if hasattr(first_example, "to_dict"):
            first_example = first_example.to_dict()  # type: ignore

        console.print("\n[yellow]Dataset fields:[/yellow]")
        for field in first_example:
            console.print(f"  • {field}")

        # Show example if small enough
        if len(json.dumps(first_example)) < 500:
            console.print("\n[yellow]First example:[/yellow]")
            console.print(json.dumps(first_example, indent=2))

    # Show next steps
    console.print("\n[dim]Next steps:[/dim]")
    console.print(f"[dim]• Use for training: hud rft run {output}[/dim]")
    console.print(f"[dim]• Use for evaluation: hud eval {output}[/dim]")


# Export the command
__all__ = ["get_command"]
