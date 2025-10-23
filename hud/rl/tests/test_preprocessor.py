#!/usr/bin/env python
import json
from pathlib import Path

import torch
from transformers import Qwen2VLProcessor

from rich.console import Console
from rich.text import Text
from hud.types import Trace
from hud.rl.preprocessor import preprocess_traces

# Disable automatic highlighting
console = Console(highlight=False)


def visualize_tokenization(input_ids, assistant_mask, tokenizer):
    """Visualize tokenization with both token IDs and decoded text, showing masking."""
    # Handle batch dimension
    if input_ids.dim() > 1:
        input_ids = input_ids[0]
    if assistant_mask.dim() > 1:
        assistant_mask = assistant_mask[0]

    console.print("\n[bold]Full Decoded Text:[/bold]")
    console.rule(style="dim")


    full_text = Text()
    for i, token_id in enumerate(input_ids):
        token = tokenizer.decode([token_id])

        if i == 0:
            full_text.append(token, style="dim white")
        elif i < len(assistant_mask):
            if assistant_mask[i]:
                full_text.append(token, style="bold green")
            else:
                full_text.append(token, style="dim white")
        else:
            full_text.append(token, style="dim red")

    console.print(full_text)

    # Statistics
    console.print("\n[bold]Statistics:[/bold]", style="cyan")
    total_tokens = len(input_ids)
    trained_tokens = int(assistant_mask.sum().item()) if len(assistant_mask) > 0 else 0
    ignored_tokens = len(assistant_mask) - trained_tokens

    console.print(f"Total tokens: {total_tokens}")
    console.print(f"Trained tokens: [green]{trained_tokens}[/green] ({100 * trained_tokens / total_tokens:.1f}%)")
    console.print(f"Ignored tokens: [dim]{ignored_tokens}[/dim] ({100 * ignored_tokens / total_tokens:.1f}%)")


def main():
    traces_files = list(Path("hud/rl/tests/data").glob("traces_*.json"))
    if not traces_files:
        console.print("[yellow]No traces files found. Please run the test_actor.py script first.[/yellow]")
        return
    traces_file = max(traces_files, key=lambda x: x.stat().st_mtime)    
    with open(traces_file) as f:
        traces_data = json.load(f)

    console.print(f"[bold cyan]Loaded {len(traces_data)} traces from {traces_file}[/bold cyan]")

    # Convert to Trace objects
    traces = [Trace(**trace_data) for trace_data in traces_data]

    # Initialize processor
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    console.print(f"\n[dim]Loading processor for {model_name}...[/dim]")
    processor = Qwen2VLProcessor.from_pretrained(model_name)

    num_to_show = min(len(traces), 10)

    for i in range(num_to_show):
        console.print(f"\n[bold cyan]{'=' * 100}[/bold cyan]")
        console.print(f"[bold]Processing Trace {i+1}/{len(traces)}[/bold]")

        # Process the trace
        processed_inputs, old_logprobs = preprocess_traces([traces[i]], processor)[0]

        # Show shape information
        console.print("\n[bold]Shape Information:[/bold]")
        for key, value in processed_inputs.items():
            console.print(f"  {key}: shape={value.shape}, dtype={value.dtype}") # type: ignore

        console.print(f"  old_logprobs: shape={old_logprobs.shape}, dtype={old_logprobs.dtype}")

        # Visualize tokenization and masking
        visualize_tokenization(
            processed_inputs["input_ids"],
            processed_inputs["assistant_mask"],
            processor.tokenizer # type: ignore
        )

        # Pause between traces
        if i < num_to_show - 1:
            console.print("\n[dim italic]Press Enter to see next trace...[/dim italic]")
            input()

    console.print(f"\n[bold green]Test completed! Processed {num_to_show}/{len(traces)} traces.[/bold green]")


if __name__ == "__main__":
    main()