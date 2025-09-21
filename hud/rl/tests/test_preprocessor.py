#!/usr/bin/env python
import json
from pathlib import Path

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

    console.print("\n[bold]Tokenization Visualization[/bold]")
    console.print(Text("TRAINED", style="bold green") + " / " + Text("IGNORED", style="dim") + " / " + Text("LAST TOKEN (no target)", style="dim red"))
    console.rule(style="dim")

    chunk_size = 15
    for start_idx in range(0, len(input_ids), chunk_size):
        end_idx = min(start_idx + chunk_size, len(input_ids))

        id_text = Text()
        dec_text = Text()

        id_text.append("IDs:  ")
        dec_text.append("Text: ")

        for i in range(start_idx, end_idx):
            token_id = input_ids[i].item()
            decoded_token = tokenizer.decode([token_id])

            # Clean up token for display
            display_token = decoded_token.replace('\n', '⏎').replace('\t', '→').replace(' ', '·')
            if len(display_token) > 8:
                display_token = display_token[:7] + "…"

            # Determine style based on mask
            # assistant_mask[j] tells us if we train to predict token j+1
            # So token i should be green if assistant_mask[i-1] is True
            if i == 0:
                # First token is never predicted (no previous token)
                style = "dim white"
            elif i <= len(assistant_mask):
                style = "bold green" if assistant_mask[i-1] else "dim white"
            else:
                style = "dim red"

            id_text.append(f"{token_id:6d} ", style=style)
            dec_text.append(f"{display_token:8s} ", style=style)

        # Print position header and rows
        console.print(f"\n[dim]Position {start_idx:4d}-{end_idx-1:4d}:[/dim]")
        console.print(id_text)
        console.print(dec_text)


    console.print("\n[bold]Full Decoded Text:[/bold]")
    console.rule(style="dim")


    full_text = Text()
    for i, token_id in enumerate(input_ids):
        token = tokenizer.decode([token_id])

        if i == 0:
            full_text.append(token, style="dim white")
        elif i <= len(assistant_mask):
            if assistant_mask[i-1]:
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
    # Load traces from JSON
    traces_file = Path("hud/rl/tests/data/traces_de8ea147-3c52-4117-ad24-d1dbaa39a088.json")
    with open(traces_file) as f:
        traces_data = json.load(f)

    console.print(f"[bold cyan]Loaded {len(traces_data)} traces from {traces_file}[/bold cyan]")

    # Convert to Trace objects
    traces = [Trace(**trace_data) for trace_data in traces_data]

    # Initialize processor
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    console.print(f"\n[dim]Loading processor for {model_name}...[/dim]")
    processor = Qwen2VLProcessor.from_pretrained(model_name)

    # Process and visualize all traces
    num_to_show = len(traces)

    for i in range(num_to_show):
        console.print(f"\n[bold cyan]{'=' * 100}[/bold cyan]")
        console.print(f"[bold]Processing Trace {i+1}/{len(traces)}[/bold]")

        # Process the trace
        processed = preprocess_traces([traces[i]], processor)[0]

        # Show shape information
        console.print("\n[bold]Shape Information:[/bold]")
        for key, value in processed.items():
            console.print(f"  {key}: shape={value.shape}, dtype={value.dtype}") # type: ignore

        # Visualize tokenization and masking
        visualize_tokenization(
            processed['input_ids'],
            processed['assistant_mask'],
            processor.tokenizer # type: ignore
        )

        # Pause between traces
        if i < num_to_show - 1:
            console.print("\n[dim italic]Press Enter to see next trace...[/dim italic]")
            input()

    console.print(f"\n[bold green]Test completed! Processed {num_to_show}/{len(traces)} traces.[/bold green]")


if __name__ == "__main__":
    main()