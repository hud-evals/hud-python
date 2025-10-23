#!/usr/bin/env python
"""Interactive TUI for inspecting turn tokens and logprobs."""
from pathlib import Path
import json
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

console = Console()


def load_turns():
    """Load turns from trace data."""
    traces_files = list(Path('hud/rl/tests/data').glob('traces_*.json'))
    if not traces_files:
        console.print("[red]No trace files found![/red]")
        sys.exit(1)

    traces_file = max(traces_files, key=lambda x: x.stat().st_mtime)
    with open(traces_file) as f:
        traces_data = json.load(f)

    trace_dict = traces_data[0]

    turns = []
    for step in trace_dict['trace']:
        if step.get('function.result_type') == 'AgentResponse':
            result = json.loads(step['function.result'])
            raw = result['raw']

            # Extract prompt tokens
            prompt_logprobs_data = raw.get('prompt_logprobs', [])
            prompt_ids = []
            prompt_logps = []
            for entry in prompt_logprobs_data:
                if entry:
                    token_id_str, info = next(iter(entry.items()))
                    prompt_ids.append(int(token_id_str))
                    prompt_logps.append(info.get('logprob', 0.0) if isinstance(info, dict) else 0.0)

            # Extract completion tokens
            completion_data = raw['choices'][0]['logprobs']['content']
            completion_ids = []
            completion_logps = []
            for item in completion_data:
                token_label = item.get('token', '')
                if token_label.startswith('token_id:'):
                    token_id = int(token_label.split(':', 1)[1])
                    completion_ids.append(token_id)
                    completion_logps.append(item.get('logprob', 0.0))

            turns.append({
                'prompt_ids': prompt_ids,
                'prompt_logps': prompt_logps,
                'completion_ids': completion_ids,
                'completion_logps': completion_logps,
            })

    return turns


def show_turn_overview(turns, current_idx):
    """Show overview of all turns."""
    table = Table(title=f"Turns Overview (showing turn {current_idx})")
    table.add_column("Turn", style="cyan")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Completion Tokens", justify="right")
    table.add_column("Status", justify="center")

    for idx, turn in enumerate(turns):
        status = "→" if idx == current_idx else ""
        style = "bold yellow" if idx == current_idx else ""
        table.add_row(
            f"Turn {idx}",
            str(len(turn['prompt_ids'])),
            str(len(turn['completion_ids'])),
            status,
            style=style
        )

    return table


def show_completion_detail(turn, turn_idx):
    """Show completion tokens in detail."""
    table = Table(title=f"Turn {turn_idx} - Completion Tokens")
    table.add_column("Idx", style="dim", width=6)
    table.add_column("Token ID", justify="right", width=10)
    table.add_column("Logprob", justify="right", width=20)

    for i, (token_id, logprob) in enumerate(zip(turn['completion_ids'], turn['completion_logps'])):
        table.add_row(
            f"[{i}]",
            str(token_id),
            f"{logprob:.16f}"
        )

    return table


def show_comparison(curr_turn, next_turn, curr_idx):
    """Show comparison between current completion and next prompt."""
    curr_len = len(curr_turn['prompt_ids'])
    completion_len = len(curr_turn['completion_ids'])
    expected_end = curr_len + completion_len

    table = Table(title=f"Turn {curr_idx} → Turn {curr_idx + 1} Comparison")
    table.add_column("Pos", width=6)
    table.add_column("Completion Token", justify="right", width=12)
    table.add_column("Completion Logp", justify="right", width=20)
    table.add_column("Next Prompt Token", justify="right", width=12)
    table.add_column("Next Prompt Logp", justify="right", width=20)
    table.add_column("Match", width=8)

    for i in range(completion_len):
        comp_token = curr_turn['completion_ids'][i]
        comp_logp = curr_turn['completion_logps'][i]

        pos = curr_len + i
        if pos < len(next_turn['prompt_ids']):
            next_token = next_turn['prompt_ids'][pos]
            next_logp = next_turn['prompt_logps'][pos]

            token_match = comp_token == next_token
            logp_match = comp_logp == next_logp

            if token_match and logp_match:
                match_str = Text("✓", style="green")
            elif token_match:
                match_str = Text("≈", style="yellow")
            else:
                match_str = Text("✗", style="red")

            style = "" if token_match and logp_match else "red"
        else:
            next_token = "N/A"
            next_logp = "N/A"
            match_str = Text("✗", style="red")
            style = "red"

        table.add_row(
            f"[{i}]",
            str(comp_token),
            f"{comp_logp:.16f}",
            str(next_token),
            f"{next_logp:.16f}" if isinstance(next_logp, float) else next_logp,
            match_str,
            style=style
        )

    # Check if there are mismatches
    mismatches = []
    for i in range(completion_len):
        pos = curr_len + i
        if pos < len(next_turn['prompt_ids']):
            if curr_turn['completion_ids'][i] != next_turn['prompt_ids'][pos]:
                mismatches.append(i)

    if mismatches:
        summary = Text(f"\n⚠ {len(mismatches)} token mismatches found at positions: {mismatches[:10]}", style="bold red")
    else:
        summary = Text(f"\n✓ All {completion_len} tokens match!", style="bold green")

    return Panel(table, subtitle=summary)


def main():
    console.clear()
    turns = load_turns()
    current_idx = 0

    while True:
        console.clear()

        # Show overview
        console.print(show_turn_overview(turns, current_idx))
        console.print()

        # Show current turn completion
        console.print(show_completion_detail(turns[current_idx], current_idx))
        console.print()

        # Show comparison if not last turn
        if current_idx < len(turns) - 1:
            console.print(show_comparison(turns[current_idx], turns[current_idx + 1], current_idx))
        else:
            console.print("[yellow]Last turn - no next turn to compare[/yellow]")

        console.print()
        console.print("[dim]Commands: [n]ext, [p]rev, [q]uit, [number] to jump to turn[/dim]")

        try:
            cmd = console.input(">>> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if cmd == 'q':
            break
        elif cmd == 'n':
            current_idx = min(current_idx + 1, len(turns) - 1)
        elif cmd == 'p':
            current_idx = max(current_idx - 1, 0)
        elif cmd.isdigit():
            idx = int(cmd)
            if 0 <= idx < len(turns):
                current_idx = idx
            else:
                console.print(f"[red]Invalid turn number. Must be 0-{len(turns)-1}[/red]")
                console.input("Press Enter to continue...")

    console.clear()
    console.print("[green]Goodbye![/green]")


if __name__ == "__main__":
    main()
