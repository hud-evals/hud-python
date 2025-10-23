#!/usr/bin/env python
"""Inspect tokens and logprobs for each turn in detail."""
from pathlib import Path
import json

# Load trace
traces_files = list(Path('hud/rl/tests/data').glob('traces_*.json'))
traces_file = max(traces_files, key=lambda x: x.stat().st_mtime)
with open(traces_file) as f:
    traces_data = json.load(f)

trace_dict = traces_data[0]

# Extract turns from trace steps
turns = []
for step in trace_dict['trace']:
    if step.get('function.result_type') == 'AgentResponse':
        result = json.loads(step['function.result'])
        raw = result['raw']
        turns.append(raw)

print(f"Total turns: {len(turns)}\n")

for idx, turn in enumerate(turns):
    print("=" * 80)
    print(f"TURN {idx}")
    print("=" * 80)

    # Extract prompt tokens
    prompt_logprobs_data = turn.get('prompt_logprobs', [])
    prompt_ids = []
    prompt_logps = []
    for entry in prompt_logprobs_data:
        if entry:
            token_id_str, info = next(iter(entry.items()))
            prompt_ids.append(int(token_id_str))
            prompt_logps.append(info.get('logprob', 0.0) if isinstance(info, dict) else 0.0)

    # Extract completion tokens
    completion_data = turn['choices'][0]['logprobs']['content']
    completion_ids = []
    completion_logps = []
    completion_tokens = []
    for item in completion_data:
        token_label = item.get('token', '')
        if token_label.startswith('token_id:'):
            token_id = int(token_label.split(':', 1)[1])
            completion_ids.append(token_id)
            completion_logps.append(item.get('logprob', 0.0))
            completion_tokens.append(token_label)

    print(f"\nPrompt: {len(prompt_ids)} tokens")
    print(f"Completion: {len(completion_ids)} tokens")

    # Show ALL prompt tokens
    print("\nAll prompt tokens:")
    for i in range(len(prompt_ids)):
        token_id = prompt_ids[i]
        logprob = prompt_logps[i]
        print(f"  [{i:4d}] {token_id:6d} logp={logprob:.16f}")

    # Show ALL completion tokens
    print(f"\nAll completion tokens:")
    for i in range(len(completion_ids)):
        token_id = completion_ids[i]
        logprob = completion_logps[i]
        token_label = completion_tokens[i]
        print(f"  [{i:4d}] {token_id:6d} logp={logprob:.16f} {token_label}")

    # If there's a next turn, check overlap
    if idx < len(turns) - 1:
        next_turn = turns[idx + 1]

        # Extract next prompt tokens
        next_prompt_logprobs_data = next_turn.get('prompt_logprobs', [])
        next_prompt_ids = []
        for entry in next_prompt_logprobs_data:
            if entry:
                token_id_str, _ = next(iter(entry.items()))
                next_prompt_ids.append(int(token_id_str))

        print(f"\n--- Checking overlap with Turn {idx+1} ---")
        print(f"Current prompt length: {len(prompt_ids)}")
        print(f"Next prompt length: {len(next_prompt_ids)}")

        # Check if current prompt is prefix of next
        prefix_match = True
        for i in range(len(prompt_ids)):
            if i >= len(next_prompt_ids) or prompt_ids[i] != next_prompt_ids[i]:
                prefix_match = False
                print(f"Prefix MISMATCH at position {i}")
                print(f"  Current: {prompt_ids[i]}")
                print(f"  Next: {next_prompt_ids[i] if i < len(next_prompt_ids) else 'N/A'}")
                break

        if prefix_match:
            print("✓ Current prompt is prefix of next prompt")

            # Check completion overlap
            start_pos = len(prompt_ids)
            matched = 0
            for i, comp_id in enumerate(completion_ids):
                pos = start_pos + i
                if pos >= len(next_prompt_ids):
                    break
                if next_prompt_ids[pos] != comp_id:
                    print(f"\nCompletion matching stopped at offset {i}")
                    print(f"  Completion token: {comp_id}")
                    print(f"  Next prompt token: {next_prompt_ids[pos]}")
                    break
                matched += 1

            print(f"Completion tokens matched: {matched}/{len(completion_ids)}")

            # Show gap info
            if matched < len(completion_ids):
                gap_start = start_pos + matched
                gap_end = len(next_prompt_ids)
                gap_len = gap_end - gap_start
                print(f"\nGap: {gap_len} tokens from position {gap_start} to {gap_end}")

    print("\n")
