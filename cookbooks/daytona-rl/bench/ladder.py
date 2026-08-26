"""Concurrency ladder for Daytona sandboxes: N in {1, 8, 32, 128}.

    uv run ladder.py                      # infra mode, no LLM spend
    uv run ladder.py --mode rollout       # real Claude rollouts (costs tokens)
    uv run ladder.py --levels 1,8         # override the ladder

Infra mode measures the thing we actually doubt: how sandbox spin-up and the
one-SSH-connection-per-sandbox transport (runtime.py:780) behave under fan-out.
Each worker acquires a sandbox, opens a *fresh control connection* to it, does
one real handshake (`client.manifest()`), and exits — no agent, no tokens.

Writes ladder.csv and ladder.svg.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import resource
import statistics
import time
from pathlib import Path

from daytona import Image
from hud.clients import connect
from hud.eval import DaytonaRuntime

from env import fix_calc
from snapshot import snapshot_name
from timing import TimedProvider

SNAPSHOT = snapshot_name()
OUT_CSV = Path("ladder.csv")
OUT_SVG = Path("ladder.svg")


async def one_infra(provider, idx: int) -> dict:
    """Acquire a sandbox, open a fresh control connection, handshake, tear down."""
    t0 = time.perf_counter()
    row = {"idx": idx, "spin_up_s": None, "connect_s": None, "ok": False, "error": ""}
    try:
        async with provider(fix_calc()) as rt:
            row["spin_up_s"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            async with connect(rt) as client:
                assert client.manifest is not None
                row["connect_s"] = time.perf_counter() - t1
            row["ok"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["total_s"] = time.perf_counter() - t0
    return row


async def one_rollout(provider, idx: int) -> dict:
    from hud.agents.claude import ClaudeAgent

    timed = TimedProvider(provider)
    t0 = time.perf_counter()
    row = {"idx": idx, "spin_up_s": None, "connect_s": None, "ok": False, "error": ""}
    try:
        job = await fix_calc().run(ClaudeAgent(), runtime=timed)
        row["ok"] = True
        row["reward"] = job.reward
        row["spin_up_s"] = timed.spin_ups[0] if timed.spin_ups else None
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["total_s"] = time.perf_counter() - t0
    return row


async def run_level(n: int, mode: str) -> dict:
    provider = DaytonaRuntime(SNAPSHOT, image=Image.from_dockerfile("Dockerfile.hud"))
    worker = one_infra if mode == "infra" else one_rollout

    t0 = time.perf_counter()
    rows = await asyncio.gather(*(worker(provider, i) for i in range(n)))
    wall = time.perf_counter() - t0

    ok = [r for r in rows if r["ok"]]
    spin = sorted(r["spin_up_s"] for r in ok if r["spin_up_s"] is not None)
    errors = [r["error"] for r in rows if r["error"]]

    def pct(p: float) -> float | None:
        if not spin:
            return None
        return spin[min(len(spin) - 1, int(p * len(spin)))]

    summary = {
        "n": n,
        "mode": mode,
        "wall_s": round(wall, 2),
        "ok": len(ok),
        "failed": n - len(ok),
        "per_min": round(len(ok) / wall * 60, 1) if wall else 0.0,
        "spin_min_s": round(spin[0], 2) if spin else None,
        "spin_med_s": round(statistics.median(spin), 2) if spin else None,
        "spin_p90_s": round(pct(0.9), 2) if spin else None,
        "spin_max_s": round(spin[-1], 2) if spin else None,
    }
    print(f"[N={n:>3}] {summary}")
    for e in dict.fromkeys(errors):
        print(f"        error: {e[:200]}")
    return summary


def chart(rows: list[dict]) -> None:
    """Minimal hand-rolled SVG: throughput bars + median spin-up line."""
    w, h, pad = 640, 320, 56
    plot_w, plot_h = w - 2 * pad, h - 2 * pad
    max_rate = max((r["per_min"] or 0) for r in rows) or 1
    max_spin = max((r["spin_max_s"] or 0) for r in rows) or 1
    bar_w = plot_w / max(len(rows), 1) * 0.55

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="ui-sans-serif,system-ui,sans-serif">',
        f'<rect width="{w}" height="{h}" fill="#fff"/>',
        f'<text x="{pad}" y="28" font-size="14" font-weight="600">'
        f"Daytona concurrency ladder — {rows[0]['mode']} mode</text>",
        f'<line x1="{pad}" y1="{pad + plot_h}" x2="{pad + plot_w}" y2="{pad + plot_h}" '
        f'stroke="#999"/>',
    ]
    pts = []
    for i, r in enumerate(rows):
        cx = pad + plot_w * (i + 0.5) / len(rows)
        bh = plot_h * (r["per_min"] or 0) / max_rate
        parts.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{pad + plot_h - bh:.1f}" '
            f'width="{bar_w:.1f}" height="{bh:.1f}" fill="#6366f1" opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{pad + plot_h + 16:.1f}" font-size="11" '
            f'text-anchor="middle">N={r["n"]}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{pad + plot_h - bh - 6:.1f}" font-size="10" '
            f'text-anchor="middle" fill="#4338ca">{r["per_min"]}/min</text>'
        )
        sy = pad + plot_h - plot_h * (r["spin_med_s"] or 0) / max_spin
        pts.append(f"{cx:.1f},{sy:.1f}")
        parts.append(f'<circle cx="{cx:.1f}" cy="{sy:.1f}" r="3.5" fill="#ef4444"/>')
        if r["failed"]:
            parts.append(
                f'<text x="{cx:.1f}" y="{pad - 8:.1f}" font-size="10" '
                f'text-anchor="middle" fill="#dc2626">{r["failed"]} failed</text>'
            )
    parts.append(
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="#ef4444" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{pad}" y="{h - 14}" font-size="11" fill="#4338ca">bars: completions/min</text>'
        f'<text x="{pad + 190}" y="{h - 14}" font-size="11" fill="#ef4444">'
        f"line: median spin-up (max {max_spin}s)</text>"
    )
    parts.append("</svg>")
    OUT_SVG.write_text("\n".join(parts))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("infra", "rollout"), default="infra")
    ap.add_argument("--levels", default="1,8,32,128")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = max(levels) * 16 + 256
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(want, hard), hard))
    print(f"fd limit: {soft} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]} (hard {hard})")

    rows = []
    for n in levels:
        rows.append(await run_level(n, args.mode))

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    chart(rows)
    print(f"\nwrote {OUT_CSV} and {OUT_SVG}")


if __name__ == "__main__":
    asyncio.run(main())
