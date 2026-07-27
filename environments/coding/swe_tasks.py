"""The SWE-bench Pro task source: fetch instances, build their images, mint rows.

Run as a script to add instances — it fetches the dataset row (HuggingFace)
and the official ``run_script.sh`` + ``parser.py`` (scaleapi/SWE-bench_Pro-os)
into ``instances/<id>/``, then builds ``Dockerfile.hud`` with the instance's
prebuilt image (``jefzda/sweap-images:<dockerhub_tag>``) as ``BASE`` and its
assets baked in; image refs land in ``.hud-images.json``::

    uv run swe_tasks.py instance_NodeBB__NodeBB-04998908...-vnan
    uv run swe_tasks.py <id>... --push registry.io/acme   # push for cloud runtimes
    uv run swe_tasks.py <id>... --fetch-only

Imported, it exposes one ``Task`` row per fetched instance, stamped with its
image so container placements run each instance in its own image::

    hud eval swe_tasks.py claude
    hud eval swe_tasks.py claude --task-ids nodebb-04998908

The environment lives inside the image (``env.py`` plus the baked instance
assets). This module must not import it: rows whose module declares no env
are routed to their images instead of a local serve.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
from hud.eval import Task
from hud.eval.runtime import RuntimeConfig

ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = ROOT / "instances"
IMAGES_MANIFEST = ROOT / ".hud-images.json"

DATASET_API = "https://datasets-server.huggingface.co/filter"
SCRIPTS_RAW = "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/run_scripts"
BASE_IMAGE = "jefzda/sweap-images"


# ─── task rows ───────────────────────────────────────────────────────────


def _slug(row: dict[str, Any]) -> str:
    """A short, readable slug: repo tail + commit prefix.

    ``instance_NodeBB__NodeBB-04998908...-vnan`` -> ``nodebb-04998908``.
    """
    repo_tail = row["repo"].split("/")[-1].lower()
    commit = row["instance_id"].removeprefix(f"instance_{row['repo'].replace('/', '__')}-")
    return f"{repo_tail}-{commit[:8]}"


def _load() -> list[Task]:
    images: dict[str, str] = json.loads(IMAGES_MANIFEST.read_text("utf-8")) if IMAGES_MANIFEST.is_file() else {}
    rows = [
        json.loads((instance_dir / "instance.json").read_text("utf-8"))
        for instance_dir in sorted(INSTANCES_DIR.iterdir() if INSTANCES_DIR.is_dir() else [])
        if (instance_dir / "instance.json").is_file()
    ]
    return [
        Task(
            env="coding",
            id=row["instance_id"],
            slug=_slug(row),
            columns={
                "repo": row["repo"],
                "language": row["repo_language"],
                "categories": row["issue_categories"],
            },
            runtime_config=(RuntimeConfig(image=images[row["instance_id"]]) if row["instance_id"] in images else None),
        )
        for row in rows
    ]


tasks = _load()


# ─── instance fetching + image builds (script mode) ──────────────────────


def fetch_instance(client: httpx.Client, instance_id: str) -> dict[str, Any]:
    """One dataset row by instance id, via the datasets-server filter API."""
    where = urllib.parse.quote(f"\"instance_id\"='{instance_id}'")
    url = f"{DATASET_API}?dataset=ScaleAI%2FSWE-bench_Pro&config=default&split=test&where={where}&length=1"
    payload = client.get(url).raise_for_status().json()
    rows = payload.get("rows", [])
    if not rows:
        raise SystemExit(f"instance not found in ScaleAI/SWE-bench_Pro: {instance_id}")
    return rows[0]["row"]


def fetch_assets(instance_id: str) -> dict[str, Any]:
    """Fetch the row + official scripts into ``instances/<id>/``. Idempotent."""
    target = INSTANCES_DIR / instance_id
    row_path = target / "instance.json"
    if row_path.is_file():
        print(f"[{instance_id}] assets cached")
        return json.loads(row_path.read_text("utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        row = fetch_instance(client, instance_id)
        for name in ("run_script.sh", "parser.py"):
            text = client.get(f"{SCRIPTS_RAW}/{instance_id}/{name}").raise_for_status().text
            (target / name).write_text(text, encoding="utf-8", newline="\n")
    row_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[{instance_id}] fetched row + scripts")
    return row


def build(instance_id: str, row: dict[str, Any], push: str | None) -> str:
    """Build (and optionally push) the instance image: Dockerfile.hud with the
    instance's prebuilt image as BASE and its assets baked in."""
    ref = f"{push}/swe-bench-pro:{instance_id}" if push else f"hud-swe-bench-pro:{instance_id}"
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--build-arg",
            f"BASE={BASE_IMAGE}:{row['dockerhub_tag']}",
            "--build-arg",
            f"INSTANCE_ID={instance_id}",
            "--tag",
            ref,
            "--file",
            str(ROOT / "Dockerfile.hud"),
            str(ROOT),
        ],
        check=True,
    )
    if push:
        subprocess.run(["docker", "push", ref], check=True)
    return ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SWE-bench Pro instances and build their images.")
    parser.add_argument("instance_ids", nargs="+", help="SWE-bench Pro instance ids")
    parser.add_argument("--push", help="registry prefix to push built images to")
    parser.add_argument("--fetch-only", action="store_true", help="fetch assets, skip builds")
    args = parser.parse_args()

    images: dict[str, str] = json.loads(IMAGES_MANIFEST.read_text("utf-8")) if IMAGES_MANIFEST.is_file() else {}
    for instance_id in args.instance_ids:
        row = fetch_assets(instance_id)
        if args.fetch_only:
            continue
        images[instance_id] = build(instance_id, row, args.push)
        print(f"[{instance_id}] image {images[instance_id]}")

    if not args.fetch_only:
        IMAGES_MANIFEST.write_text(json.dumps(images, indent=2) + "\n", encoding="utf-8")
        print(f"manifest: {IMAGES_MANIFEST}")


if __name__ == "__main__":
    sys.exit(main())
