"""Mark existing public environments Featured from the template workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel

from hud.utils.platform import PlatformClient

CATALOG = Path(__file__).with_name("featured_environments.json")


class Example(BaseModel):
    registry_id: UUID
    taskset_id: UUID
    example_task_slug: str
    sample_trace_id: UUID


def catalog() -> dict[str, Example]:
    return {
        name: Example.model_validate(value)
        for name, value in json.loads(CATALOG.read_text()).items()
    }


def mark_featured(name: str, example: Example, client: PlatformClient) -> None:
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
    ):
        raise RuntimeError("Featured changes run only in the main-branch template workflow")

    registry_id = str(example.registry_id)
    registry = client.get(f"/registry/{registry_id}")
    if registry["id"] != registry_id or registry["name"] != name or registry["public"] is not True:
        raise ValueError(f"{name}: target must be the existing public environment in the catalog")

    response = client.put(f"/admin/registry/{registry_id}/featured", json={"featured": True})
    if response["id"] != registry_id or response["featured"] is not True:
        raise ValueError("Platform did not confirm Featured status")


def main() -> None:
    examples = catalog()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=list(examples))
    args = parser.parse_args()
    example = examples[args.name]
    mark_featured(args.name, example, PlatformClient.from_settings())
    print(f"{args.name}: Featured — https://hud.ai/environments/{example.registry_id}")


if __name__ == "__main__":
    main()
