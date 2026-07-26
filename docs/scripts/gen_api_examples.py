"""Fill the endpoint blocks in docs/platform/rest-api.mdx from an OpenAPI document.

The page owns its prose. This script owns everything between the markers:

    {/* api:GET /v2/jobs */}
    ... generated parameter table, request, and response ...
    {/* /api */}

Run from the repository root after the API changes:

    python docs/scripts/gen_api_examples.py

The spec is downloaded to a gitignored `docs/openapi.json` on first run. Pass
--check to fail instead of writing (for CI), and --list to print every
operation in the spec that the page does not yet mention.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

# This file lives at docs/scripts/; the docs root is one level up.
DOCS = Path(__file__).resolve().parent.parent
SPEC_PATH = DOCS / "openapi.json"
PAGE_PATH = DOCS / "platform" / "rest-api.mdx"
BASE_URL = "https://api.beta.hud.ai"

MARKER = re.compile(
    r"(?P<open>\{/\* api:(?P<method>[A-Z]+) (?P<path>\S+) \*/\}\n)"
    r".*?"
    r"(?P<close>\{/\* /api \*/\})",
    re.DOTALL,
)

# Placeholder values by string format, so examples read like real payloads.
FORMAT_SAMPLES = {
    "uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "date-time": "2026-07-26T17:04:11Z",
    "date": "2026-07-26",
    "email": "user@example.com",
    "uri": "https://example.com",
}

MAX_PROPERTIES = 12
MAX_DEPTH = 4


class SpecError(RuntimeError):
    pass


def load_spec(refresh: bool = False) -> dict[str, Any]:
    if refresh or not SPEC_PATH.exists():
        url = f"{BASE_URL}/openapi.json"
        print(f"fetching {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                SPEC_PATH.write_bytes(response.read())
        except OSError as exc:
            raise SpecError(f"could not fetch {url}: {exc}") from None
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def deref(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    seen: set[str] = set()
    while "$ref" in schema:
        ref = schema["$ref"]
        if ref in seen:
            return {}
        seen.add(ref)
        node: Any = spec
        for part in ref.lstrip("#/").split("/"):
            node = node.get(part, {})
        schema = node
    return schema


def collapse(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve a schema to something with a usable `type`.

    Unions are common in this spec because optional fields serialize as
    `anyOf: [T, null]`; pick the first non-null branch. `allOf` is merged.
    """
    schema = deref(spec, schema)
    if "allOf" in schema:
        merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for part in schema["allOf"]:
            part = collapse(spec, part)
            merged["properties"].update(part.get("properties", {}))
            merged["required"].extend(part.get("required", []))
        return merged
    for key in ("anyOf", "oneOf"):
        if key in schema:
            branches = [b for b in schema[key] if deref(spec, b).get("type") != "null"]
            if branches:
                return collapse(spec, branches[0])
            return {"type": "null"}
    return schema


def type_name(spec: dict[str, Any], schema: dict[str, Any]) -> str:
    raw = deref(spec, schema)
    nullable = False
    for key in ("anyOf", "oneOf"):
        if key in raw:
            nullable = any(deref(spec, b).get("type") == "null" for b in raw[key])
    resolved = collapse(spec, schema)
    kind = resolved.get("type", "object")
    if kind == "array":
        inner = collapse(spec, resolved.get("items", {}))
        kind = f"{inner.get('type', 'object')}[]"
    elif resolved.get("enum"):
        values = resolved["enum"]
        kind = " \\| ".join(str(v) for v in values) if len(values) <= 5 else "enum"
    elif resolved.get("format") in ("uuid", "date-time"):
        kind = resolved["format"]
    return f"{kind}?" if nullable else kind


def sample(spec: dict[str, Any], schema: dict[str, Any], depth: int = 0) -> Any:
    resolved = collapse(spec, schema)
    for key in ("example", "default"):
        if key in resolved and resolved[key] is not None:
            return resolved[key]
    if resolved.get("enum"):
        return resolved["enum"][0]

    kind = resolved.get("type")
    if kind == "object" or "properties" in resolved:
        if depth >= MAX_DEPTH:
            return {}
        properties: dict[str, Any] = resolved.get("properties", {})
        required = [k for k in resolved.get("required", []) if k in properties]
        ordered = required + [k for k in properties if k not in required]
        return {
            key: sample(spec, properties[key], depth + 1) for key in ordered[:MAX_PROPERTIES]
        }
    if kind == "array":
        if depth >= MAX_DEPTH:
            return []
        return [sample(spec, resolved.get("items", {}), depth + 1)]
    if kind == "string":
        return FORMAT_SAMPLES.get(resolved.get("format", ""), "string")
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    return None


def operation(spec: dict[str, Any], method: str, path: str) -> dict[str, Any]:
    try:
        return spec["paths"][path][method.lower()]
    except KeyError:
        raise SpecError(f"{method} {path} is not in the spec") from None


def clean_text(text: str) -> str:
    """Normalize handler docstrings to the docs house style.

    Descriptions come from Python source, so they arrive with em dashes and
    RST-style double backticks that do not belong in the rendered page.
    """
    for dash in ("\u2014", "\u2013"):
        text = text.replace(f" {dash} ", " - ").replace(dash, "-")
    return text.replace("``", "`").replace("\n", " ").strip()


def is_binary(spec: dict[str, Any], schema: dict[str, Any]) -> bool:
    resolved = collapse(spec, schema)
    return resolved.get("format") == "binary" or "contentMediaType" in resolved


def parameter_rows(spec: dict[str, Any], op: dict[str, Any]) -> list[str]:
    rows = []
    for param in op.get("parameters", []):
        schema = param.get("schema", {})
        resolved = collapse(spec, schema)
        default = resolved.get("default")
        notes = clean_text(param.get("description", ""))
        if default is not None and "default" not in notes.lower():
            notes = f"{notes} Defaults to `{default}`." if notes else f"Defaults to `{default}`."
        flag = "yes" if param.get("required") else ""
        rows.append(
            f"| `{param['name']}` | {param['in']} | `{type_name(spec, schema)}` | {flag} | {notes} |"
        )
    return rows


def body_schema(spec: dict[str, Any], op: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    content = op.get("requestBody", {}).get("content", {})
    for media in ("application/json", "multipart/form-data"):
        if media in content:
            return media, content[media].get("schema", {})
    return None


def success_response(spec: dict[str, Any], op: dict[str, Any]) -> tuple[str, Any] | None:
    for code, response in op.get("responses", {}).items():
        if not code.startswith("2"):
            continue
        schema = response.get("content", {}).get("application/json", {}).get("schema")
        if schema is None:
            return code, None
        return code, sample(spec, schema)
    return None


def curl_block(spec: dict[str, Any], method: str, path: str, op: dict[str, Any]) -> str:
    url = BASE_URL + re.sub(r"\{(\w+)\}", r"<\1>", path)
    required_query = [
        p for p in op.get("parameters", []) if p["in"] == "query" and p.get("required")
    ]
    if required_query:
        query = "&".join(f"{p['name']}=<{p['name']}>" for p in required_query)
        url = f'"{url}?{query}"'

    lines = [f"curl {url} \\", '  -H "Authorization: Bearer $HUD_API_KEY"']
    if method != "GET":
        lines[-1] += " \\"
        lines.append(f"  -X {method}")

    body = body_schema(spec, op)
    if body:
        media, schema = body
        lines[-1] += " \\"
        if media == "multipart/form-data":
            fields = collapse(spec, schema).get("properties", {})
            parts = [
                f'  -F "{name}=@./{name}"' if is_binary(spec, sub) else f'  -F "{name}=<{name}>"'
                for name, sub in fields.items()
            ]
            lines.extend(" \\\n".join(parts).splitlines())
        else:
            payload = json.dumps(sample(spec, schema), indent=2)
            payload = "\n".join(
                ("  " + line if i else line) for i, line in enumerate(payload.splitlines())
            )
            lines.append('  -H "Content-Type: application/json" \\')
            lines.append(f"  -d '{payload}'")
    return "\n".join(lines)


def render(spec: dict[str, Any], method: str, path: str) -> str:
    op = operation(spec, method, path)
    parts: list[str] = []

    rows = parameter_rows(spec, op)
    if rows:
        parts.append(
            "\n".join(
                [
                "| Parameter | In | Type | Required | Description |",
                "| --- | --- | --- | --- | --- |",
            ]
                + rows
            )
        )

    parts.append("```bash Request\n" + curl_block(spec, method, path, op) + "\n```")

    result = success_response(spec, op)
    if result:
        code, payload = result
        if payload is None:
            parts.append(f"Returns `{code}` with an empty body.")
        else:
            parts.append(
                f"```json Response {code}\n" + json.dumps(payload, indent=2) + "\n```"
            )
    return "\n\n".join(parts)


def documented(page: str) -> list[tuple[str, str]]:
    return [(m.group("method"), m.group("path")) for m in MARKER.finditer(page)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the page is stale")
    parser.add_argument("--list", action="store_true", help="print undocumented operations")
    parser.add_argument("--refresh", action="store_true", help="re-download the spec first")
    args = parser.parse_args()

    spec = load_spec(refresh=args.refresh)
    page = PAGE_PATH.read_text(encoding="utf-8")

    if args.list:
        covered = set(documented(page))
        for path, methods in spec["paths"].items():
            for method in methods:
                if method.upper() in ("GET", "POST", "PATCH", "PUT", "DELETE"):
                    if (method.upper(), path) not in covered:
                        print(f"{method.upper()} {path}")
        return 0

    def replace(match: re.Match[str]) -> str:
        body = render(spec, match.group("method"), match.group("path"))
        return f"{match.group('open')}\n{body}\n\n{match.group('close')}"

    updated = MARKER.sub(replace, page)
    count = len(documented(page))

    if args.check:
        if updated != page:
            print(f"{PAGE_PATH.name} is out of date; run scripts/gen_api_examples.py")
            return 1
        print(f"{count} endpoint blocks up to date")
        return 0

    PAGE_PATH.write_text(updated, encoding="utf-8")
    print(f"filled {count} endpoint blocks in {PAGE_PATH.name}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
