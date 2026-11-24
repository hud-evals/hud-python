"""``${NAME}`` and ``${NAME:-default}`` values resolved from the process environment."""

from __future__ import annotations

import os
import re

ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_templates(env: dict[str, str]) -> dict[str, str]:
    """Resolve each value that is wholly a template; other values pass through unchanged.

    ``${NAME}`` takes ``NAME`` from the process environment; ``${NAME:-default}`` falls
    back to ``default`` when ``NAME`` is unset or empty. A template whose variable is
    unset and has no default raises ``ValueError`` naming the key and the variable.
    """
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = ENV_TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
            continue
        name, default = match.group(1), match.group(2)
        runtime_value = os.environ.get(name)
        if runtime_value is not None and (runtime_value or default is None):
            resolved[key] = runtime_value
        elif default is not None:
            resolved[key] = default
        else:
            raise ValueError(
                f"env template for {key!r} needs {name!r}; "
                "the environment it resolves in must provide it"
            )
    return resolved
