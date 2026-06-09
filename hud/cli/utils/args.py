"""Shared Docker argument parsing helpers."""

from __future__ import annotations


def _parse_kv_flag(args: list[str], i: int, short: str, long: str) -> tuple[str, str, int] | None:
    """Try to consume a key=value flag at position *i*. Returns (key, value, new_i) or None."""
    arg = args[i]

    # -e VAL or --env VAL
    if arg in (short, long) and i + 1 < len(args):
        val = args[i + 1]
        if "=" in val:
            k, v = val.split("=", 1)
            return k.strip(), v.strip(), i + 2

    # --env=VAL
    prefix = f"{long}="
    if arg.startswith(prefix):
        val = arg[len(prefix) :]
        if "=" in val:
            k, v = val.split("=", 1)
            return k.strip(), v.strip(), i + 1

    return None


def split_docker_passthrough(
    args: list[str],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Split a raw arg list into env vars, build args, and remaining passthrough args.

    Returns ``(env_vars, build_args, remaining)``.
    """
    env_vars: dict[str, str] = {}
    build_args: dict[str, str] = {}
    remaining: list[str] = []
    i = 0
    while i < len(args):
        env = _parse_kv_flag(args, i, "-e", "--env")
        if env:
            env_vars[env[0]] = env[1]
            i = env[2]
            continue
        ba = _parse_kv_flag(args, i, "--build-arg", "--build-arg")
        if ba:
            build_args[ba[0]] = ba[1]
            i = ba[2]
            continue
        remaining.append(args[i])
        i += 1
    return env_vars, build_args, remaining
