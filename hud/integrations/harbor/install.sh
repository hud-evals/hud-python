#!/bin/sh
set -eu

root=/controller
python_version=3.12

export UV_PYTHON_INSTALL_DIR="$root/python"
export UV_PYTHON_BIN_DIR="$root/bin"
export UV_NO_CACHE=1
export XDG_CONFIG_HOME="$root/config"
export PATH="$root/bin:$PATH"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq bubblewrap util-linux python3 python3-venv python3-pip git curl ca-certificates
  rm -rf /var/lib/apt/lists/*
elif command -v apk >/dev/null 2>&1; then
  apk add --no-cache bash bubblewrap util-linux python3 py3-pip git curl ca-certificates
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y bubblewrap util-linux python3 python3-pip git curl ca-certificates
  dnf clean all
else
  echo "hud: Harbor environments require an apt-, apk-, or dnf-based image" >&2
  exit 1
fi

if [ "${1:-}" = "--system-only" ]; then
  exit 0
fi

requirement="${1:-hud}"

python="$(command -v python3)"
if ! "$python" -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))'; then
  uv python install "$python_version"
  python="$root/bin/python$python_version"
fi

uv venv "$root/venv" --python "$python"
uv pip install --python "$root/venv/bin/python" "$requirement"
