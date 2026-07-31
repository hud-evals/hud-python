#!/bin/sh
set -eu

requirement="${1:-hud}"
root=/media/hud
python_version=3.12

export UV_PYTHON_INSTALL_DIR="$root/python"
export UV_PYTHON_BIN_DIR="$root/bin"
export UV_NO_CACHE=1
export XDG_CONFIG_HOME="$root/config"
export PATH="$root/bin:$PATH"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq bubblewrap python3 python3-venv python3-pip git curl ca-certificates
  rm -rf /var/lib/apt/lists/*
elif command -v apk >/dev/null 2>&1; then
  apk add --no-cache bubblewrap python3 py3-pip git curl ca-certificates
else
  echo "hud: Harbor environments require an apt- or apk-based image" >&2
  exit 1
fi

uv python install "$python_version"
uv venv "$root/venv" --python "$python_version"
uv pip install --python "$root/venv/bin/python" "$requirement"
