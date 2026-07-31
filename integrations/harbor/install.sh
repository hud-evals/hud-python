#!/bin/sh
set -eu

requirement="${1:-hud}"
root=/media/hud
python_version=3.12

export UV_INSTALL_DIR="$root/bin"
export UV_PYTHON_INSTALL_DIR="$root/python"
export UV_PYTHON_BIN_DIR="$root/bin"
export UV_NO_CACHE=1
export XDG_CONFIG_HOME="$root/config"
export PATH="$root/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    { apt-get update -qq && apt-get install -y -qq curl ca-certificates; } \
      || apk add --no-cache curl ca-certificates
  fi
  { command -v curl >/dev/null 2>&1 && curl -LsSf https://astral.sh/uv/install.sh | sh; } \
    || { command -v wget >/dev/null 2>&1 && wget -qO- https://astral.sh/uv/install.sh | sh; } \
    || pip install -q -U uv
fi

command -v bwrap >/dev/null 2>&1 \
  || { apt-get update -qq && apt-get install -y -qq bubblewrap; } \
  || apk add --no-cache bubblewrap

uv python install "$python_version"
uv venv "$root/venv" --python "$python_version"
uv pip install --python "$root/venv/bin/python" "$requirement"

apt_packages=""
apk_packages=""
for spec in \
  "python3|python3 python3-venv|python3" \
  "pip3|python3-pip|py3-pip" \
  "git|git|git" \
  "curl|curl ca-certificates|curl ca-certificates"
do
  command="${spec%%|*}"
  if command -v "$command" >/dev/null 2>&1; then
    continue
  fi
  rest="${spec#*|}"
  apt_packages="$apt_packages ${rest%%|*}"
  apk_packages="$apk_packages ${rest##*|}"
done

if [ -n "$apt_packages" ]; then
  { apt-get update -qq && apt-get install -y -qq $apt_packages; } \
    || apk add --no-cache $apk_packages
fi
