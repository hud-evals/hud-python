#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  SUDO=""
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "This script requires root privileges. Install sudo or re-run as root." >&2
    exit 1
  fi
  SUDO="sudo"
fi

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd curl
need_cmd apt-get
need_cmd gpg
need_cmd tee
need_cmd install

APT_KEYRING="/usr/share/keyrings/cloud.google.gpg"
APT_LIST="/etc/apt/sources.list.d/google-cloud-sdk.list"
APT_SOURCES="/etc/apt/sources.list.d/google-cloud-sdk.sources"

SOURCES_CONTENT=$'Types: deb\nURIs: https://packages.cloud.google.com/apt\nSuites: cloud-sdk\nComponents: main\nSigned-By: /usr/share/keyrings/cloud.google.gpg\n'

echo "Installing Google Cloud CLI via apt repository..."

${SUDO} apt-get update
${SUDO} apt-get install -y apt-transport-https ca-certificates gnupg curl

TMP_ASC=$(mktemp)
TMP_GPG=$(mktemp)
trap 'rm -f "${TMP_ASC}" "${TMP_GPG}"' EXIT

curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg >"${TMP_ASC}"
gpg --dearmor --yes --output "${TMP_GPG}" "${TMP_ASC}"
${SUDO} install -D -m 644 "${TMP_GPG}" "${APT_KEYRING}"

${SUDO} rm -f "${APT_LIST}"
echo "${SOURCES_CONTENT}" | ${SUDO} tee "${APT_SOURCES}" >/dev/null

${SUDO} apt-get update
${SUDO} apt-get install -y google-cloud-cli

echo
if command -v gcloud >/dev/null 2>&1; then
  gcloud --version
  echo
  echo "Run 'gcloud init' next to authenticate and configure the CLI."
else
  echo "gcloud command not found after installation." >&2
  exit 1
fi
