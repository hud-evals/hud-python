#!/usr/bin/env bash
set -euo pipefail

# Usage: ./gcloud_sheetbench_setup.sh <PROJECT_ID> [CLIENT_SECRET_JSON]
# Creates the project if absent, enables Drive/Sheets APIs,
# optionally ingests a Desktop OAuth client JSON, exports env vars,
# runs gcloud application-default login, and executes setup_google_auth.py.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <PROJECT_ID> [CLIENT_SECRET_JSON]" >&2
  exit 1
fi

PROJECT_ID=$1
CLIENT_SECRET_JSON=${2:-}
PROJECT_NAME="SheetBench ${PROJECT_ID}"
ENV_EXPORT_FILE=".sheetbench_env"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd gcloud
need_cmd python3

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

log "Ensuring project ${PROJECT_ID} exists or creating it."
if ! gcloud projects describe "${PROJECT_ID}" >/dev/null 2>&1; then
  log "Project not found. Attempting to create ${PROJECT_ID}."
  if ! gcloud projects create "${PROJECT_ID}" --name="${PROJECT_NAME}" >/dev/null; then
    cat <<'TEXT'
❌ Unable to create project automatically.
   • Accept Google Cloud Terms at https://console.cloud.google.com if prompted.
   • Ensure you have remaining project quota.
   • If your account requires an organization/folder, create the project manually
     and rerun this script with the project ID and client secret JSON.
TEXT
    exit 1
  fi
else
  log "Project already exists and is accessible."
fi

log "Setting active project to ${PROJECT_ID}."
if ! gcloud config set project "${PROJECT_ID}" >/dev/null; then
  echo "❌ Failed to set active project. Check your permissions." >&2
  exit 1
fi

log "Enabling Google Drive and Sheets APIs."
if ! gcloud services enable drive.googleapis.com sheets.googleapis.com >/dev/null; then
  echo "❌ Failed to enable required APIs. Ensure you have Service Usage permissions." >&2
  exit 1
fi

log "Confirming APIs are enabled."
gcloud services list --enabled --filter="NAME:drive.googleapis.com OR NAME:sheets.googleapis.com"

if [[ -z ${CLIENT_SECRET_JSON} && ( -z ${SHEETBENCH_CLIENT_ID:-} || -z ${SHEETBENCH_CLIENT_SECRET:-} ) ]]; then
  cat <<'TEXT'
⚠️  No OAuth client provided.
   Download your Desktop OAuth client JSON from Google Cloud Console → APIs & Services → Credentials,
   then rerun this script as: ./gcloud_sheetbench_setup.sh <PROJECT_ID> /path/to/client_secret.json
   or export SHEETBENCH_CLIENT_ID and SHEETBENCH_CLIENT_SECRET before rerunning.
TEXT
  exit 0
fi

if [[ -n ${CLIENT_SECRET_JSON} ]]; then
  if [[ ! -f ${CLIENT_SECRET_JSON} ]]; then
    echo "❌ Client secret JSON not found: ${CLIENT_SECRET_JSON}" >&2
    exit 1
  fi
  log "Extracting client ID and secret from ${CLIENT_SECRET_JSON}."
  read -r SHEETBENCH_CLIENT_ID SHEETBENCH_CLIENT_SECRET < <(
    python3 - <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as fh:
    data = json.load(fh)
try:
    installed = data["installed"]
except KeyError as exc:
    raise SystemExit("Provided JSON is not a Desktop OAuth client (missing 'installed').") from exc
print(installed["client_id"], installed["client_secret"])
PY
    "${CLIENT_SECRET_JSON}"
  )
  export SHEETBENCH_CLIENT_ID SHEETBENCH_CLIENT_SECRET
fi

log "Writing environment variables to ${ENV_EXPORT_FILE}."
cat <<EOF > "${ENV_EXPORT_FILE}"
export SHEETBENCH_CLIENT_ID="${SHEETBENCH_CLIENT_ID}"
export SHEETBENCH_CLIENT_SECRET="${SHEETBENCH_CLIENT_SECRET}"
EOF
chmod 600 "${ENV_EXPORT_FILE}" || true

echo "Saved exports to ${ENV_EXPORT_FILE}."
echo "SHEETBENCH_CLIENT_ID=$(printf '%s' "${SHEETBENCH_CLIENT_ID}" | sed 's/\(..............\).*/\1.../')"
echo "SHEETBENCH_CLIENT_SECRET=$(printf '%s' "${SHEETBENCH_CLIENT_SECRET}" | sed 's/\(.....\).*/\1.../')"

if [[ -n ${CLIENT_SECRET_JSON} ]]; then
  log "Authenticating Application Default Credentials with your client."
  if ! gcloud auth application-default login \
    --client-id-file="${CLIENT_SECRET_JSON}" \
    --scopes="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive.file,https://www.googleapis.com/auth/spreadsheets"; then
    echo "❌ gcloud auth application-default login failed." >&2
    exit 1
  fi
else
  log "Skipping gcloud application-default login (no client secret JSON provided)."
fi

log "Running setup_google_auth.py to produce google_credentials.json."
python3 setup_google_auth.py

log "All steps complete. Run 'python test_google_sheets_direct.py' to verify access."
