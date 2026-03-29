#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HF_BIN="${HF_BIN:-hf}"
HF_PYTHON="${HF_PYTHON:-}"
REPO_ID="${1:-${AIWS_HF_ASSETS_REPO:-}}"
REPO_TYPE="${AIWS_HF_ASSETS_REPO_TYPE:-dataset}"
REVISION="${AIWS_HF_ASSETS_REVISION:-main}"
PRIVATE="${AIWS_HF_ASSETS_PRIVATE:-1}"
NUM_WORKERS="${AIWS_HF_ASSETS_NUM_WORKERS:-4}"
UPLOAD_ROOT="${AIWS_HF_ASSETS_UPLOAD_ROOT:-${REPO_ROOT}}"
INCLUDE_PATTERN="${AIWS_HF_ASSETS_INCLUDE:-assets/**}"
NO_BARS="${AIWS_HF_ASSETS_NO_BARS:-1}"
NO_REPORT="${AIWS_HF_ASSETS_NO_REPORT:-0}"

log() {
  echo "[upload_assets_to_hf] $*"
}

bool_is_true() {
  case "${1}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_hf_python() {
  if [[ -n "${HF_PYTHON}" ]]; then
    printf '%s\n' "${HF_PYTHON}"
    return 0
  fi

  local hf_path
  local first_line
  hf_path="$(command -v "${HF_BIN}")"
  first_line="$(head -n 1 "${hf_path}")"
  if [[ "${first_line}" == '#!'* ]]; then
    printf '%s\n' "${first_line#\#!}"
    return 0
  fi

  return 1
}

run_hf() {
  env \
    -u ALL_PROXY \
    -u all_proxy \
    "${HF_BIN}" "$@"
}

run_hf_python() {
  local hf_python="$1"
  shift

  env \
    -u ALL_PROXY \
    -u all_proxy \
    "${hf_python}" "$@"
}

resolve_default_repo_id() {
  local username
  username="$(
    run_hf auth whoami 2>/dev/null | awk '/^user:/ { print $2; exit }'
  )"
  if [[ -z "${username}" ]]; then
    echo "failed to resolve Hugging Face username; pass <hf_repo_id> or set AIWS_HF_ASSETS_REPO" >&2
    exit 1
  fi
  printf '%s\n' "${username}/aiws5.1-assets"
}

if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  echo "hf CLI not found: ${HF_BIN}" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/assets" ]]; then
  echo "assets directory not found: ${REPO_ROOT}/assets" >&2
  exit 1
fi

if [[ ! -d "${UPLOAD_ROOT}" ]]; then
  echo "upload root not found: ${UPLOAD_ROOT}" >&2
  exit 1
fi

if [[ -z "${REPO_ID}" ]]; then
  REPO_ID="$(resolve_default_repo_id)"
fi

if bool_is_true "${PRIVATE}"; then
  PRIVATE_FLAG="--private"
else
  PRIVATE_FLAG="--no-private"
fi

UPLOAD_ARGS=(
  "${SCRIPT_DIR}/hf_upload_large_folder_no_create.py"
  "${REPO_ID}" "${UPLOAD_ROOT}"
  --repo-type "${REPO_TYPE}"
  --revision "${REVISION}"
  --allow-pattern "${INCLUDE_PATTERN}"
  --num-workers "${NUM_WORKERS}"
)

if bool_is_true "${NO_REPORT}"; then
  UPLOAD_ARGS+=(--no-report)
fi

if bool_is_true "${PRIVATE}"; then
  UPLOAD_ARGS+=(--private)
fi

log "Ensuring ${REPO_TYPE} repo exists: ${REPO_ID}"
run_hf repos create "${REPO_ID}" \
  --type "${REPO_TYPE}" \
  "${PRIVATE_FLAG}" \
  --exist-ok

HF_PYTHON_BIN="$(resolve_hf_python || true)"
if [[ -z "${HF_PYTHON_BIN}" ]]; then
  echo "failed to resolve python interpreter for ${HF_BIN}" >&2
  exit 1
fi

log "Uploading ${INCLUDE_PATTERN} from ${UPLOAD_ROOT} to ${REPO_TYPE} repo ${REPO_ID}@${REVISION}"
run_hf_python "${HF_PYTHON_BIN}" "${UPLOAD_ARGS[@]}"

log "Upload finished: https://huggingface.co/${REPO_TYPE}s/${REPO_ID}/tree/${REVISION}"
