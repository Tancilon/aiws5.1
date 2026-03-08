#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID="${1:-${AIWS_HF_WEIGHTS_REPO:-}}"
BUNDLE_DIR="${2:-}"
REVISION="${AIWS_HF_WEIGHTS_REVISION:-main}"
PRIVATE_FLAG="${AIWS_HF_WEIGHTS_PRIVATE:-true}"
UPLOAD_MODE="${AIWS_HF_UPLOAD_MODE:-upload}"
HF_BIN="${HF_BIN:-hf}"
HF_PYTHON="${HF_PYTHON:-}"
UPLOAD_BACKEND="${AIWS_HF_UPLOAD_BACKEND:-hf-cli}"
LEGACY_VENV_DIR="${AIWS_HF_LEGACY_VENV_DIR:-/tmp/aiws-hf-upload-venv}"
LEGACY_PYTHON_BIN="${LEGACY_VENV_DIR}/bin/python"
LEGACY_PIP_BIN="${LEGACY_VENV_DIR}/bin/pip"

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
  local hf_python
  hf_python="$(resolve_hf_python || true)"

  if [[ -n "${ALL_PROXY:-}${all_proxy:-}" ]] && [[ -n "${hf_python}" ]] && "${hf_python}" -c "import socksio" >/dev/null 2>&1; then
    env \
      -u HTTP_PROXY \
      -u HTTPS_PROXY \
      -u http_proxy \
      -u https_proxy \
      "${HF_BIN}" "$@"
    return
  fi

  env \
    -u ALL_PROXY \
    -u all_proxy \
    "${HF_BIN}" "$@"
}

if [[ -z "${REPO_ID}" ]]; then
  echo "usage: bash scripts/upload_weights_to_hf.sh <hf_repo_id> [bundle_dir]" >&2
  echo "or set AIWS_HF_WEIGHTS_REPO=username/repo" >&2
  exit 1
fi

if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  echo "hf CLI not found. Install it first, e.g. 'brew install huggingface-cli' or the official installer." >&2
  exit 1
fi

if [[ -z "${BUNDLE_DIR}" ]]; then
  BUNDLE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aiws-hf-weights.XXXXXX")"
  trap 'rm -rf "${BUNDLE_DIR}"' EXIT
fi

bash "${SCRIPT_DIR}/prepare_hf_weights_bundle.sh" "${BUNDLE_DIR}" >/dev/null

case "${UPLOAD_BACKEND}" in
  hf-cli)
    repo_create_cmd=(repos create "${REPO_ID}" --repo-type model --exist-ok)
    if [[ "${PRIVATE_FLAG}" == "true" ]]; then
      repo_create_cmd+=(--private)
    fi
    run_hf "${repo_create_cmd[@]}"

    case "${UPLOAD_MODE}" in
      upload)
        run_hf upload "${REPO_ID}" "${BUNDLE_DIR}" . \
          --repo-type model \
          --revision "${REVISION}" \
          --commit-message "Upload AIWS runtime weights"
        ;;
      upload-large-folder)
        run_hf upload-large-folder "${REPO_ID}" "${BUNDLE_DIR}" \
          --repo-type model \
          --revision "${REVISION}"
        ;;
      *)
        echo "unsupported AIWS_HF_UPLOAD_MODE: ${UPLOAD_MODE}" >&2
        echo "supported values: upload, upload-large-folder" >&2
        exit 1
        ;;
    esac
    ;;
  legacy-python)
    if [[ ! -x "${LEGACY_PYTHON_BIN}" ]]; then
      conda run -n aiws python -m venv "${LEGACY_VENV_DIR}"
      env \
        HTTP_PROXY="${HTTP_PROXY:-}" \
        HTTPS_PROXY="${HTTPS_PROXY:-}" \
        NO_PROXY="${NO_PROXY:-}" \
        "${LEGACY_PIP_BIN}" install --upgrade pip setuptools wheel >/dev/null
      env \
        HTTP_PROXY="${HTTP_PROXY:-}" \
        HTTPS_PROXY="${HTTPS_PROXY:-}" \
        NO_PROXY="${NO_PROXY:-}" \
        "${LEGACY_PIP_BIN}" install "huggingface_hub<1" requests >/dev/null
    fi

    env \
      HTTP_PROXY="${HTTP_PROXY:-}" \
      HTTPS_PROXY="${HTTPS_PROXY:-}" \
      NO_PROXY="${NO_PROXY:-}" \
      HF_TOKEN="${HF_TOKEN:-}" \
      "${LEGACY_PYTHON_BIN}" "${SCRIPT_DIR}/upload_weights_via_hub.py" \
      "${REPO_ID}" "${BUNDLE_DIR}" "${REVISION}" "${PRIVATE_FLAG}"
    ;;
  *)
    echo "unsupported AIWS_HF_UPLOAD_BACKEND: ${UPLOAD_BACKEND}" >&2
    echo "supported values: hf-cli, legacy-python" >&2
    exit 1
    ;;
esac

echo
echo "Upload finished:"
echo "https://huggingface.co/${REPO_ID}"
echo
echo "Recommended download command on a new machine:"
echo "AIWS_HF_WEIGHTS_REPO=${REPO_ID} bash scripts/download_weights.sh"
