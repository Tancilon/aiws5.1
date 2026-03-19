#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/aiws_env_bundle_lib.sh"

REPO_ID="${1:-${AIWS_HF_WEIGHTS_REPO:-}}"
REVISION="${AIWS_HF_WEIGHTS_REVISION:-main}"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aiws-hf-download.XXXXXX")"
HF_BIN="${HF_BIN:-hf}"
HF_PYTHON="${HF_PYTHON:-}"
DOWNLOAD_BACKEND="${AIWS_HF_DOWNLOAD_BACKEND:-hf-cli}"
LEGACY_VENV_DIR="${AIWS_HF_LEGACY_VENV_DIR:-/tmp/aiws-hf-download-venv}"
LEGACY_BOOTSTRAP_PYTHON="${AIWS_HF_LEGACY_BOOTSTRAP_PYTHON:-python3}"
LEGACY_PYTHON_BIN="${LEGACY_VENV_DIR}/bin/python"
LEGACY_PIP_BIN="${LEGACY_VENV_DIR}/bin/pip"
DOWNLOAD_PATHS=()

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

ensure_legacy_venv() {
  if [[ -x "${LEGACY_PYTHON_BIN}" ]]; then
    return 0
  fi

  if ! command -v "${LEGACY_BOOTSTRAP_PYTHON}" >/dev/null 2>&1; then
    echo "legacy bootstrap python not found: ${LEGACY_BOOTSTRAP_PYTHON}" >&2
    exit 1
  fi

  "${LEGACY_BOOTSTRAP_PYTHON}" -m venv "${LEGACY_VENV_DIR}"
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
}

collect_download_paths() {
  local rel_path
  local entry_type
  local rel_dst

  DOWNLOAD_PATHS=()

  while IFS= read -r rel_path; do
    [[ -z "${rel_path}" ]] && continue
    DOWNLOAD_PATHS+=("${rel_path}")
  done < <(aiws_project_weight_paths)

  while IFS='|' read -r entry_type _src rel_dst _restore_name; do
    [[ -z "${entry_type}" ]] && continue
    case "${entry_type}" in
      file|dir|archive_dir)
        DOWNLOAD_PATHS+=("${rel_dst}")
        ;;
      *)
        echo "unknown manifest entry type: ${entry_type}" >&2
        exit 1
        ;;
    esac
  done < <(aiws_torch_cache_entries)
}

trap 'rm -rf "${STAGE_DIR}"' EXIT

if [[ -z "${REPO_ID}" ]]; then
  echo "usage: bash scripts/download_weights.sh <hf_repo_id>" >&2
  echo "or set AIWS_HF_WEIGHTS_REPO=username/repo" >&2
  exit 1
fi

case "${DOWNLOAD_BACKEND}" in
  hf-cli)
    if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
      echo "hf CLI not found. Install it first, or set AIWS_HF_DOWNLOAD_BACKEND=legacy-python." >&2
      exit 1
    fi
    ;;
  legacy-python)
    ensure_legacy_venv
    ;;
  *)
    echo "unsupported AIWS_HF_DOWNLOAD_BACKEND: ${DOWNLOAD_BACKEND}" >&2
    echo "supported values: hf-cli, legacy-python" >&2
    exit 1
    ;;
esac

collect_download_paths

case "${DOWNLOAD_BACKEND}" in
  hf-cli)
    run_hf download "${REPO_ID}" \
      "${DOWNLOAD_PATHS[@]}" \
      --repo-type model \
      --revision "${REVISION}" \
      --local-dir "${STAGE_DIR}" >/dev/null
    ;;
  legacy-python)
    env \
      HTTP_PROXY="${HTTP_PROXY:-}" \
      HTTPS_PROXY="${HTTPS_PROXY:-}" \
      NO_PROXY="${NO_PROXY:-}" \
      HF_TOKEN="${HF_TOKEN:-}" \
      "${LEGACY_PYTHON_BIN}" - "${REPO_ID}" "${REVISION}" "${STAGE_DIR}" "${DOWNLOAD_PATHS[@]}" >/dev/null <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

try:
    from huggingface_hub import HfFolder
except Exception:
    HfFolder = None


def resolve_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    if HfFolder is not None:
        token = HfFolder.get_token()
        if token:
            return token

    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token

    return None


def main():
    if len(sys.argv) < 5:
        raise SystemExit("usage: python - <repo_id> <revision> <local_dir> <remote_path>...")

    repo_id = sys.argv[1]
    revision = sys.argv[2]
    local_dir = Path(sys.argv[3]).resolve()
    remote_paths = sys.argv[4:]
    local_dir.mkdir(parents=True, exist_ok=True)

    token = resolve_token()
    for remote_path in remote_paths:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=remote_path,
            revision=revision,
            local_dir=str(local_dir),
            token=token,
        )


if __name__ == "__main__":
    main()
PY
    ;;
esac

if [[ -f "${STAGE_DIR}/SHA256SUMS" ]]; then
  (
    cd "${STAGE_DIR}"
    sha256sum -c SHA256SUMS
  )
fi

aiws_restore_weights_from_stage "${STAGE_DIR}" "${PROJECT_ROOT}"

echo "Weights restored into project and torch cache."
echo
echo "Project weights:"
aiws_project_weight_paths
echo
echo "Torch cache:"
aiws_torch_cache_entries
