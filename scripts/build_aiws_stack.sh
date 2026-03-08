#!/usr/bin/env bash
set -euo pipefail

# One-shot bootstrap for a new machine:
# 1. create or update the host-side aiws orchestrator conda environment
# 2. download runtime weights from Hugging Face
# 3. build the three algorithm Docker images
# Optional overrides:
#   AIWS_CONDA_ENV_NAME
#   AIWS_CONDA_FILE
#   AIWS_DOWNLOAD_WEIGHTS
#   AIWS_HF_WEIGHTS_REPO
#   AIWS_HF_WEIGHTS_REVISION
#   AIWS_HF_DOWNLOAD_BACKEND
#   AIWS_YOLO_IMAGE
#   AIWS_GENPOSE2_IMAGE
#   AIWS_FOUNDATIONPOSE_IMAGE
#   HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / NO_PROXY

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

AIWS_CONDA_ENV_NAME=${AIWS_CONDA_ENV_NAME:-aiws}
AIWS_CONDA_FILE=${AIWS_CONDA_FILE:-${REPO_ROOT}/environment-aiws.yml}
AIWS_DOWNLOAD_WEIGHTS=${AIWS_DOWNLOAD_WEIGHTS:-auto}
AIWS_HF_WEIGHTS_REPO=${AIWS_HF_WEIGHTS_REPO:-tancilon/aiws5.1-weights}
AIWS_HF_WEIGHTS_REVISION=${AIWS_HF_WEIGHTS_REVISION:-main}
AIWS_HF_DOWNLOAD_BACKEND=${AIWS_HF_DOWNLOAD_BACKEND:-legacy-python}

AIWS_YOLO_IMAGE=${AIWS_YOLO_IMAGE:-yolov11-seg:infer}
AIWS_GENPOSE2_IMAGE=${AIWS_GENPOSE2_IMAGE:-genpose2-env:test}
AIWS_FOUNDATIONPOSE_IMAGE=${AIWS_FOUNDATIONPOSE_IMAGE:-foundationpose-env:test}

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-0}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[build_aiws_stack] Missing command: ${cmd}" >&2
    exit 1
  fi
}

log() {
  echo "[build_aiws_stack] $*"
}

require_cmd conda
require_cmd docker

if [[ -f "${REPO_ROOT}/scripts/weights_manifest.sh" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/scripts/weights_manifest.sh"
fi

weights_ready() {
  local rel_path
  local entry_type
  local rel_dst
  local cache_path

  while IFS= read -r rel_path; do
    [[ -z "${rel_path}" ]] && continue
    [[ -f "${REPO_ROOT}/${rel_path}" ]] || return 1
  done < <(aiws_project_weight_paths)

  while IFS='|' read -r entry_type _src rel_dst restore_name; do
    [[ -z "${entry_type}" ]] && continue
    case "${entry_type}" in
      file)
        cache_path="${HOME}/.cache/torch/hub/checkpoints/$(basename "${rel_dst}")"
        [[ -f "${cache_path}" ]] || return 1
        ;;
      dir)
        cache_path="${HOME}/.cache/torch/hub/$(basename "${rel_dst}")"
        [[ -d "${cache_path}" ]] || return 1
        ;;
      archive_dir)
        cache_path="${HOME}/.cache/torch/hub/${restore_name}"
        [[ -d "${cache_path}" ]] || return 1
        ;;
      *)
        return 1
        ;;
    esac
  done < <(aiws_torch_cache_entries)

  return 0
}

should_download_weights() {
  case "${AIWS_DOWNLOAD_WEIGHTS}" in
    1|true|TRUE|yes|YES|always)
      return 0
      ;;
    0|false|FALSE|no|NO|never)
      return 1
      ;;
    auto)
      if weights_ready; then
        return 1
      fi
      return 0
      ;;
    *)
      echo "[build_aiws_stack] Unsupported AIWS_DOWNLOAD_WEIGHTS=${AIWS_DOWNLOAD_WEIGHTS}" >&2
      echo "[build_aiws_stack] Supported values: auto, always, 1, true, 0, false" >&2
      exit 1
      ;;
  esac
}

if [[ ! -f "${AIWS_CONDA_FILE}" ]]; then
  echo "[build_aiws_stack] Missing conda env file: ${AIWS_CONDA_FILE}" >&2
  exit 1
fi

log "Repo root: ${REPO_ROOT}"
log "Conda env: ${AIWS_CONDA_ENV_NAME}"
log "Conda file: ${AIWS_CONDA_FILE}"
log "Weights mode: ${AIWS_DOWNLOAD_WEIGHTS}"
log "Weights repo: ${AIWS_HF_WEIGHTS_REPO}"
log "Weights revision: ${AIWS_HF_WEIGHTS_REVISION}"
log "Weights backend: ${AIWS_HF_DOWNLOAD_BACKEND}"
log "YOLO image: ${AIWS_YOLO_IMAGE}"
log "GenPose2 image: ${AIWS_GENPOSE2_IMAGE}"
log "FoundationPose image: ${AIWS_FOUNDATIONPOSE_IMAGE}"

if conda env list | awk '{print $1}' | grep -qx "${AIWS_CONDA_ENV_NAME}"; then
  log "Updating existing conda env ${AIWS_CONDA_ENV_NAME}"
  conda env update -n "${AIWS_CONDA_ENV_NAME}" -f "${AIWS_CONDA_FILE}" --prune
else
  log "Creating conda env ${AIWS_CONDA_ENV_NAME}"
  conda env create -f "${AIWS_CONDA_FILE}"
fi

if should_download_weights; then
  log "Downloading runtime weights from Hugging Face"
  AIWS_CONDA_PYTHON="$(conda run -n "${AIWS_CONDA_ENV_NAME}" python -c 'import sys; print(sys.executable)' | tail -n 1 | tr -d '\r')"
  if [[ -z "${AIWS_CONDA_PYTHON}" ]]; then
    echo "[build_aiws_stack] Failed to resolve python from conda env ${AIWS_CONDA_ENV_NAME}" >&2
    exit 1
  fi

  env \
    AIWS_HF_WEIGHTS_REPO="${AIWS_HF_WEIGHTS_REPO}" \
    AIWS_HF_WEIGHTS_REVISION="${AIWS_HF_WEIGHTS_REVISION}" \
    AIWS_HF_DOWNLOAD_BACKEND="${AIWS_HF_DOWNLOAD_BACKEND}" \
    AIWS_HF_LEGACY_BOOTSTRAP_PYTHON="${AIWS_CONDA_PYTHON}" \
    bash "${REPO_ROOT}/scripts/download_weights.sh"
else
  log "Skipping weight download"
fi

log "Building YOLOv11 image"
bash "${REPO_ROOT}/yolov11-seg-aiws/docker/build_with_proxy.sh" "${AIWS_YOLO_IMAGE}"

log "Building GenPose2 image"
bash "${REPO_ROOT}/GenPose2/docker/build_with_proxy.sh" "${AIWS_GENPOSE2_IMAGE}"

log "Building FoundationPose image"
bash "${REPO_ROOT}/aiws_alignment-feat-model-free/docker/build_with_proxy.sh" "${AIWS_FOUNDATIONPOSE_IMAGE}"

cat <<EOF
[build_aiws_stack] Done.

Optional runtime exports:
  export AIWS_HF_WEIGHTS_REPO=${AIWS_HF_WEIGHTS_REPO}
  export AIWS_HF_WEIGHTS_REVISION=${AIWS_HF_WEIGHTS_REVISION}
  export AIWS_YOLO_IMAGE=${AIWS_YOLO_IMAGE}
  export AIWS_GENPOSE2_IMAGE=${AIWS_GENPOSE2_IMAGE}
  export AIWS_FOUNDATIONPOSE_IMAGE=${AIWS_FOUNDATIONPOSE_IMAGE}

Run:
  bash scripts/run_aiws.sh aiws_sub "data/AIWS/F120/1_color.png" "data/AIWS/F120/1_depth.exr" 0
EOF
