#!/usr/bin/env bash
set -euo pipefail

# One-shot bootstrap for a new machine:
# 1. create or update the host-side aiws orchestrator conda environment
# 2. restore runtime weights from a local offline bundle or download them
# 3. load prebuilt Docker images from a local offline bundle or build them
#
# Optional overrides:
#   AIWS_CONDA_ENV_NAME
#   AIWS_CONDA_FILE
#   AIWS_SKIP_CONDA_SETUP
#   AIWS_DOWNLOAD_WEIGHTS
#   AIWS_HF_WEIGHTS_REPO
#   AIWS_HF_WEIGHTS_REVISION
#   AIWS_HF_DOWNLOAD_BACKEND
#   AIWS_ENV_BUNDLE_DIR
#   AIWS_ENV_BUNDLE_MODE         (auto | prefer | require | off)
#   AIWS_YOLO_IMAGE
#   AIWS_GENPOSE2_IMAGE
#   AIWS_FOUNDATIONPOSE_IMAGE
#   HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / NO_PROXY

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/aiws_env_bundle_lib.sh"

AIWS_CONDA_ENV_NAME=${AIWS_CONDA_ENV_NAME:-aiws}
AIWS_CONDA_FILE=${AIWS_CONDA_FILE:-${REPO_ROOT}/environment-aiws.yml}
AIWS_SKIP_CONDA_SETUP=${AIWS_SKIP_CONDA_SETUP:-0}
AIWS_DOWNLOAD_WEIGHTS=${AIWS_DOWNLOAD_WEIGHTS:-auto}
AIWS_HF_WEIGHTS_REPO=${AIWS_HF_WEIGHTS_REPO:-tancilon/aiws5.1-weights}
AIWS_HF_WEIGHTS_REVISION=${AIWS_HF_WEIGHTS_REVISION:-main}
AIWS_HF_DOWNLOAD_BACKEND=${AIWS_HF_DOWNLOAD_BACKEND:-legacy-python}
AIWS_ENV_BUNDLE_DIR=${AIWS_ENV_BUNDLE_DIR:-${REPO_ROOT}/assets/env_files}
AIWS_ENV_BUNDLE_MODE=${AIWS_ENV_BUNDLE_MODE:-auto}

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

bundle_mode_uses_local_assets() {
  case "${AIWS_ENV_BUNDLE_MODE}" in
    auto|prefer|require)
      return 0
      ;;
    off|0|false|FALSE|no|NO)
      return 1
      ;;
    *)
      echo "[build_aiws_stack] Unsupported AIWS_ENV_BUNDLE_MODE=${AIWS_ENV_BUNDLE_MODE}" >&2
      echo "[build_aiws_stack] Supported values: auto, prefer, require, off" >&2
      exit 1
      ;;
  esac
}

bundle_mode_requires_local_assets() {
  [[ "${AIWS_ENV_BUNDLE_MODE}" == "require" ]]
}

bool_is_true() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

weights_ready() {
  local rel_path
  local entry_type
  local rel_dst
  local restore_name
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

offline_weights_bundle_ready() {
  aiws_weights_stage_ready "${AIWS_ENV_BUNDLE_DIR}/weights"
}

offline_image_tar_path() {
  aiws_bundle_image_tar_path "${AIWS_ENV_BUNDLE_DIR}" "$1"
}

offline_image_bundle_ready() {
  local tar_path
  tar_path="$(offline_image_tar_path "$1")"
  [[ -f "${tar_path}" ]]
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

setup_conda_env() {
  local conda_args=()

  if bool_is_true "${AIWS_SKIP_CONDA_SETUP}"; then
    log "Skipping conda env setup"
    return 0
  fi

  require_cmd conda

  if [[ ! -f "${AIWS_CONDA_FILE}" ]]; then
    echo "[build_aiws_stack] Missing conda env file: ${AIWS_CONDA_FILE}" >&2
    exit 1
  fi

  if bundle_mode_requires_local_assets; then
    conda_args+=(--offline)
  fi

  if conda env list | awk '{print $1}' | grep -qx "${AIWS_CONDA_ENV_NAME}"; then
    log "Updating existing conda env ${AIWS_CONDA_ENV_NAME}"
    conda env update -n "${AIWS_CONDA_ENV_NAME}" -f "${AIWS_CONDA_FILE}" --prune "${conda_args[@]}"
  else
    log "Creating conda env ${AIWS_CONDA_ENV_NAME}"
    conda env create -f "${AIWS_CONDA_FILE}" "${conda_args[@]}"
  fi
}

restore_weights_from_bundle() {
  log "Restoring runtime weights from offline bundle: ${AIWS_ENV_BUNDLE_DIR}/weights"
  aiws_restore_weights_from_stage "${AIWS_ENV_BUNDLE_DIR}/weights" "${REPO_ROOT}"
}

download_weights_from_hf() {
  local aiws_conda_python

  log "Downloading runtime weights from Hugging Face"
  aiws_conda_python="$(conda run -n "${AIWS_CONDA_ENV_NAME}" python -c 'import sys; print(sys.executable)' | tail -n 1 | tr -d '\r')"
  if [[ -z "${aiws_conda_python}" ]]; then
    echo "[build_aiws_stack] Failed to resolve python from conda env ${AIWS_CONDA_ENV_NAME}" >&2
    exit 1
  fi

  env \
    AIWS_HF_WEIGHTS_REPO="${AIWS_HF_WEIGHTS_REPO}" \
    AIWS_HF_WEIGHTS_REVISION="${AIWS_HF_WEIGHTS_REVISION}" \
    AIWS_HF_DOWNLOAD_BACKEND="${AIWS_HF_DOWNLOAD_BACKEND}" \
    AIWS_HF_LEGACY_BOOTSTRAP_PYTHON="${aiws_conda_python}" \
    bash "${REPO_ROOT}/scripts/download_weights.sh"
}

ensure_weights() {
  if weights_ready; then
    log "Runtime weights already present"
    return 0
  fi

  if bundle_mode_uses_local_assets && offline_weights_bundle_ready; then
    restore_weights_from_bundle
    return 0
  fi

  if bundle_mode_requires_local_assets; then
    echo "[build_aiws_stack] Offline mode requires weights bundle under ${AIWS_ENV_BUNDLE_DIR}/weights" >&2
    exit 1
  fi

  if should_download_weights; then
    download_weights_from_hf
  else
    log "Skipping weight download"
  fi
}

load_image_from_bundle() {
  local label="$1"
  local image_name="$2"
  local tar_path
  tar_path="$(offline_image_tar_path "${image_name}")"
  log "Loading ${label} image from offline bundle: ${tar_path}"
  docker load -i "${tar_path}" >/dev/null
}

build_image_online() {
  local label="$1"
  local image_name="$2"
  local build_script="$3"
  log "Building ${label} image"
  bash "${build_script}" "${image_name}"
}

ensure_image() {
  local label="$1"
  local image_name="$2"
  local build_script="$3"

  if bundle_mode_uses_local_assets && offline_image_bundle_ready "${image_name}"; then
    load_image_from_bundle "${label}" "${image_name}"
    return 0
  fi

  if bundle_mode_requires_local_assets; then
    echo "[build_aiws_stack] Offline mode requires image bundle for ${image_name} under ${AIWS_ENV_BUNDLE_DIR}/docker-images" >&2
    exit 1
  fi

  build_image_online "${label}" "${image_name}" "${build_script}"
}

require_cmd docker

log "Repo root: ${REPO_ROOT}"
log "Conda env: ${AIWS_CONDA_ENV_NAME}"
log "Conda file: ${AIWS_CONDA_FILE}"
log "Skip conda setup: ${AIWS_SKIP_CONDA_SETUP}"
log "Weights mode: ${AIWS_DOWNLOAD_WEIGHTS}"
log "Weights repo: ${AIWS_HF_WEIGHTS_REPO}"
log "Weights revision: ${AIWS_HF_WEIGHTS_REVISION}"
log "Weights backend: ${AIWS_HF_DOWNLOAD_BACKEND}"
log "Offline bundle dir: ${AIWS_ENV_BUNDLE_DIR}"
log "Offline bundle mode: ${AIWS_ENV_BUNDLE_MODE}"
log "YOLO image: ${AIWS_YOLO_IMAGE}"
log "GenPose2 image: ${AIWS_GENPOSE2_IMAGE}"
log "FoundationPose image: ${AIWS_FOUNDATIONPOSE_IMAGE}"

setup_conda_env
ensure_weights
ensure_image "YOLOv11" "${AIWS_YOLO_IMAGE}" "${REPO_ROOT}/yolov11-seg-aiws/docker/build_with_proxy.sh"
ensure_image "GenPose2" "${AIWS_GENPOSE2_IMAGE}" "${REPO_ROOT}/GenPose2/docker/build_with_proxy.sh"
ensure_image "FoundationPose" "${AIWS_FOUNDATIONPOSE_IMAGE}" "${REPO_ROOT}/aiws_alignment-feat-model-free/docker/build_with_proxy.sh"

cat <<EOF
[build_aiws_stack] Done.

Optional runtime exports:
  export AIWS_HF_WEIGHTS_REPO=${AIWS_HF_WEIGHTS_REPO}
  export AIWS_HF_WEIGHTS_REVISION=${AIWS_HF_WEIGHTS_REVISION}
  export AIWS_YOLO_IMAGE=${AIWS_YOLO_IMAGE}
  export AIWS_GENPOSE2_IMAGE=${AIWS_GENPOSE2_IMAGE}
  export AIWS_FOUNDATIONPOSE_IMAGE=${AIWS_FOUNDATIONPOSE_IMAGE}

Offline bundle:
  export AIWS_ENV_BUNDLE_DIR=${AIWS_ENV_BUNDLE_DIR}
  export AIWS_ENV_BUNDLE_MODE=${AIWS_ENV_BUNDLE_MODE}

Run:
  bash scripts/run_aiws.sh aiws_sub "data/AIWS/F120/1_color.png" "data/AIWS/F120/1_depth.exr" 0
EOF
