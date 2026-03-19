#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/aiws_env_bundle_lib.sh"

AIWS_ENV_BUNDLE_DIR=${AIWS_ENV_BUNDLE_DIR:-${REPO_ROOT}/assets/env_files}
AIWS_EXPORT_WEIGHTS=${AIWS_EXPORT_WEIGHTS:-1}
AIWS_EXPORT_IMAGES=${AIWS_EXPORT_IMAGES:-1}

AIWS_YOLO_IMAGE=${AIWS_YOLO_IMAGE:-yolov11-seg:infer}
AIWS_GENPOSE2_IMAGE=${AIWS_GENPOSE2_IMAGE:-genpose2-env:test}
AIWS_FOUNDATIONPOSE_IMAGE=${AIWS_FOUNDATIONPOSE_IMAGE:-foundationpose-env:test}

log() {
  echo "[export_aiws_env_bundle] $*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[export_aiws_env_bundle] Missing command: ${cmd}" >&2
    exit 1
  fi
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

write_manifest() {
  local manifest_path="${AIWS_ENV_BUNDLE_DIR}/bundle-manifest.txt"
  cat >"${manifest_path}" <<EOF
created_at=$(date -Iseconds)
repo_root=${REPO_ROOT}
weights_dir=${AIWS_ENV_BUNDLE_DIR}/weights
docker_images_dir=${AIWS_ENV_BUNDLE_DIR}/docker-images
docker_image=${AIWS_YOLO_IMAGE}|$(aiws_bundle_image_tar_path "${AIWS_ENV_BUNDLE_DIR}" "${AIWS_YOLO_IMAGE}")
docker_image=${AIWS_GENPOSE2_IMAGE}|$(aiws_bundle_image_tar_path "${AIWS_ENV_BUNDLE_DIR}" "${AIWS_GENPOSE2_IMAGE}")
docker_image=${AIWS_FOUNDATIONPOSE_IMAGE}|$(aiws_bundle_image_tar_path "${AIWS_ENV_BUNDLE_DIR}" "${AIWS_FOUNDATIONPOSE_IMAGE}")
EOF
}

export_weights() {
  local weights_dir="${AIWS_ENV_BUNDLE_DIR}/weights"
  rm -rf "${weights_dir}"
  mkdir -p "${weights_dir}"
  log "Exporting runtime weights into ${weights_dir}"
  aiws_copy_weights_to_stage "${REPO_ROOT}" "${weights_dir}"
  cp -f "${REPO_ROOT}/environment-aiws.yml" "${AIWS_ENV_BUNDLE_DIR}/environment-aiws.yml"
}

export_image() {
  local label="$1"
  local image_name="$2"
  local tar_path

  tar_path="$(aiws_bundle_image_tar_path "${AIWS_ENV_BUNDLE_DIR}" "${image_name}")"
  mkdir -p "$(dirname "${tar_path}")"
  if ! docker image inspect "${image_name}" >/dev/null 2>&1; then
    echo "[export_aiws_env_bundle] Docker image not found locally: ${image_name}" >&2
    exit 1
  fi

  log "Saving ${label} image ${image_name} -> ${tar_path}"
  rm -f "${tar_path}"
  docker save -o "${tar_path}" "${image_name}"
}

if bool_is_true "${AIWS_EXPORT_IMAGES}"; then
  require_cmd docker
fi

mkdir -p "${AIWS_ENV_BUNDLE_DIR}"

if bool_is_true "${AIWS_EXPORT_WEIGHTS}"; then
  export_weights
fi

if bool_is_true "${AIWS_EXPORT_IMAGES}"; then
  export_image "YOLOv11" "${AIWS_YOLO_IMAGE}"
  export_image "GenPose2" "${AIWS_GENPOSE2_IMAGE}"
  export_image "FoundationPose" "${AIWS_FOUNDATIONPOSE_IMAGE}"
fi

write_manifest

log "Done"
log "Bundle dir: ${AIWS_ENV_BUNDLE_DIR}"
