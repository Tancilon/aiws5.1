#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/weights_manifest.sh"

OUTPUT_DIR="${1:-${PROJECT_ROOT}/dist/hf-weights-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"

copy_file_into_bundle() {
  local src="$1"
  local rel_dst="$2"
  local dst="${OUTPUT_DIR}/${rel_dst}"

  if [[ ! -f "${src}" ]]; then
    echo "missing file: ${src}" >&2
    return 1
  fi

  mkdir -p "$(dirname "${dst}")"
  cp -f "${src}" "${dst}"
}

copy_dir_into_bundle() {
  local src="$1"
  local rel_dst="$2"
  local dst="${OUTPUT_DIR}/${rel_dst}"

  if [[ ! -d "${src}" ]]; then
    echo "missing directory: ${src}" >&2
    return 1
  fi

  mkdir -p "${dst}"
  cp -a "${src}/." "${dst}/"
}

archive_dir_into_bundle() {
  local src="$1"
  local rel_dst="$2"
  local dst="${OUTPUT_DIR}/${rel_dst}"

  if [[ ! -d "${src}" ]]; then
    echo "missing directory: ${src}" >&2
    return 1
  fi

  mkdir -p "$(dirname "${dst}")"
  tar -czf "${dst}" -C "$(dirname "${src}")" "$(basename "${src}")"
}

while IFS= read -r rel_path; do
  [[ -z "${rel_path}" ]] && continue
  copy_file_into_bundle "${PROJECT_ROOT}/${rel_path}" "${rel_path}"
done < <(aiws_project_weight_paths)

while IFS='|' read -r entry_type src rel_dst _restore_name; do
  [[ -z "${entry_type}" ]] && continue
  case "${entry_type}" in
    file)
      copy_file_into_bundle "${src}" "${rel_dst}"
      ;;
    dir)
      copy_dir_into_bundle "${src}" "${rel_dst}"
      ;;
    archive_dir)
      archive_dir_into_bundle "${src}" "${rel_dst}"
      ;;
    *)
      echo "unknown manifest entry type: ${entry_type}" >&2
      exit 1
      ;;
  esac
done < <(aiws_torch_cache_entries)

cat > "${OUTPUT_DIR}/README.md" <<'EOF'
# AIWS runtime weights

This bundle contains only the files required to run the AIWS pipeline:

- YOLOv11 segmentation checkpoint
- GenPose2 checkpoints
- FoundationPose checkpoints
- GenPose2 torch.hub cache for DINOv2

Files are arranged so they can be restored to the project tree by
`scripts/download_weights.sh`.
EOF

(
  cd "${OUTPUT_DIR}"
  find . -type f ! -path './.cache/*' ! -name 'SHA256SUMS' -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

echo "Prepared Hugging Face bundle:"
echo "${OUTPUT_DIR}"
echo
echo "Included project weights:"
aiws_project_weight_paths
echo
echo "Included extra cache entries:"
aiws_torch_cache_entries
