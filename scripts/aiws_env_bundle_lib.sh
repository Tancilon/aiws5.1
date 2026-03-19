#!/usr/bin/env bash

aiws_project_weight_paths() {
  cat <<'EOF'
yolov11-seg-aiws/runs/segment/train2/weights/best.pt
GenPose2/results/ckpts/ScoreNet/scorenet.pth
GenPose2/results/ckpts/EnergyNet/energynet.pth
GenPose2/results/ckpts/ScaleNet/scalenet.pth
aiws_alignment-feat-model-free/weights/2024-01-11-20-02-45/model_best.pth
aiws_alignment-feat-model-free/weights/2023-10-28-18-33-37/model_best.pth
EOF
}

aiws_torch_cache_entries() {
  cat <<EOF
file|${HOME}/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth|torch-hub/checkpoints/dinov2_vits14_pretrain.pth
archive_dir|${HOME}/.cache/torch/hub/facebookresearch_dinov2_main|torch-hub/facebookresearch_dinov2_main.tar.gz|facebookresearch_dinov2_main
EOF
}

aiws_sanitize_image_name() {
  local image_name="$1"
  local sanitized="${image_name//\//__}"
  sanitized="${sanitized//:/--}"
  sanitized="${sanitized//@/__at__}"
  sanitized="${sanitized//[^a-zA-Z0-9._-]/_}"
  printf '%s\n' "${sanitized}"
}

aiws_bundle_image_tar_path() {
  local bundle_dir="$1"
  local image_name="$2"
  printf '%s/docker-images/%s.tar\n' "${bundle_dir}" "$(aiws_sanitize_image_name "${image_name}")"
}

aiws_copy_weights_to_stage() {
  local repo_root="$1"
  local stage_dir="$2"
  local rel_path
  local entry_type
  local src
  local rel_dst
  local restore_name
  local dst

  mkdir -p "${stage_dir}"

  while IFS= read -r rel_path; do
    [[ -z "${rel_path}" ]] && continue
    src="${repo_root}/${rel_path}"
    if [[ ! -f "${src}" ]]; then
      echo "[aiws_env_bundle] Missing project weight: ${src}" >&2
      return 1
    fi
    dst="${stage_dir}/${rel_path}"
    mkdir -p "$(dirname "${dst}")"
    cp -f "${src}" "${dst}"
  done < <(aiws_project_weight_paths)

  while IFS='|' read -r entry_type src rel_dst restore_name; do
    [[ -z "${entry_type}" ]] && continue
    dst="${stage_dir}/${rel_dst}"
    mkdir -p "$(dirname "${dst}")"
    case "${entry_type}" in
      file)
        if [[ ! -f "${src}" ]]; then
          echo "[aiws_env_bundle] Missing torch cache file: ${src}" >&2
          return 1
        fi
        cp -f "${src}" "${dst}"
        ;;
      dir)
        if [[ ! -d "${src}" ]]; then
          echo "[aiws_env_bundle] Missing torch cache directory: ${src}" >&2
          return 1
        fi
        rm -rf "${dst}"
        mkdir -p "${dst}"
        cp -a "${src}/." "${dst}/"
        ;;
      archive_dir)
        if [[ ! -d "${src}" ]]; then
          echo "[aiws_env_bundle] Missing torch cache directory: ${src}" >&2
          return 1
        fi
        tar -czf "${dst}" -C "$(dirname "${src}")" "${restore_name}"
        ;;
      *)
        echo "[aiws_env_bundle] Unknown torch cache entry type: ${entry_type}" >&2
        return 1
        ;;
    esac
  done < <(aiws_torch_cache_entries)
}

aiws_restore_weights_from_stage() {
  local stage_dir="$1"
  local repo_root="$2"
  local rel_path
  local entry_type
  local rel_dst
  local restore_name
  local src
  local dst

  while IFS= read -r rel_path; do
    [[ -z "${rel_path}" ]] && continue
    src="${stage_dir}/${rel_path}"
    dst="${repo_root}/${rel_path}"
    if [[ ! -f "${src}" ]]; then
      echo "[aiws_env_bundle] Offline bundle missing project weight: ${src}" >&2
      return 1
    fi
    mkdir -p "$(dirname "${dst}")"
    cp -f "${src}" "${dst}"
  done < <(aiws_project_weight_paths)

  while IFS='|' read -r entry_type _src rel_dst restore_name; do
    [[ -z "${entry_type}" ]] && continue
    src="${stage_dir}/${rel_dst}"
    case "${entry_type}" in
      file)
        if [[ ! -f "${src}" ]]; then
          echo "[aiws_env_bundle] Offline bundle missing torch cache file: ${src}" >&2
          return 1
        fi
        dst="${HOME}/.cache/torch/hub/checkpoints/$(basename "${src}")"
        mkdir -p "$(dirname "${dst}")"
        cp -f "${src}" "${dst}"
        ;;
      dir)
        if [[ ! -d "${src}" ]]; then
          echo "[aiws_env_bundle] Offline bundle missing torch cache directory: ${src}" >&2
          return 1
        fi
        dst="${HOME}/.cache/torch/hub/$(basename "${src}")"
        mkdir -p "${dst}"
        cp -a "${src}/." "${dst}/"
        ;;
      archive_dir)
        if [[ ! -f "${src}" ]]; then
          echo "[aiws_env_bundle] Offline bundle missing torch cache archive: ${src}" >&2
          return 1
        fi
        dst="${HOME}/.cache/torch/hub"
        mkdir -p "${dst}"
        tar -xzf "${src}" -C "${dst}"
        ;;
      *)
        echo "[aiws_env_bundle] Unknown torch cache entry type: ${entry_type}" >&2
        return 1
        ;;
    esac
  done < <(aiws_torch_cache_entries)
}

aiws_weights_stage_ready() {
  local stage_dir="$1"
  local rel_path
  local entry_type
  local rel_dst

  while IFS= read -r rel_path; do
    [[ -z "${rel_path}" ]] && continue
    [[ -f "${stage_dir}/${rel_path}" ]] || return 1
  done < <(aiws_project_weight_paths)

  while IFS='|' read -r entry_type _src rel_dst _restore_name; do
    [[ -z "${entry_type}" ]] && continue
    case "${entry_type}" in
      file|archive_dir)
        [[ -f "${stage_dir}/${rel_dst}" ]] || return 1
        ;;
      dir)
        [[ -d "${stage_dir}/${rel_dst}" ]] || return 1
        ;;
      *)
        return 1
        ;;
    esac
  done < <(aiws_torch_cache_entries)
}
