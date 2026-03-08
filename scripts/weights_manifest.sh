#!/usr/bin/env bash

set -euo pipefail

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
