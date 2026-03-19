#!/usr/bin/env bash
# example: 
# bash eval/run_eval.sh aiws_sub data/aiws5.1-dataset-test 0
# bash eval/run_eval.sh aiws_sub data/aiws5.1-dataset-test 0 --output-dir /tmp/aiws_eval_full
# 等价的conda命令：
# conda run --no-capture-output -n aiws python -u eval/run_eval.py \
  # --config config/aiws_sub.yaml \
  # --dataset-dir data/aiws5.1-dataset-test \
  # --gpu-id 0 \
  # --output-dir /tmp/aiws_eval_full

set -euo pipefail

CONFIG_INPUT="${1:-aiws_sub}"
DATASET_DIR="${2:-data/aiws5.1-dataset-test}"
GPU_ID="${3:-0}"

if [[ "${CONFIG_INPUT}" == *.yaml || "${CONFIG_INPUT}" == *.yml || "${CONFIG_INPUT}" == /* || "${CONFIG_INPUT}" == ./* || "${CONFIG_INPUT}" == ../* ]]; then
  CONFIG_PATH="${CONFIG_INPUT}"
else
  CONFIG_PATH="config/${CONFIG_INPUT}.yaml"
fi

shift $(( $# >= 3 ? 3 : $# ))

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
conda run --no-capture-output -n aiws python -u eval/run_eval.py \
  --config "${CONFIG_PATH}" \
  --dataset-dir "${DATASET_DIR}" \
  --gpu-id "${GPU_ID}" \
  "$@"
