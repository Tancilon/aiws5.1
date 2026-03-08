#!/usr/bin/env bash

set -euo pipefail

IMAGE_TAG="${1:-genpose2-env:test}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

HTTP_PROXY_VALUE="${HTTP_PROXY:-http://127.0.0.1:7890}"
HTTPS_PROXY_VALUE="${HTTPS_PROXY:-${HTTP_PROXY_VALUE}}"
ALL_PROXY_VALUE="${ALL_PROXY:-socks5h://127.0.0.1:7890}"
NO_PROXY_VALUE="${NO_PROXY:-localhost,127.0.0.1}"

docker run --rm \
  --gpus all \
  --ipc host \
  --network host \
  -v "${REPO_ROOT}/docker/smoke_test.py:/opt/genpose2/docker/smoke_test.py:ro" \
  -e HTTP_PROXY="${HTTP_PROXY_VALUE}" \
  -e HTTPS_PROXY="${HTTPS_PROXY_VALUE}" \
  -e ALL_PROXY="${ALL_PROXY_VALUE}" \
  -e NO_PROXY="${NO_PROXY_VALUE}" \
  -e http_proxy="${HTTP_PROXY_VALUE}" \
  -e https_proxy="${HTTPS_PROXY_VALUE}" \
  -e all_proxy="${ALL_PROXY_VALUE}" \
  -e no_proxy="${NO_PROXY_VALUE}" \
  -e PYTHONPATH="/opt/genpose2" \
  -e LD_LIBRARY_PATH="/opt/micromamba/envs/genpose2/lib:/opt/micromamba/envs/genpose2/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64" \
  "${IMAGE_TAG}" \
  python docker/smoke_test.py
