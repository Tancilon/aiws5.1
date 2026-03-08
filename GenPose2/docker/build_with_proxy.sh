#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
IMAGE_TAG="${1:-genpose2-env:test}"

HTTP_PROXY_VALUE="${HTTP_PROXY:-http://127.0.0.1:7890}"
HTTPS_PROXY_VALUE="${HTTPS_PROXY:-${HTTP_PROXY_VALUE}}"
ALL_PROXY_VALUE="${ALL_PROXY:-socks5h://127.0.0.1:7890}"
NO_PROXY_VALUE="${NO_PROXY:-localhost,127.0.0.1}"

docker build \
  --network host \
  --build-arg HTTP_PROXY="${HTTP_PROXY_VALUE}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY_VALUE}" \
  --build-arg ALL_PROXY="${ALL_PROXY_VALUE}" \
  --build-arg NO_PROXY="${NO_PROXY_VALUE}" \
  --build-arg http_proxy="${HTTP_PROXY_VALUE}" \
  --build-arg https_proxy="${HTTPS_PROXY_VALUE}" \
  --build-arg all_proxy="${ALL_PROXY_VALUE}" \
  --build-arg no_proxy="${NO_PROXY_VALUE}" \
  -t "${IMAGE_TAG}" \
  "${REPO_ROOT}"
