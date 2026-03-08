#!/usr/bin/env bash
set -euo pipefail

TARGET_BASE="${1:-/media/bakhda4}"
TARGET_BASE="${TARGET_BASE%/}"
DOCKER_SOURCE="/var/lib/docker"
CONTAINERD_SOURCE="/var/lib/containerd"
DOCKER_TARGET="${DOCKER_TARGET:-${TARGET_BASE}/docker}"
CONTAINERD_TARGET="${CONTAINERD_TARGET:-${TARGET_BASE}/containerd}"
KEEP_OLD="${KEEP_OLD:-0}"
TIMESTAMP="$(date +%F-%H%M%S)"

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run this script with sudo or as root."
    exit 1
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

backup_file() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    cp "${path}" "${path}.bak.${TIMESTAMP}"
  fi
}

require_root
require_cmd rsync
require_cmd python3
require_cmd docker
require_cmd systemctl

if ! findmnt "${TARGET_BASE}" >/dev/null 2>&1; then
  echo "Target mount ${TARGET_BASE} does not exist."
  exit 1
fi

if [[ "${DOCKER_TARGET}" == "${DOCKER_SOURCE}" || "${CONTAINERD_TARGET}" == "${CONTAINERD_SOURCE}" ]]; then
  echo "Target directories must differ from the current Docker/containerd roots."
  exit 1
fi

mkdir -p "${DOCKER_TARGET}" "${CONTAINERD_TARGET}"

backup_file /etc/docker/daemon.json
backup_file /etc/containerd/config.toml

echo "Stopping docker and containerd..."
systemctl stop docker.service docker.socket || true
systemctl stop containerd.service || true

echo "Syncing ${DOCKER_SOURCE} -> ${DOCKER_TARGET}"
rsync -aHAX --numeric-ids --info=progress2 "${DOCKER_SOURCE}/" "${DOCKER_TARGET}/"

echo "Syncing ${CONTAINERD_SOURCE} -> ${CONTAINERD_TARGET}"
rsync -aHAX --numeric-ids --info=progress2 "${CONTAINERD_SOURCE}/" "${CONTAINERD_TARGET}/"

python3 - "${DOCKER_TARGET}" <<'PY'
import json
import sys
from pathlib import Path

target = sys.argv[1]
path = Path("/etc/docker/daemon.json")
if path.exists():
    data = json.loads(path.read_text())
else:
    data = {}

data["data-root"] = target
path.write_text(json.dumps(data, indent=4) + "\n")
PY

python3 - "${CONTAINERD_TARGET}" <<'PY'
import re
import sys
from pathlib import Path

target = sys.argv[1]
path = Path("/etc/containerd/config.toml")
text = path.read_text()

if re.search(r"(?m)^\s*root\s*=", text):
    text = re.sub(r'(?m)^\s*root\s*=.*$', f'root = "{target}"', text)
elif '#root = "/var/lib/containerd"' in text:
    text = text.replace('#root = "/var/lib/containerd"', f'root = "{target}"')
else:
    text = text.rstrip() + f'\nroot = "{target}"\n'

path.write_text(text)
PY

echo "Restarting containerd and docker..."
systemctl daemon-reload
systemctl start containerd.service
systemctl start docker.service

docker_root="$(docker info --format '{{.DockerRootDir}}')"
if [[ "${docker_root}" != "${DOCKER_TARGET}" ]]; then
  echo "Docker Root Dir mismatch: ${docker_root}"
  exit 1
fi

if [[ "${KEEP_OLD}" != "1" ]]; then
  echo "Removing old Docker data from root disk..."
  rm -rf "${DOCKER_SOURCE}" "${CONTAINERD_SOURCE}"
  mkdir -p "${DOCKER_SOURCE}" "${CONTAINERD_SOURCE}"
fi

echo
echo "Migration complete."
echo "Docker Root Dir: ${docker_root}"
echo "Docker runtimes:"
docker info | sed -n '/Runtimes:/,/Default Runtime:/p'
echo "Disk usage:"
df -h / "${TARGET_BASE}"
