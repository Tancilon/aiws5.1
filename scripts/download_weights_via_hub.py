#!/usr/bin/env python3

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

try:
    from huggingface_hub import HfFolder
except Exception:  # pragma: no cover
    HfFolder = None


def resolve_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    if HfFolder is not None:
        token = HfFolder.get_token()
        if token:
            return token

    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token

    return None


def main() -> int:
    if len(sys.argv) < 5:
        print(
            "usage: download_weights_via_hub.py <repo_id> <revision> <local_dir> <remote_path>...",
            file=sys.stderr,
        )
        return 1

    repo_id = sys.argv[1]
    revision = sys.argv[2]
    local_dir = Path(sys.argv[3]).resolve()
    remote_paths = sys.argv[4:]
    local_dir.mkdir(parents=True, exist_ok=True)

    token = resolve_token()
    for remote_path in remote_paths:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=remote_path,
            revision=revision,
            local_dir=str(local_dir),
            token=token,
        )

    print(str(local_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
