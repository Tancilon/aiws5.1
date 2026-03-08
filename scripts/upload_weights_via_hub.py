#!/usr/bin/env python3

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

try:
    from huggingface_hub import HfFolder
except Exception:  # pragma: no cover
    HfFolder = None


def resolve_token() -> str:
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

    raise RuntimeError("No Hugging Face token found. Run 'hf auth login' first or set HF_TOKEN.")


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: upload_weights_via_hub.py <repo_id> <bundle_dir> <revision> <private_flag>",
            file=sys.stderr,
        )
        return 1

    repo_id = sys.argv[1]
    bundle_dir = Path(sys.argv[2]).resolve()
    revision = sys.argv[3]
    private_flag = sys.argv[4].lower() == "true"

    if not bundle_dir.exists():
        raise FileNotFoundError(f"bundle_dir not found: {bundle_dir}")

    token = resolve_token()
    api = HfApi(token=token)

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private_flag,
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(bundle_dir),
        path_in_repo=".",
        revision=revision,
        commit_message="Upload AIWS runtime weights",
    )

    print(f"https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
