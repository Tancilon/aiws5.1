#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import HfApi
from huggingface_hub._upload_large_folder import upload_large_folder_internal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a large folder to Hugging Face while skipping the internal create_repo call."
    )
    parser.add_argument("repo_id")
    parser.add_argument("folder_path")
    parser.add_argument("--repo-type", required=True, choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--allow-pattern", action="append", dest="allow_patterns", default=[])
    parser.add_argument("--ignore-pattern", action="append", dest="ignore_patterns", default=[])
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder_path = Path(args.folder_path).expanduser().resolve()

    api = HfApi()
    api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type, revision=args.revision)

    def skip_create_repo(*_args, **_kwargs):
        return SimpleNamespace(repo_id=args.repo_id)

    api.create_repo = skip_create_repo  # type: ignore[method-assign]

    upload_large_folder_internal(
        api=api,
        repo_id=args.repo_id,
        folder_path=folder_path,
        repo_type=args.repo_type,
        revision=args.revision,
        private=args.private,
        allow_patterns=args.allow_patterns or None,
        ignore_patterns=args.ignore_patterns or None,
        num_workers=args.num_workers,
        print_report=not args.no_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
