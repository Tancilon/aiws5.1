import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.infer_genpose2 import infer_genpose2


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GenPose2 smoke test")

    data_path = REPO_ROOT / "data" / "AIWS" / "000"
    save_dir = REPO_ROOT / "results" / "docker_smoke"

    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pose, length = infer_genpose2(data_path=str(data_path), save_dir=str(save_dir))

    pose_list = [item.cpu().numpy().tolist() for item in pose]
    length_list = [item.cpu().numpy().tolist() for item in length]

    if not pose_list or not length_list:
        raise RuntimeError("GenPose2 smoke test returned empty pose or length outputs")

    print(json.dumps({
        "cuda_available": True,
        "pose_batches": len(pose_list),
        "length_batches": len(length_list),
        "first_length": length_list[0][0],
        "save_dir": str(save_dir),
    }))


if __name__ == "__main__":
    main()
