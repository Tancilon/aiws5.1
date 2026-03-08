import json
import math
import shutil
import subprocess

from itertools import permutations
from pathlib import Path
from typing import Mapping

import numpy as np
from termcolor import cprint
from components.base_algorithm import BaseAlgorithm
from utils.docker_runtime import DockerMount, build_docker_run_cmd
from utils.genpose2_depth import GenPose2DepthManagement

try:
    import cv2
except Exception:
    cv2 = None

class DimensionMeasurement(BaseAlgorithm):
    def __init__(self,
                 data_path: Path,
                 tmp_output_path: Path,
                 intrinsics: Mapping,
                 workpiece_info_path: Path,
                 sigma: float,
                 image_name: str,
                 torch_cache_host: Path,
                 query_mode: bool = True,
                 prefix: str = 'genpose2_'):
        super().__init__()

        data_path = Path(data_path)
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            cprint("[DimensionMeasurement] data_path does not exist, creating it", "blue")
        
        tmp_output_path = Path(tmp_output_path)
        if not tmp_output_path.exists():
            raise FileNotFoundError(f"[DimensionMeasurement] tmp_output_path not found: {tmp_output_path}")
        if tmp_output_path.exists() and not tmp_output_path.is_dir():
            raise NotADirectoryError(f"[DimensionMeasurement] {tmp_output_path} is not a directory")

        workpiece_info_path = Path(workpiece_info_path)
        if not workpiece_info_path.exists():
            raise FileNotFoundError(f"[DimensionMeasurement] workpiece_info_path not found: {workpiece_info_path}")

        self.data_path = data_path
        self.tmp_output_path = tmp_output_path
        self.prefix = prefix
        self.query_mode = query_mode
        self.intrinsics = intrinsics
        self.workpiece_info_path = workpiece_info_path
        self.sigma = sigma
        self.image_name = image_name
        self.torch_cache_host = Path(torch_cache_host).resolve()
        self.torch_cache_host.mkdir(parents=True, exist_ok=True)

    def infer(self,
              rgb_path: Path,
              depth_path: Path,
              class_name: str):
        rgb_path = Path(rgb_path)
        if not rgb_path.exists():
           raise FileNotFoundError(f"[DimensionMeasurement] rgb_path not found: {rgb_path}")
        depth_path = Path(depth_path)
        if not depth_path.exists():
           raise FileNotFoundError(f"[DimensionMeasurement] depth_path not found: {depth_path}")
        self.prepare_data(rgb_path, depth_path)
        cmd = self.build_cmd()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"[DimensionMeasurement] GenPose2 failed\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}") from e

        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("[DimensionMeasurement] No inference output received")
        payload = json.loads(lines[-1])
        length_1d = payload["length"][0][0]
        
        if not self.query_mode:
            return length_1d
        
        else:
            return self.query(length_1d, class_name)

    def prepare_data(self, rgb_path: Path, depth_path: Path):
        data_path = Path(self.data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        rgb_dst = self.data_path / "0_color.png"
        depth_dst = self.data_path / f"0_depth{depth_path.suffix}"

        shutil.copy2(rgb_path, rgb_dst)
        shutil.copy2(depth_path, depth_dst)

        GenPose2DepthManagement.check_depth(depth_dst)

        mask_candidates = [
            path for path in self.tmp_output_path.iterdir()
            if path.is_file() and path.stem.endswith("_mask")
        ]
        if not mask_candidates:
            raise FileNotFoundError("[DimensionMeasurement] Mask file not found.")
        if len(mask_candidates) != 1:
            raise RuntimeError(f"[DimensionMeasurement] Mask file count abnormal: {len(mask_candidates)}")

        mask_src = mask_candidates[0]
        mask_dst = self.data_path / f"0_mask{mask_src.suffix}"
        shutil.copy2(str(mask_src), mask_dst)

        intrinsics = dict(self.intrinsics)
        if "width" not in intrinsics or "height" not in intrinsics:
            width = None
            height = None
            if depth_path.suffix.lower() == ".exr":
                depth, _ = GenPose2DepthManagement._read_exr(depth_path)
                height, width = depth.shape
            elif cv2 is not None:
                img = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    height, width = img.shape[:2]
            else:
                ext = rgb_path.suffix.lower()
                if ext in (".npy", ".npz"):
                    arr = np.load(str(rgb_path))
                    if isinstance(arr, np.lib.npyio.NpzFile):
                        key = list(arr.keys())[0]
                        arr = arr[key]
                    height, width = arr.shape[:2]
            if width is None or height is None:
                raise RuntimeError("[DimensionMeasurement] Missing intrinsics width/height and failed to infer from data")
            intrinsics["width"] = int(width)
            intrinsics["height"] = int(height)

        meta_path = self.data_path / "0_meta.json"
        meta = {"camera": {"intrinsics": intrinsics}}
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f)

        

    def build_cmd(self):
        data_path = Path(self.data_path).resolve()
        save_dir = Path(self.tmp_output_path).resolve()
        host_genpose_root = Path(__file__).resolve().parents[1] / "GenPose2"
        genpose_root = Path("/opt/genpose2")
        infer_script = genpose_root / "runners" / "infer_genpose2.py"

        script = f"""
import importlib.util
import json
import os
from pathlib import Path

genpose_root = Path(r"{genpose_root}")
infer_script = Path(r"{infer_script}")
data_path = r"{data_path}"
save_dir = r"{save_dir}"

os.chdir(str(genpose_root))
spec = importlib.util.spec_from_file_location("infer_genpose2", str(infer_script))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pose, length = module.infer_genpose2(data_path=data_path, save_dir=save_dir)

pose_out = []
length_out = []
if pose is not None:
    pose_out = [p.cpu().numpy().tolist() for p in pose]
if length is not None:
    length_out = [l.cpu().numpy().tolist() for l in length]

print(json.dumps({{"pose": pose_out, "length": length_out}}))
"""
        return build_docker_run_cmd(
            image_name=self.image_name,
            container_cmd=["python", "-u", "-c", script],
            workdir=genpose_root,
            env={
                "TORCH_HOME": "/workspace_cache/torch",
                "PYTHONPATH": str(genpose_root),
            },
            mounts=[
                DockerMount(source=host_genpose_root.resolve(), target=genpose_root, mode="rw"),
                DockerMount(source=data_path, target=data_path, mode="rw"),
                DockerMount(source=save_dir, target=save_dir, mode="rw"),
                DockerMount(
                    source=self.torch_cache_host,
                    target=Path("/workspace_cache/torch"),
                    mode="rw",
                )
            ],
            use_gpu=True,
            ipc_host=True,
            network_host=True,
        )

    def query(self, length, class_name: str):
        if not class_name:
            raise ValueError("[DimensionMeasurement] class_name is required for query")

        with self.workpiece_info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)

        if class_name not in info:
            raise KeyError(f"[DimensionMeasurement] class_name not found: {class_name}")

        sizes = info[class_name].get("sizes", [])
        if not sizes:
            raise ValueError(f"[DimensionMeasurement] No sizes for class: {class_name}")

        if isinstance(length, np.ndarray):
            length = length.tolist()

        pred = length
        pred = [float(x) for x in pred]

        best_err = float("inf")
        best_size = None
        for size in sizes:
            size_vals = [float(x) for x in size]
            for perm in permutations(pred, 3):
                err = sum(abs(p - c) / c for p, c in zip(perm, size_vals)) / 3.0
                if err < best_err:
                    best_err = err
                    best_size = size_vals

        if best_size is None:
            raise RuntimeError("[DimensionMeasurement] No size matched")

        confidence = math.exp(-best_err / self.sigma)
        return np.array(best_size, dtype=np.float32), float(confidence)
