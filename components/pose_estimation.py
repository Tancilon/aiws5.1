import json
import shutil
import subprocess

from pathlib import Path
from typing import Mapping

import numpy as np
from components.base_algorithm import BaseAlgorithm
from utils.docker_runtime import DockerMount, build_docker_run_cmd
from utils.foundationpose_depth import FoundDepthManagement
from utils.foundationpose_mask import FoundMaskManagement


class PoseEstimation(BaseAlgorithm):
    def __init__(self,
                 data_path: Path,
                 tmp_output_path: Path,
                 intrinsics: Mapping,
                 workpiece_info_path: Path,
                 image_name: str,
                 warp_cache_host: Path,
                 prefix:str = 'foundationpose_'):
        super().__init__()

        self.data_path = Path(data_path).resolve()
        self.tmp_output_path = Path(tmp_output_path).resolve()
        self.intrinsics = intrinsics
        self.workpiece_info_path = Path(workpiece_info_path).resolve()
        self.image_name = image_name
        self.warp_cache_host = Path(warp_cache_host).resolve()
        self.warp_cache_host.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
    
    def infer(self,
              rgb_path: Path,
              depth_path: Path,
              class_name: str,
              instance_length: np.ndarray):
        rgb_path = Path(rgb_path)
        if not rgb_path.exists():
            raise FileNotFoundError(f"[PoseEstimation] rgb_path not found: {rgb_path}")
        depth_path = Path(depth_path)
        if not depth_path.exists():
           raise FileNotFoundError(f"[PoseEstimation] depth_path not found: {depth_path}")
        
        instance_length = np.asarray(instance_length, dtype=float)
        if instance_length.ndim != 1:
            raise ValueError("[PoseEstimation] instance_length must be 1D numpy array")
        if instance_length.shape[0] != 3:
            raise ValueError("[PoseEstimation] instance_length must have 3 elements")

        self.prepare_data(rgb_path, depth_path, instance_length, class_name)

        cmd = self.build_cmd()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"[PoseEstimation] FoundationPose failed\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
            ) from e
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        payload = None
        for line in reversed(lines):
            if line.startswith("__POSE_JSON__"):
                payload = json.loads(line.replace("__POSE_JSON__", "", 1))
                break
        if payload is None:
            raise RuntimeError(
                "[PoseEstimation] No pose JSON output received\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return np.array(payload["pose"], dtype=np.float32)

    def build_cmd(self):
        if not hasattr(self, "mesh_file_path"):
            raise RuntimeError("[PoseEstimation] mesh_file_path not set")

        host_foundationpose_root = Path(__file__).resolve().parents[1] / "aiws_alignment-feat-model-free"
        foundationpose_root = Path("/opt/foundationpose")

        script = f"""
import json
import os
import sys
from pathlib import Path

foundationpose_root = Path(r"{foundationpose_root}")
sys.path.insert(0, str(foundationpose_root))
os.chdir(str(foundationpose_root))

from foundationpose_aiws import foundationpose_aiws

mesh_file = Path(r"{self.mesh_file_path}").resolve()
test_scene_dir = Path(r"{self.data_path}").resolve()
debug_dir = Path(r"{self.tmp_output_path / "foundation_debug"}").resolve()

pose = foundationpose_aiws(mesh_file=mesh_file, test_scene_dir=test_scene_dir, debug_dir=debug_dir, debug=2)

pose_out = None
if pose is not None:
    pose_out = pose.tolist()

print("__POSE_JSON__" + json.dumps({{"pose": pose_out}}))
"""
        return build_docker_run_cmd(
            image_name=self.image_name,
            container_cmd=["python", "-c", script],
            workdir=foundationpose_root,
            env={
                "PYTHONPATH": str(foundationpose_root),
                "WARP_CACHE_DIR": "/tmp/warp-cache",
                "OPENCV_IO_ENABLE_OPENEXR": "1",
            },
            mounts=[
                DockerMount(
                    source=host_foundationpose_root.resolve(),
                    target=foundationpose_root,
                    mode="rw",
                ),
                DockerMount(source=self.data_path, target=self.data_path, mode="rw"),
                DockerMount(source=self.tmp_output_path, target=self.tmp_output_path, mode="rw"),
                DockerMount(
                    source=self.warp_cache_host,
                    target=Path("/tmp/warp-cache"),
                    mode="rw",
                )
            ],
            use_gpu=True,
            ipc_host=True,
            network_host=True,
        )
    
    def prepare_data(self, rgb_path: Path, depth_path: Path, length: np.ndarray, class_name: str):
        data_path = Path(self.data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        rgb_dir = data_path / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        rgb_dst = rgb_dir / f"0{rgb_path.suffix}"
        
        depth_dir = data_path / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_dst = depth_dir / f"0{depth_path.suffix}"

        mask_dir = data_path / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_candidates = [
            path for path in self.tmp_output_path.iterdir()
            if path.is_file() and path.stem.endswith("_mask")
        ]
        if not mask_candidates:
            raise FileNotFoundError("[PoseEstimation] Mask file not found.")
        if len(mask_candidates) != 1:
            raise RuntimeError(f"[PoseEstimation] Mask file count abnormal: {len(mask_candidates)}")

        mask_src = mask_candidates[0]
        mask_dst = mask_dir / f"0{mask_src.suffix}"
        
        shutil.copy2(rgb_path, rgb_dst)
        shutil.copy2(depth_path, depth_dst)
        shutil.copy2(str(mask_src), mask_dst)

        FoundDepthManagement.check_depth(depth_dst)
        FoundMaskManagement.convert_mask(mask_dst)

        intrinsics = dict(self.intrinsics)
        required_keys = ("fx", "fy", "cx", "cy")
        missing = [key for key in required_keys if key not in intrinsics]
        if missing:
            raise KeyError(f"[PoseEstimation] Missing intrinsics keys: {missing}")

        cam_k_path = self.data_path / "cam_K.txt"
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        with cam_k_path.open("w", encoding="utf-8") as f:
            f.write(f"{fx:.6f} 0.000000 {cx:.6f}\n")
            f.write(f"0.000000 {fy:.6f} {cy:.6f}\n")
            f.write("0.000000 0.000000 1.000000\n")

        workpiece_info_path = Path(self.workpiece_info_path)
        if not workpiece_info_path.exists():
            raise FileNotFoundError(f"[PoseEstimation] workpiece_info_path not found: {workpiece_info_path}")

        with workpiece_info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)

        if class_name not in info:
            raise KeyError(f"[PoseEstimation] class_name not found: {class_name}")

        obj_path_value = info[class_name].get("obj_path")
        if not obj_path_value:
            raise ValueError(f"[PoseEstimation] obj_path missing for class: {class_name}")

        obj_path = Path(obj_path_value)
        if not obj_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            obj_path = repo_root / obj_path

        if not obj_path.exists():
            raise FileNotFoundError(f"[PoseEstimation] obj_path not found: {obj_path}")

        mesh_dir = self.data_path / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        scaled_obj_dst = mesh_dir / obj_path.name
        with obj_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        scaled_lines = []
        scale_x, scale_y, scale_z = [float(x) for x in length]
        for line in lines:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x = float(parts[1]) * scale_x
                    y = float(parts[2]) * scale_y
                    z = float(parts[3]) * scale_z
                    rest = " ".join(parts[4:])
                    if rest:
                        scaled_lines.append(f"v {x:.6f} {y:.6f} {z:.6f} {rest}\n")
                    else:
                        scaled_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                else:
                    scaled_lines.append(line)
            else:
                scaled_lines.append(line)

        with scaled_obj_dst.open("w", encoding="utf-8") as f:
            f.writelines(scaled_lines)

        self.mesh_file_path = scaled_obj_dst.resolve()

        for item in obj_path.parent.iterdir():
            if not item.is_file():
                continue
            if item.name == obj_path.name:
                continue
            shutil.copy2(item, mesh_dir / item.name)
        
