#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import Imath
import OpenEXR
import numpy as np
import trimesh
from PIL import Image


DEFAULT_EXR_DEPTH_CHANNELS = ("Z", "Y", "R")


class PoseAlignmentMetricEvaluator:
    """Evaluate pose alignment quality by comparing observed and rendered depth."""

    def __init__(
        self,
        batch_size: int = 10000,
        preferred_exr_depth_channels: Sequence[str] = DEFAULT_EXR_DEPTH_CHANNELS,
        verbose: bool = True,
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.preferred_exr_depth_channels = tuple(preferred_exr_depth_channels)
        self.verbose = bool(verbose)

    def evaluate(
        self,
        obs_depth_path: Path | str,
        mesh_path: Path | str,
        pose: np.ndarray,
        camera_intrinsics: np.ndarray,
        mask_path: Path | str | None = None,
        output_dir: Path | str | None = None,
        rendered_depth_filename: str = "rendered_depth.exr",
    ) -> Dict[str, float]:
        obs_depth = self.read_exr_to_array(obs_depth_path)
        pose_array = np.asarray(pose, dtype=float)
        camera_matrix = np.asarray(camera_intrinsics, dtype=float)

        if pose_array.shape != (4, 4):
            raise ValueError(f"Expected pose shape (4, 4), got {pose_array.shape}")
        if camera_matrix.shape != (3, 3):
            raise ValueError(f"Expected camera intrinsics shape (3, 3), got {camera_matrix.shape}")

        height, width = obs_depth.shape
        rendered_depth = self.render_depth(mesh_path, pose_array, camera_matrix, height=height, width=width)

        if output_dir is not None:
            rendered_depth_path = Path(output_dir) / rendered_depth_filename
            self.save_depth_as_exr(rendered_depth, rendered_depth_path)

        return self.calculate_depth_metrics(obs_depth, rendered_depth, mask_path)

    def evaluate_from_files(
        self,
        obs_depth_path: Path | str,
        mesh_path: Path | str,
        pose_path: Path | str,
        cam_k_path: Path | str,
        mask_path: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> Dict[str, float]:
        pose_path = Path(pose_path)
        pose = self.load_pose(pose_path)
        camera_matrix = self.load_camera_intrinsic(cam_k_path)
        rendered_depth_filename = f"{pose_path.stem}_rendered.exr"
        return self.evaluate(
            obs_depth_path=obs_depth_path,
            mesh_path=mesh_path,
            pose=pose,
            camera_intrinsics=camera_matrix,
            mask_path=mask_path,
            output_dir=output_dir,
            rendered_depth_filename=rendered_depth_filename,
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def get_exr_depth_channel(self, channels) -> Tuple[str, Imath.PixelType, np.dtype]:
        if not channels:
            raise ValueError("No channel is present in the EXR file.")

        channel_name = None
        for preferred_name in self.preferred_exr_depth_channels:
            if preferred_name in channels:
                channel_name = preferred_name
                break

        if channel_name is None:
            channel_name = next(iter(channels.keys()))

        pixel_type = channels[channel_name].type
        if pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
            dtype = np.float32
        elif pixel_type == Imath.PixelType(Imath.PixelType.HALF):
            dtype = np.float16
        elif pixel_type == Imath.PixelType(Imath.PixelType.UINT):
            dtype = np.uint32
        else:
            dtype = np.float32

        return channel_name, pixel_type, dtype

    def read_exr_to_array(self, file_path: Path | str) -> np.ndarray:
        input_file = OpenEXR.InputFile(str(file_path))
        try:
            header = input_file.header()
            data_window = header["dataWindow"]
            width = data_window.max.x - data_window.min.x + 1
            height = data_window.max.y - data_window.min.y + 1
            channel_name, pixel_type, dtype = self.get_exr_depth_channel(header["channels"])
            depth_bytes = input_file.channel(channel_name, pixel_type)
            depth_data = np.frombuffer(depth_bytes, dtype=dtype).reshape(height, width)
            return depth_data.astype(np.float32, copy=False)
        finally:
            input_file.close()

    def convert_depth_to_mm(self, depth: np.ndarray, depth_name: str) -> np.ndarray:
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        depth = depth.astype(np.float32, copy=False)

        if valid_depth.size == 0:
            return depth

        is_millimeter = float(valid_depth.max()) > 10 or float(valid_depth.mean()) > 1000
        if is_millimeter:
            return depth

        self._log(f"[INFO] Converting {depth_name} from meters to millimeters.")
        return depth * 1000.0

    def load_mask(self, mask_path: Path | str) -> np.ndarray:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask > 0

    def render_depth(
        self,
        mesh_path: Path | str,
        pose: np.ndarray,
        camera_intrinsics: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        mesh = trimesh.load(mesh_path)
        self._log(f"[DEBUG] Mesh loaded: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
        self._log(f"[DEBUG] Mesh bounds: {mesh.bounds}")

        mesh.apply_transform(pose)

        fx = float(camera_intrinsics[0, 0])
        fy = float(camera_intrinsics[1, 1])
        cx = float(camera_intrinsics[0, 2])
        cy = float(camera_intrinsics[1, 2])
        self._log(f"[DEBUG] Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

        depth = np.zeros((height, width), dtype=np.float32)
        camera_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

        total_pixels = height * width
        self._log("[INFO] Rendering depth map using ray casting (this may take a while)...")
        for batch_start in range(0, total_pixels, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_pixels)
            pixel_indices = np.arange(batch_start, batch_end)
            v = pixel_indices // width
            u = pixel_indices % width

            x_norm = (u - cx) / fx
            y_norm = (v - cy) / fy
            ray_directions = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)
            ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)
            ray_origins = np.tile(camera_origin, (len(ray_directions), 1))

            locations, index_ray, _ = intersector.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=False,
            )

            if len(locations) > 0:
                depth.flat[batch_start + index_ray] = locations[:, 2]

            if self.verbose and (batch_start // self.batch_size) % 10 == 0:
                progress = (batch_end / total_pixels) * 100.0
                print(f"  Progress: {progress:.1f}%", end="\r")

        if self.verbose:
            valid_pixels = int((depth > 0).sum())
            min_depth = float(depth[depth > 0].min()) if valid_pixels > 0 else 0.0
            print(
                f"\n[DEBUG] Rendered depth: min={min_depth:.4f}, "
                f"max={float(depth.max()):.4f}, valid_pixels={valid_pixels}/{depth.size}"
            )

        return depth

    def calculate_depth_metrics(
        self,
        obs_depth: np.ndarray,
        rendered_depth: np.ndarray,
        mask_path: Path | str | None = None,
    ) -> Dict[str, float]:
        obs_depth_mm = self.convert_depth_to_mm(obs_depth, "observed depth")
        rendered_depth_mm = self.convert_depth_to_mm(rendered_depth, "rendered depth")

        obs_mask = obs_depth_mm > 0
        template_mask = rendered_depth_mm > 0

        if mask_path is not None:
            foreground_mask = self.load_mask(mask_path)
            if foreground_mask.shape != obs_depth_mm.shape:
                raise ValueError(f"Mask shape {foreground_mask.shape} != depth shape {obs_depth_mm.shape}")
            obs_mask = obs_mask & foreground_mask

        inter_mask = obs_mask & template_mask
        template_total = int(template_mask.sum())
        if template_total == 0:
            self._log("Warning: No valid template pixels found!")
            return {"obs_point_cloud_cover_rate": 0.0, "avg_dist": 0.0}

        obs_point_cloud_cover_rate = float(inter_mask.sum() / template_total)

        err_dist = np.abs(obs_depth_mm - rendered_depth_mm)
        inter_mask = inter_mask & (err_dist < 80.0)
        if int(inter_mask.sum()) == 0:
            self._log("Warning: No valid intersection pixels after filtering!")
            avg_dist = 0.0
        else:
            avg_dist = float(err_dist[inter_mask].mean())

        return {
            "obs_point_cloud_cover_rate": obs_point_cloud_cover_rate,
            "avg_dist": avg_dist,
        }

    def load_camera_intrinsic(self, cam_k_path: Path | str) -> np.ndarray:
        return np.loadtxt(cam_k_path)

    def load_pose(self, pose_path: Path | str) -> np.ndarray:
        return np.loadtxt(pose_path)

    def save_depth_as_exr(self, depth: np.ndarray, output_path: Path | str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = depth.shape

        header = OpenEXR.Header(width, height)
        header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

        output_file = OpenEXR.OutputFile(str(output_path), header)
        try:
            output_file.writePixels({"Z": depth.astype(np.float32).tobytes()})
        finally:
            output_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose alignment metric evaluation.")
    parser.add_argument("--obs_depth", type=str, required=True)
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--pose", type=str, required=True)
    parser.add_argument("--cam_K", type=str, required=True)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluator = PoseAlignmentMetricEvaluator(batch_size=args.batch_size, verbose=not args.quiet)
    result = evaluator.evaluate_from_files(
        obs_depth_path=args.obs_depth,
        mesh_path=args.mesh,
        pose_path=args.pose,
        cam_k_path=args.cam_K,
        mask_path=args.mask,
        output_dir=args.output_dir,
    )
    print("\nResults:")
    print(f"Coverage Rate: {result['obs_point_cloud_cover_rate']:.4f}")
    print(f"Avg Distance: {result['avg_dist']:.4f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
