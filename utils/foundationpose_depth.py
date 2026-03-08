from pathlib import Path

import Imath
import OpenEXR
import cv2
import numpy as np

from common.depth_management import DepthManagement


class FoundDepthManagement(DepthManagement):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _read_exr(depth_path: Path):
        exr_file = OpenEXR.InputFile(str(depth_path))
        header = exr_file.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        channels = header["channels"]
        if "Y" in channels:
            channel_name = "Y"
        elif "R" in channels:
            channel_name = "R"
        else:
            channel_name = list(channels.keys())[0]

        pt = channels[channel_name].type
        if pt == Imath.PixelType(Imath.PixelType.FLOAT):
            dtype = np.float32
        elif pt == Imath.PixelType(Imath.PixelType.HALF):
            dtype = np.float16
        elif pt == Imath.PixelType(Imath.PixelType.UINT):
            dtype = np.uint32
        else:
            dtype = np.float32

        channel_str = exr_file.channel(channel_name, pt)
        depth = np.frombuffer(channel_str, dtype=dtype).reshape((height, width))
        return depth

    @staticmethod
    def check_depth(depth_path: Path):
        depth_path = Path(depth_path)
        if not depth_path.exists():
            raise FileNotFoundError(f"[FoundDepthManagement] depth_path not found: {depth_path}")

        if depth_path.suffix.lower() == ".png":
            return depth_path

        if depth_path.suffix.lower() != ".exr":
            raise ValueError(f"[FoundDepthManagement] Unsupported depth format: {depth_path.suffix}")

        depth = FoundDepthManagement._read_exr(depth_path)
        valid = depth[depth > 0]

        is_millimeter = False
        if valid.size > 0:
            if float(valid.max()) > 10:
                is_millimeter = True
            if float(valid.mean()) > 1000:
                is_millimeter = True

        if not is_millimeter:
            depth = depth * 1000.0

        depth_mm = np.clip(depth, 0, 65535).astype(np.uint16)
        new_path = depth_path.with_suffix(".png")
        if not cv2.imwrite(str(new_path), depth_mm):
            raise RuntimeError(f"[FoundDepthManagement] Failed to write PNG: {new_path}")

        depth_path.unlink()
        return new_path
