from pathlib import Path

import Imath
import OpenEXR
import numpy as np
from termcolor import cprint

from common.depth_management import DepthManagement


class GenPose2DepthManagement(DepthManagement):
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
        return depth, channel_name

    @staticmethod
    def _write_exr(depth_path: Path, depth: np.ndarray, channel_name: str = "Y"):
        height, width = depth.shape
        depth = depth.astype(np.float32)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

        header = OpenEXR.Header(width, height)
        header["channels"] = {channel_name: Imath.Channel(pixel_type)}
        exr = OpenEXR.OutputFile(str(depth_path), header)
        exr.writePixels({channel_name: depth.tobytes()})
        exr.close()

    @staticmethod
    def check_depth(depth_path: Path):
        depth_path = Path(depth_path)
        if not depth_path.exists():
            raise FileNotFoundError(f"[GenPose2DepthManagement] depth_path not found: {depth_path}")
        if depth_path.suffix.lower() != ".exr":
            raise ValueError(f"[GenPose2DepthManagement] depth_path is not an EXR file: {depth_path}")

        depth, channel_name = GenPose2DepthManagement._read_exr(depth_path)
        valid_depth = depth[depth > 0]
        if valid_depth.size == 0:
            return

        is_millimeter = False
        if float(valid_depth.max()) > 10:
            is_millimeter = True
        if float(valid_depth.mean()) > 1000:
            is_millimeter = True

        needs_channel_fix = channel_name != "Y"
        is_meter = not is_millimeter
        cprint(f"[GenPose2DepthManagement] depth unit is meter: {is_meter}", "blue")
        cprint(f"[GenPose2DepthManagement] depth channel is Y: {not needs_channel_fix}", "blue")
        if not is_millimeter and not needs_channel_fix:
            return

        if is_millimeter:
            depth = depth / 1000.0

        GenPose2DepthManagement._write_exr(depth_path, depth, channel_name="Y")
