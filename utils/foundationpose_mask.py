from pathlib import Path

import Imath
import OpenEXR
import cv2
import numpy as np

from common.mask_management import MaskManagement


class FoundMaskManagement(MaskManagement):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _read_exr(mask_path: Path):
        exr_file = OpenEXR.InputFile(str(mask_path))
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
        mask = np.frombuffer(channel_str, dtype=dtype).reshape((height, width))
        return mask

    @staticmethod
    def convert_mask(mask_path: Path):
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"[FoundMaskManagement] mask_path not found: {mask_path}")

        if mask_path.suffix.lower() == ".png":
            return mask_path

        if mask_path.suffix.lower() != ".exr":
            raise ValueError(f"[FoundMaskManagement] Unsupported mask format: {mask_path.suffix}")

        mask = FoundMaskManagement._read_exr(mask_path)

        foreground = mask < 0.5
        binary = np.zeros(mask.shape, dtype=np.uint8)
        binary[foreground] = 255

        new_path = mask_path.with_suffix(".png")
        if not cv2.imwrite(str(new_path), binary):
            raise RuntimeError(f"[FoundMaskManagement] Failed to write PNG: {new_path}")

        mask_path.unlink()
        return new_path
