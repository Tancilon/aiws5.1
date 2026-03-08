import json
import shutil
import subprocess

from pathlib import Path

from components.base_algorithm import BaseAlgorithm
from utils.docker_runtime import DockerMount, build_docker_run_cmd


class CategoryRecognition(BaseAlgorithm):
    def __init__(self, 
                 ckpt_path: Path, 
                 tmp_output_path: Path,
                 image_name: str,
                 algorithm_type: str = "yolov11-seg",
                 prefix: str = 'yolo11_'
                 ):
        super().__init__()

        if not isinstance(algorithm_type, str):
            raise TypeError("[CategoryRecognition] algorithm_type must be str")

        if algorithm_type != "yolov11-seg":
            raise ValueError(f"[CategoryRecognition] Unsupported algorithm_type: {algorithm_type}")

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"[CategoryRecognition] ckpt_path not found: {ckpt_path}")
        
        tmp_output_path = Path(tmp_output_path)
        if not tmp_output_path.exists():
            raise FileNotFoundError(f"[CategoryRecognition] tmp_output_path not found: {tmp_output_path}")
        if tmp_output_path.exists() and not tmp_output_path.is_dir():
            raise NotADirectoryError(f"[CategoryRecognition] {tmp_output_path} is not a directory")


        self.algorithm_type = algorithm_type
        self.ckpt_path = ckpt_path.resolve()
        self.tmp_output_path = tmp_output_path.resolve()
        self.image_name = image_name
        self.prefix = prefix

    def infer(self, 
              rgb_path: Path,
              ):
        rgb_path = Path(rgb_path)
        if not rgb_path.exists():
           raise FileNotFoundError(f"[CategoryRecognition] rgb_path not found: {rgb_path}")
        tmp_dir = Path(self.tmp_output_path)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dst = (tmp_dir / (self.prefix + rgb_path.name)).resolve()
        shutil.copy2(rgb_path, dst)
        cmd = self.build_cmd(dst)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"[CategoryRecognition] YOLOv11 failed\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
            ) from e
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("[CategoryRecognition] No inference output received")
        payload = json.loads(lines[-1])
        return payload["predictions"]
        
        
    def build_cmd(self,
                  rgb_path:Path,
                  **kwargs):
        ckpt_path = self.ckpt_path.resolve()
        out_dir = self.tmp_output_path.resolve()
        script = f"""
from pathlib import Path
import json
import os
import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
from ultralytics import YOLO

ckpt_path = r"{ckpt_path}"
img_path = r"{rgb_path}"
out_dir = Path(r"{out_dir}")
out_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(ckpt_path)
results = model.predict(source=img_path, save=False, verbose=False, retina_masks=True)
res = results[0]
names = res.names or model.names
orig_h, orig_w = res.orig_shape

predictions = []
if res.boxes is not None and res.boxes.cls is not None:
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy().tolist()
    for cid, conf in zip(cls_ids, confs):
        predictions.append({{"class_id": int(cid), "class_name": str(names[int(cid)]), "confidence": float(conf)}})

mask_path = out_dir / (Path(img_path).stem + "_mask.exr")
chosen = None
if res.masks is not None and res.masks.data is not None and res.masks.data.numel() > 0:
    masks = res.masks.data.cpu().numpy()
    bin_masks = masks > 0.5
    areas = bin_masks.reshape(bin_masks.shape[0], -1).sum(axis=1)
    if areas.max() > 0:
        chosen = bin_masks[int(np.argmax(areas))]

if chosen is not None and chosen.shape != (orig_h, orig_w):
    chosen = cv2.resize(
        chosen.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

base = np.ones((orig_h, orig_w), dtype=np.float32)
if chosen is not None:
    base[chosen] = 1.0 / 255.0

mask = np.repeat(base[:, :, None], 3, axis=2)
cv2.imwrite(str(mask_path), mask)

print(json.dumps({{"predictions": predictions, "mask_path": str(mask_path)}}))
"""
        return build_docker_run_cmd(
            image_name=self.image_name,
            container_cmd=["python", "-c", script],
            env={
                "OPENCV_IO_ENABLE_OPENEXR": "1",
                "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            },
            mounts=[
                DockerMount(source=rgb_path.parent.resolve(), target=rgb_path.parent.resolve(), mode="ro"),
                DockerMount(source=ckpt_path.parent.resolve(), target=ckpt_path.parent.resolve(), mode="ro"),
                DockerMount(source=out_dir, target=out_dir, mode="rw"),
            ],
            use_gpu=True,
            ipc_host=False,
            network_host=True,
        )
        
        
