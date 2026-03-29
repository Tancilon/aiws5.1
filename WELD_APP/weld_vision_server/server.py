import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import OmegaConf
import hydra
from datetime import datetime


class VisionService:
    def __init__(self, cfg_path: Path):
        self.repo_root = Path(__file__).resolve().parents[2]
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        os.chdir(self.repo_root)

        cfg_path = Path(cfg_path)
        if not cfg_path.is_absolute():
            cfg_path = (self.repo_root / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        if not OmegaConf.has_resolver("now"):
            OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg

        self.category_algo = hydra.utils.instantiate(cfg.category_recognition)
        self.dimension_algo = hydra.utils.instantiate(cfg.dimension_measurement)
        self.pose_algo = hydra.utils.instantiate(cfg.pose_estimation)

        self.tmp_dir = Path(cfg.tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def _clean_tmp(self):
        if not self.tmp_dir.exists():
            return
        for item in self.tmp_dir.iterdir():
            try:
                if item.is_dir():
                    import shutil
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception:
                # Best effort cleanup; keep going.
                pass

    def _normalize_intrinsics(self, intrinsics: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if intrinsics is None:
            return None
        if "K" in intrinsics:
            k = np.asarray(intrinsics["K"], dtype=float)
            data = {
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
            }
            for key in ("width", "height"):
                if key in intrinsics:
                    data[key] = float(intrinsics[key])
            return data
        # Assume already in fx/fy/cx/cy format (optionally width/height).
        data = {k: float(v) for k, v in intrinsics.items() if k in ("fx", "fy", "cx", "cy", "width", "height")}
        return data

    def _apply_intrinsics_override(self, intrinsics: Optional[Dict[str, Any]]):
        intr = self._normalize_intrinsics(intrinsics)
        if not intr:
            return
        self.dimension_algo.intrinsics = intr
        self.pose_algo.intrinsics = intr

    def _find_latest_mask(self) -> Optional[Path]:
        if not self.tmp_dir.exists():
            return None
        candidates = [
            p for p in self.tmp_dir.iterdir()
            if p.is_file() and p.stem.endswith("_mask")
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _find_first_debug_vis(self) -> Optional[Path]:
        vis_dir = self.tmp_dir / "foundation_debug" / "track_vis"
        if not vis_dir.exists():
            return None
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        images = [p for p in vis_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not images:
            return None
        images.sort(key=lambda p: p.name)
        return images[0]

    def category_recognition(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rgb_path = payload.get("rgb_path")
        if not rgb_path:
            raise ValueError("rgb_path is required")

        if payload.get("clean_tmp", True):
            self._clean_tmp()

        predictions = self.category_algo.infer(rgb_path)
        selected = None
        if predictions:
            selected = max(predictions, key=lambda p: p.get("confidence", 0))

        mask_path = self._find_latest_mask()
        return {
            "predictions": predictions,
            "selected": selected,
            "mask_path": str(mask_path) if mask_path else None,
        }

    def dimension_measurement(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rgb_path = payload.get("rgb_path")
        depth_path = payload.get("depth_path")
        class_name = payload.get("class_name")
        if not rgb_path or not depth_path or not class_name:
            raise ValueError("rgb_path, depth_path, and class_name are required")

        self._apply_intrinsics_override(payload.get("intrinsics"))

        result = self.dimension_algo.infer(rgb_path, depth_path, class_name)
        if self.dimension_algo.query_mode:
            size_mm, confidence = result
        else:
            size_mm, confidence = result, None

        size_mm = np.asarray(size_mm, dtype=float).reshape(-1).tolist()
        return {
            "size_mm": size_mm,
            "confidence": float(confidence) if confidence is not None else None,
            "size_source": "genpose2",
        }

    def pose_estimation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rgb_path = payload.get("rgb_path")
        depth_path = payload.get("depth_path")
        class_name = payload.get("class_name")
        size_mm = payload.get("size_mm")
        if not rgb_path or not depth_path or not class_name or size_mm is None:
            raise ValueError("rgb_path, depth_path, class_name, and size_mm are required")

        self._apply_intrinsics_override(payload.get("intrinsics"))

        pose = self.pose_algo.infer(rgb_path, depth_path, class_name, size_mm)
        pose = np.asarray(pose, dtype=float)

        R = None
        t_mm = None
        if pose.size == 16:
            pose = pose.reshape(4, 4)
        if pose.shape == (4, 4):
            R = pose[:3, :3].tolist()
            t_mm = pose[:3, 3].reshape(-1).tolist()

        debug_vis = self._find_first_debug_vis()
        return {
            "pose_matrix": pose.tolist() if pose.size else None,
            "R": R,
            "t_mm": t_mm,
            "pose_source": "foundationpose",
            "debug_vis_path": str(debug_vis) if debug_vis else None,
        }

    def pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rgb_path = payload.get("rgb_path")
        depth_path = payload.get("depth_path")
        if not rgb_path or not depth_path:
            raise ValueError("rgb_path and depth_path are required")

        self._apply_intrinsics_override(payload.get("intrinsics"))

        cat = self.category_recognition({"rgb_path": rgb_path, "clean_tmp": payload.get("clean_tmp", True)})
        selected = cat.get("selected") or {}
        class_name = selected.get("class_name")
        if not class_name:
            raise RuntimeError("No class_name from category recognition")

        dim = self.dimension_measurement({
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "class_name": class_name,
        })
        size_mm = dim.get("size_mm")
        if not size_mm:
            raise RuntimeError("No size_mm from dimension measurement")

        pose = self.pose_estimation({
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "class_name": class_name,
            "size_mm": size_mm,
        })

        return {
            "category_recognition": cat,
            "dimension_measurement": dim,
            "pose_estimation": pose,
        }


class RequestHandler(BaseHTTPRequestHandler):
    service: VisionService = None

    def _send_json(self, status: int, payload: Dict[str, Any]):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON"})
            return

        try:
            if self.path == "/health":
                self._send_json(200, {"ok": True, "status": "ok"})
                return
            if self.path == "/category_recognition":
                data = self.service.category_recognition(payload)
            elif self.path == "/dimension_measurement":
                data = self.service.dimension_measurement(payload)
            elif self.path == "/pose_estimation":
                data = self.service.pose_estimation(payload)
            elif self.path == "/pipeline":
                data = self.service.pipeline(payload)
            else:
                self._send_json(404, {"ok": False, "error": f"Unknown path: {self.path}"})
                return

            self._send_json(200, {"ok": True, "data": data})
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(traceback.format_exc())
            self._send_json(500, {"ok": False, "error": err, "trace": traceback.format_exc()})

    def log_message(self, format: str, *args):
        # Reduce noisy stdout logs; keep minimal info
        sys.stdout.write("[weld_vision_server] " + (format % args) + "\n")


def main():
    parser = argparse.ArgumentParser(description="WELD vision server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--config",
        default="config/aiws_sub.yaml",
        help="Path to config yaml (relative to repo root or absolute)",
    )
    args = parser.parse_args()

    service = VisionService(Path(args.config))
    RequestHandler.service = service

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"[weld_vision_server] Listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
