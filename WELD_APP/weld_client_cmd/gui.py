"""
Internal GUI for algorithm pipeline validation.

Pipeline: YOLOv11-Seg -> GenPose++ -> FoundationPose
Focus: local RGB-D input, step-by-step debug, and clear visualizations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import yaml
except Exception:
    yaml = None

try:
    import OpenEXR
    import Imath
except Exception:
    OpenEXR = None
    Imath = None

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    from weld_client_sdk import WeldClient, WeldClientError
except Exception:
    WeldClient = None
    WeldClientError = Exception


RGB_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz")
DEPTH_EXTS = (".png", ".tif", ".tiff", ".npy", ".npz", ".exr")
INTR_EXTS = (".json", ".yml", ".yaml")


# ----------------------------
# Helpers
# ----------------------------

def _np_to_qimage(img: np.ndarray) -> QImage:
    """Convert numpy image to QImage (grayscale or BGR/RGB uint8)."""
    if img is None:
        raise ValueError("img is None")

    if img.ndim == 2:
        h, w = img.shape
        img_u8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        return QImage(img_u8.data, w, h, w, QImage.Format_Grayscale8).copy()

    if img.ndim == 3 and img.shape[2] in (3, 4):
        h, w, c = img.shape
        img_u8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)

        if c == 3:
            rgb = img_u8[..., ::-1].copy()
            bytes_per_line = 3 * w
            return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        rgba = img_u8[..., [2, 1, 0, 3]].copy()
        bytes_per_line = 4 * w
        return QImage(rgba.data, w, h, bytes_per_line, QImage.Format_RGBA8888).copy()

    raise ValueError(f"Unsupported image shape: {img.shape}")


def _read_image_any(path: str) -> np.ndarray:
    """Read RGB image. Returns BGR uint8 if possible."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".npy", ".npz"):
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            key = list(arr.keys())[0]
            arr = arr[key]
        return arr

    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"cv2 failed to read: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    qimg = QImage(path)
    if qimg.isNull():
        raise ValueError(f"Qt failed to read: {path}")
    qimg = qimg.convertToFormat(QImage.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
    return arr[..., ::-1].copy()


def _read_exr_depth(path: str) -> np.ndarray:
    if OpenEXR is None or Imath is None:
        raise RuntimeError("Reading .exr requires OpenEXR/Imath Python bindings")
    exr_file = OpenEXR.InputFile(str(path))
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


def _read_depth_any(path: str) -> np.ndarray:
    """Read depth from npy/npz/png/tiff/exr."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".npy", ".npz"):
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            key = list(arr.keys())[0]
            arr = arr[key]
        return arr

    if ext == ".exr":
        if cv2 is not None:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if dep is not None:
                if dep.ndim == 3:
                    dep = dep[..., 0]
                return dep
        return _read_exr_depth(path)

    if cv2 is None:
        raise RuntimeError("Depth png/tiff requires opencv-python (cv2)")
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise ValueError(f"cv2 failed to read depth: {path}")
    return dep


def _normalize_depth_for_view(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to uint8 for visualization."""
    d = depth.astype(np.float32)
    valid = d[d > 0]
    if valid.size == 0:
        return np.zeros_like(d, dtype=np.uint8)
    lo = np.percentile(valid, 2)
    hi = np.percentile(valid, 98)
    if hi <= lo:
        hi = lo + 1.0
    d = np.clip((d - lo) / (hi - lo), 0, 1)
    return (d * 255).astype(np.uint8)


def _mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """Convert a mask to binary uint8 (0/1)."""
    m = mask
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype != np.uint8:
        m = m.astype(np.float32)
        m_min = float(np.nanmin(m))
        m_max = float(np.nanmax(m))
        if 0.0 <= m_min and m_max <= 1.0:
            if float(np.nanmean(m)) > 0.5:
                return (m < 0.5).astype(np.uint8)
            return (m > 0.5).astype(np.uint8)
        return (m > 0).astype(np.uint8)
    if m.max() > 1:
        return (m > 0).astype(np.uint8)
    return m


def _overlay_mask_and_bbox(rgb_bgr: np.ndarray, mask: Optional[np.ndarray], bbox_xyxy: Optional[Tuple[int, int, int, int]], alpha: float = 0.5) -> np.ndarray:
    """Overlay mask and bbox on BGR image."""
    out = rgb_bgr.copy()
    if mask is not None:
        m = _mask_to_binary(mask)
        if cv2 is not None:
            color = np.zeros_like(out, dtype=np.uint8)
            color[..., 1] = 255
            out[m > 0] = (out[m > 0] * (1 - alpha) + color[m > 0] * alpha).astype(np.uint8)
        else:
            out[m > 0] = (out[m > 0] * (1 - alpha) + np.array([0, 255, 0], dtype=np.uint8) * alpha).astype(np.uint8)

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1 = max(0, min(out.shape[1] - 1, x1))
        x2 = max(0, min(out.shape[1] - 1, x2))
        y1 = max(0, min(out.shape[0] - 1, y1))
        y2 = max(0, min(out.shape[0] - 1, y2))
        if cv2 is not None:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            out[y1:y1 + 2, x1:x2] = (0, 0, 255)
            out[y2 - 2:y2, x1:x2] = (0, 0, 255)
            out[y1:y2, x1:x1 + 2] = (0, 0, 255)
            out[y1:y2, x2 - 2:x2] = (0, 0, 255)
    return out


def _draw_axes(rgb_bgr: np.ndarray, K: np.ndarray, R: np.ndarray, t_mm: np.ndarray, axis_len_mm: float = 60.0) -> np.ndarray:
    """Draw 3D axes projected onto the RGB image."""
    if cv2 is None:
        return rgb_bgr

    K = np.asarray(K, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    t = np.asarray(t_mm, dtype=np.float32).reshape(3, 1)
    if K.shape != (3, 3):
        return rgb_bgr

    pts_obj = np.array([
        [0, 0, 0],
        [axis_len_mm, 0, 0],
        [0, axis_len_mm, 0],
        [0, 0, axis_len_mm],
    ], dtype=np.float32).reshape(-1, 3, 1)

    pts_cam = (R @ pts_obj.reshape(-1, 3).T + t).T
    z = pts_cam[:, 2:3]
    z = np.where(z == 0, 1e-6, z)
    pts_norm = pts_cam[:, :2] / z
    pts_img = (K[:2, :2] @ pts_norm.T + K[:2, 2:3]).T

    p0 = tuple(np.round(pts_img[0]).astype(int))
    px = tuple(np.round(pts_img[1]).astype(int))
    py = tuple(np.round(pts_img[2]).astype(int))
    pz = tuple(np.round(pts_img[3]).astype(int))

    out = rgb_bgr.copy()
    cv2.line(out, p0, px, (0, 0, 255), 3)
    cv2.line(out, p0, py, (0, 255, 0), 3)
    cv2.line(out, p0, pz, (255, 0, 0), 3)
    cv2.circle(out, p0, 4, (0, 255, 255), -1)
    return out


def _load_intrinsics(path: str) -> Dict[str, Any]:
    """Load intrinsics from json/yaml."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif ext in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pyyaml not installed; cannot read yaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError("Intrinsics must be .json/.yml/.yaml")

    if "K" in data:
        K = np.asarray(data["K"], dtype=np.float32)
    else:
        fx = float(data["fx"])
        fy = float(data["fy"])
        cx = float(data["cx"])
        cy = float(data["cy"])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    intr = {"K": K}
    if "width" in data:
        intr["width"] = int(data["width"])
    if "height" in data:
        intr["height"] = int(data["height"])
    return intr


def _find_file_in_folder(folder: str, exts: Tuple[str, ...], keywords: List[str]) -> str:
    if not os.path.isdir(folder):
        return ""
    files = sorted(os.listdir(folder))
    candidates = [f for f in files if os.path.splitext(f)[1].lower() in exts]
    for kw in keywords:
        for f in candidates:
            if kw in f.lower():
                return os.path.join(folder, f)
    if candidates:
        return os.path.join(folder, candidates[0])
    return ""


def _auto_pick_from_folder(folder: str) -> Tuple[str, str, str]:
    rgb = _find_file_in_folder(folder, RGB_EXTS, ["rgb", "color", "image"])
    depth = _find_file_in_folder(folder, DEPTH_EXTS, ["depth", "dpt"])
    intr = _find_file_in_folder(folder, INTR_EXTS, ["intr", "camera", "cam"])
    return rgb, depth, intr


def _make_dummy_rgb_depth(size: Tuple[int, int] = (480, 640)) -> Tuple[np.ndarray, np.ndarray]:
    h, w = size
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = 30
    rgb[..., 1] = 30
    rgb[..., 2] = 30
    if cv2 is not None:
        cv2.rectangle(rgb, (80, 60), (w - 80, h - 60), (50, 160, 230), -1)
        cv2.circle(rgb, (w // 2, h // 2), min(h, w) // 6, (30, 220, 120), -1)
    else:
        rgb[60:h - 60, 80:w - 80] = (50, 160, 230)
    depth = np.linspace(400, 900, num=h * w, dtype=np.float32).reshape(h, w)
    return rgb, depth


# ----------------------------
# Data models
# ----------------------------

@dataclass
class PipelineState:
    rgb_path: str = ""
    depth_path: str = ""
    intrinsics_path: str = ""

    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    intrinsics: Optional[Dict[str, Any]] = None

    yolo: Optional[Dict[str, Any]] = None
    size: Optional[Dict[str, Any]] = None
    pose: Optional[Dict[str, Any]] = None

    cad_model_path: str = ""
    step_index: int = 0


# ----------------------------
# Algorithm client interface
# ----------------------------

class AlgorithmClient:
    """Interface for the pipeline backend (SDK or REST)."""

    def run_yolo(self, rgb_path: str, depth_path: Optional[str], intrinsics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def run_genpose(self, rgb_path: str, depth_path: Optional[str], yolo_result: Dict[str, Any], intrinsics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def run_foundationpose(
        self,
        rgb_path: str,
        depth_path: Optional[str],
        yolo_result: Dict[str, Any],
        size_result: Dict[str, Any],
        cad_model_path: str,
        intrinsics: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class DummyAlgorithmClient(AlgorithmClient):
    """Dummy data flow for front-end testing."""

    def run_yolo(self, rgb_path: str, depth_path: Optional[str], intrinsics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if rgb_path and os.path.isfile(rgb_path):
            rgb = _read_image_any(rgb_path)
        else:
            rgb, _ = _make_dummy_rgb_depth()
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h * 0.52, w * 0.5
        ry, rx = h * 0.25, w * 0.2
        mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) < 1.0
        bbox = (int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry))
        return {
            "class_id": 1,
            "class_name": "dummy_part",
            "score": 0.88,
            "mask": mask.astype(np.uint8),
            "bbox_xyxy": bbox,
        }

    def run_genpose(self, rgb_path: str, depth_path: Optional[str], yolo_result: Dict[str, Any], intrinsics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "size_mm": (120.0, 60.0, 35.0),
            "size_source": "dummy",
        }

    def run_foundationpose(
        self,
        rgb_path: str,
        depth_path: Optional[str],
        yolo_result: Dict[str, Any],
        size_result: Dict[str, Any],
        cad_model_path: str,
        intrinsics: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        R = np.eye(3, dtype=np.float32)
        t = np.array([0.0, 0.0, 500.0], dtype=np.float32)
        return {
            "R": R,
            "t_mm": t,
            "pose_source": "dummy",
        }


class SdkAlgorithmClient(AlgorithmClient):
    """HTTP client backed by weld_client_sdk.WeldClient."""

    def __init__(self, server_url: str, timeout: float = 180.0):
        if WeldClient is None:
            raise RuntimeError("weld_client_sdk is not available")
        self.client = WeldClient(server_url, timeout=timeout)

    def _intrinsics_payload(self, intrinsics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not intrinsics:
            return None
        if "K" in intrinsics:
            k = np.asarray(intrinsics["K"], dtype=float)
            data = {
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
            }
            if "width" in intrinsics:
                data["width"] = float(intrinsics["width"])
            if "height" in intrinsics:
                data["height"] = float(intrinsics["height"])
            return data
        return intrinsics

    def run_yolo(self, rgb_path: str, depth_path: Optional[str], intrinsics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        resp = self.client.category_recognition(rgb_path, depth_path, self._intrinsics_payload(intrinsics))
        selected = resp.get("selected") or {}
        mask = None
        bbox = None
        mask_path = resp.get("mask_path")
        if mask_path and os.path.isfile(mask_path):
            try:
                mask_raw = _read_depth_any(mask_path)
                mask = _mask_to_binary(mask_raw)
                ys, xs = np.where(mask > 0)
                if ys.size and xs.size:
                    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            except Exception:
                mask = None
        result = {
            "class_id": selected.get("class_id"),
            "class_name": selected.get("class_name"),
            "score": selected.get("confidence"),
            "mask": mask,
            "bbox_xyxy": bbox,
            "mask_path": mask_path,
        }
        # Keep mask/bbox optional; GUI will handle None values.
        return result

    def run_genpose(self, rgb_path: str, depth_path: Optional[str], yolo_result: Dict[str, Any], intrinsics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        class_name = yolo_result.get("class_name")
        if not class_name:
            raise ValueError("class_name missing from YOLO result")
        resp = self.client.dimension_measurement(rgb_path, depth_path, class_name, self._intrinsics_payload(intrinsics))
        return {
            "size_mm": resp.get("size_mm"),
            "confidence": resp.get("confidence"),
            "size_source": resp.get("size_source", "genpose2"),
        }

    def run_foundationpose(
        self,
        rgb_path: str,
        depth_path: Optional[str],
        yolo_result: Dict[str, Any],
        size_result: Dict[str, Any],
        cad_model_path: str,
        intrinsics: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        class_name = yolo_result.get("class_name")
        if not class_name:
            raise ValueError("class_name missing from YOLO result")
        size_mm = size_result.get("size_mm") if isinstance(size_result, dict) else size_result
        resp = self.client.pose_estimation(rgb_path, depth_path, class_name, size_mm, self._intrinsics_payload(intrinsics))
        return {
            "R": resp.get("R"),
            "t_mm": resp.get("t_mm"),
            "pose_source": resp.get("pose_source", "foundationpose"),
            "debug_vis_path": resp.get("debug_vis_path"),
        }


# ----------------------------
# Widgets
# ----------------------------

class ImageView(QtWidgets.QLabel):
    """A QLabel that scales pixmap with aspect ratio."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color:#111; border:1px solid #333; color:#bbb;")
        if title:
            self.setText(title)

    def set_image_np(self, img: np.ndarray):
        qimg = _np_to_qimage(img)
        self._pixmap = QPixmap.fromImage(qimg)
        self._apply_scaled()

    def set_image_path(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            raise ValueError(f"Qt failed to read image: {path}")
        self._pixmap = pix
        self._apply_scaled()

    def clear_image(self, text: str = ""):
        self._pixmap = None
        self.setPixmap(QPixmap())
        if text:
            self.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled()

    def _apply_scaled(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


# ----------------------------
# UI
# ----------------------------

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 900)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        # Left panel
        self.left_panel = QtWidgets.QWidget(self.centralwidget)
        self.left_panel.setMinimumWidth(380)
        self.left_layout = QtWidgets.QVBoxLayout(self.left_panel)
        self.left_layout.setSpacing(10)

        # Files group
        self.grp_files = QtWidgets.QGroupBox("Inputs")
        self.files_layout = QtWidgets.QFormLayout(self.grp_files)

        self.folder_path = QtWidgets.QLineEdit()
        self.btn_folder = QtWidgets.QPushButton("Pick Folder...")
        self.files_layout.addRow("Folder", self._hbox(self.folder_path, self.btn_folder))

        self.rgb_path = QtWidgets.QLineEdit()
        self.btn_rgb = QtWidgets.QPushButton("Pick RGB...")
        self.depth_path = QtWidgets.QLineEdit()
        self.btn_depth = QtWidgets.QPushButton("Pick Depth...")
        self.intri_path = QtWidgets.QLineEdit()
        self.btn_intri = QtWidgets.QPushButton("Pick Intrinsics...")
        self.btn_dummy = QtWidgets.QPushButton("Load Dummy Data")

        self.files_layout.addRow("RGB", self._hbox(self.rgb_path, self.btn_rgb))
        self.files_layout.addRow("Depth", self._hbox(self.depth_path, self.btn_depth))
        self.files_layout.addRow("Intrinsics", self._hbox(self.intri_path, self.btn_intri))
        self.files_layout.addRow("", self.btn_dummy)

        # Run group
        self.grp_run = QtWidgets.QGroupBox("Pipeline")
        self.run_layout = QtWidgets.QGridLayout(self.grp_run)

        self.btn_step = QtWidgets.QPushButton("Run Next Step")
        self.btn_run_yolo = QtWidgets.QPushButton("Run YOLOv11-Seg")
        self.btn_run_genpose = QtWidgets.QPushButton("Run GenPose++")
        self.btn_run_pose = QtWidgets.QPushButton("Run FoundationPose")
        self.btn_run_all = QtWidgets.QPushButton("Run All")
        self.btn_clear = QtWidgets.QPushButton("Reset Results")
        self.lbl_step = QtWidgets.QLabel("Next: YOLOv11-Seg")

        self.run_layout.addWidget(self.btn_step, 0, 0, 1, 2)
        self.run_layout.addWidget(self.btn_run_yolo, 1, 0, 1, 2)
        self.run_layout.addWidget(self.btn_run_genpose, 2, 0, 1, 2)
        self.run_layout.addWidget(self.btn_run_pose, 3, 0, 1, 2)
        self.run_layout.addWidget(self.btn_run_all, 4, 0, 1, 1)
        self.run_layout.addWidget(self.btn_clear, 4, 1, 1, 1)
        self.run_layout.addWidget(self.lbl_step, 5, 0, 1, 2)

        # Result summary
        self.grp_summary = QtWidgets.QGroupBox("Summary")
        self.summary_layout = QtWidgets.QFormLayout(self.grp_summary)
        self.lbl_class = QtWidgets.QLabel("-")
        self.lbl_score = QtWidgets.QLabel("-")
        self.lbl_size = QtWidgets.QLabel("-")
        self.lbl_cad = QtWidgets.QLabel("-")
        self.lbl_pose = QtWidgets.QLabel("-")

        self.lbl_cad.setWordWrap(True)
        self.lbl_pose.setWordWrap(True)

        self.summary_layout.addRow("Class", self.lbl_class)
        self.summary_layout.addRow("Score", self.lbl_score)
        self.summary_layout.addRow("Size (mm)", self.lbl_size)
        self.summary_layout.addRow("CAD", self.lbl_cad)
        self.summary_layout.addRow("Pose", self.lbl_pose)

        # Log
        self.grp_log = QtWidgets.QGroupBox("Log")
        self.log_layout = QtWidgets.QVBoxLayout(self.grp_log)
        self.text_log = QtWidgets.QPlainTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setMaximumBlockCount(5000)
        self.log_layout.addWidget(self.text_log)

        self.left_layout.addWidget(self.grp_files)
        self.left_layout.addWidget(self.grp_run)
        self.left_layout.addWidget(self.grp_summary)
        self.left_layout.addWidget(self.grp_log)
        self.left_layout.setStretch(0, 0)
        self.left_layout.setStretch(1, 0)
        self.left_layout.setStretch(2, 0)
        self.left_layout.setStretch(3, 1)

        # Right panel: tabs
        self.tabs = QtWidgets.QTabWidget(self.centralwidget)

        # Tab 1 - Input
        self.tab_input = QtWidgets.QWidget()
        self.tab_input_layout = QtWidgets.QGridLayout(self.tab_input)
        self.view_rgb = ImageView("RGB")
        self.view_depth = ImageView("Depth")
        self.tab_input_layout.addWidget(self.view_rgb, 0, 0)
        self.tab_input_layout.addWidget(self.view_depth, 0, 1)
        self.tabs.addTab(self.tab_input, "Input")

        # Tab 2 - YOLO
        self.tab_yolo = QtWidgets.QWidget()
        self.tab_yolo_layout = QtWidgets.QGridLayout(self.tab_yolo)
        self.view_yolo_overlay = ImageView("YOLO Overlay")
        self.view_yolo_mask = ImageView("Mask")
        self.text_yolo = QtWidgets.QPlainTextEdit()
        self.text_yolo.setReadOnly(True)
        self.tab_yolo_layout.addWidget(self.view_yolo_overlay, 0, 0)
        self.tab_yolo_layout.addWidget(self.view_yolo_mask, 0, 1)
        self.tab_yolo_layout.addWidget(self.text_yolo, 1, 0, 1, 2)
        self.tab_yolo_layout.setRowStretch(0, 3)
        self.tab_yolo_layout.setRowStretch(1, 1)
        self.tabs.addTab(self.tab_yolo, "YOLOv11-Seg")

        # Tab 3 - GenPose
        self.tab_size = QtWidgets.QWidget()
        self.tab_size_layout = QtWidgets.QVBoxLayout(self.tab_size)
        self.text_size = QtWidgets.QPlainTextEdit()
        self.text_size.setReadOnly(True)
        self.tab_size_layout.addWidget(self.text_size)
        self.tabs.addTab(self.tab_size, "GenPose++")

        # Tab 4 - FoundationPose
        self.tab_pose = QtWidgets.QWidget()
        self.tab_pose_layout = QtWidgets.QGridLayout(self.tab_pose)
        self.view_pose_overlay = ImageView("Pose Overlay")
        self.text_pose = QtWidgets.QPlainTextEdit()
        self.text_pose.setReadOnly(True)
        self.tab_pose_layout.addWidget(self.view_pose_overlay, 0, 0)
        self.tab_pose_layout.addWidget(self.text_pose, 1, 0)
        self.tab_pose_layout.setRowStretch(0, 3)
        self.tab_pose_layout.setRowStretch(1, 1)
        self.tabs.addTab(self.tab_pose, "FoundationPose")

        # Compose
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.tabs)
        self.main_layout.setStretch(0, 0)
        self.main_layout.setStretch(1, 1)

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self._retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def _hbox(self, w1, w2):
        box = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(w1, 1)
        lay.addWidget(w2, 0)
        return box

    def _retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("Algorithm Pipeline GUI (YOLOv11-Seg -> GenPose++ -> FoundationPose)")


# ----------------------------
# Controller
# ----------------------------

class Controller(QtCore.QObject):
    def __init__(self, ui: Ui_MainWindow, client: Optional[AlgorithmClient] = None):
        super().__init__()
        self.ui = ui
        self.client = client or DummyAlgorithmClient()
        self.state = PipelineState()

        # wire signals
        self.ui.btn_folder.clicked.connect(self.on_pick_folder)
        self.ui.btn_rgb.clicked.connect(self.on_pick_rgb)
        self.ui.btn_depth.clicked.connect(self.on_pick_depth)
        self.ui.btn_intri.clicked.connect(self.on_pick_intrinsics)
        self.ui.btn_dummy.clicked.connect(self.on_load_dummy)

        self.ui.btn_step.clicked.connect(self.run_step)
        self.ui.btn_run_yolo.clicked.connect(self.run_yolo)
        self.ui.btn_run_genpose.clicked.connect(self.run_genpose)
        self.ui.btn_run_pose.clicked.connect(self.run_pose)
        self.ui.btn_run_all.clicked.connect(self.run_all)
        self.ui.btn_clear.clicked.connect(self.reset_results)

        self._set_buttons_enabled(False)
        self._log("GUI ready. Load RGB to start.")

    # -------- UI utilities --------

    def _log(self, msg: str):
        ts = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.ui.text_log.appendPlainText(f"[{ts}] {msg}")
        self.ui.statusbar.showMessage(msg, 4000)

    def _set_buttons_enabled(self, has_rgb: bool):
        self.ui.btn_step.setEnabled(has_rgb)
        self.ui.btn_run_yolo.setEnabled(has_rgb)
        self.ui.btn_run_genpose.setEnabled(has_rgb)
        self.ui.btn_run_pose.setEnabled(has_rgb)
        self.ui.btn_run_all.setEnabled(has_rgb)

    def _set_step_label(self):
        labels = ["Next: YOLOv11-Seg", "Next: GenPose++", "Next: FoundationPose", "Pipeline complete"]
        idx = max(0, min(self.state.step_index, len(labels) - 1))
        self.ui.lbl_step.setText(labels[idx])

    def _reset_steps(self):
        self.state.step_index = 0
        self._set_step_label()

    @staticmethod
    def _format_size_display(size_vals: Any) -> Tuple[Optional[List[float]], bool]:
        try:
            arr = np.asarray(size_vals, dtype=float).reshape(-1)
        except Exception:
            return None, False
        if arr.size != 3:
            return None, False
        converted = False
        if np.nanmax(arr) <= 10.0:
            arr = arr * 1000.0
            converted = True
        return arr.tolist(), converted

    @staticmethod
    def _format_translation_display(t_vals: Any) -> Tuple[Optional[List[float]], bool]:
        try:
            arr = np.asarray(t_vals, dtype=float).reshape(-1)
        except Exception:
            return None, False
        if arr.size != 3:
            return None, False
        converted = False
        if np.nanmax(np.abs(arr)) <= 10.0:
            arr = arr * 1000.0
            converted = True
        return arr.tolist(), converted

    def _update_summary(self):
        y = self.state.yolo
        s = self.state.size
        p = self.state.pose

        if y:
            cls = y.get("class_name", "-")
            cid = y.get("class_id", None)
            if cid is not None:
                cls = f"{cls} (id={cid})"
            self.ui.lbl_class.setText(cls)
            self.ui.lbl_score.setText(f"{float(y.get('score', 0)):.3f}")
        else:
            self.ui.lbl_class.setText("-")
            self.ui.lbl_score.setText("-")

        if s and "size_mm" in s:
            size_disp, converted = self._format_size_display(s["size_mm"])
            if size_disp:
                L, W, H = size_disp
                self.ui.lbl_size.setText(f"{L:.1f}, {W:.1f}, {H:.1f} mm")
            else:
                self.ui.lbl_size.setText("-")
        else:
            self.ui.lbl_size.setText("-")

        self.ui.lbl_cad.setText(self.state.cad_model_path or "-")

        if p and "t_mm" in p:
            t_disp, _ = self._format_translation_display(p["t_mm"])
            if t_disp:
                t = np.asarray(t_disp).reshape(-1)
                self.ui.lbl_pose.setText(f"t(mm)=[{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}]")
            else:
                self.ui.lbl_pose.setText("-")
        else:
            self.ui.lbl_pose.setText("-")

    # -------- data loading --------

    def on_pick_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select data folder", self.state.rgb_path or os.getcwd()
        )
        if not folder:
            return
        self.ui.folder_path.setText(folder)
        rgb_path, depth_path, intr_path = _auto_pick_from_folder(folder)

        if rgb_path:
            self._load_rgb(rgb_path)
        if depth_path:
            self._load_depth(depth_path)
        if intr_path:
            self._load_intrinsics(intr_path)

        if not rgb_path:
            self._log("No RGB auto-detected in folder. Please pick manually.")
        if depth_path:
            self._log(f"Auto-loaded depth: {depth_path}")
        if intr_path:
            self._log(f"Auto-loaded intrinsics: {intr_path}")

    def on_pick_rgb(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select RGB image",
            self.state.rgb_path or os.getcwd(),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.npy *.npz);;All Files (*)",
        )
        if not path:
            return
        self._load_rgb(path)

    def on_pick_depth(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select Depth (optional)",
            self.state.depth_path or os.getcwd(),
            "Depth (*.png *.tif *.tiff *.npy *.npz *.exr);;All Files (*)",
        )
        if not path:
            return
        self._load_depth(path)

    def on_pick_intrinsics(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select Intrinsics (optional)",
            self.state.intrinsics_path or os.getcwd(),
            "Intrinsics (*.json *.yml *.yaml);;All Files (*)",
        )
        if not path:
            return
        self._load_intrinsics(path)

    def on_load_dummy(self):
        rgb, depth = _make_dummy_rgb_depth()
        self.state.rgb = rgb
        self.state.depth = depth
        self.state.rgb_path = "<dummy>"
        self.state.depth_path = "<dummy>"
        self.ui.rgb_path.setText(self.state.rgb_path)
        self.ui.depth_path.setText(self.state.depth_path)
        self.ui.view_rgb.set_image_np(rgb)
        self.ui.view_depth.set_image_np(_normalize_depth_for_view(depth))
        h, w = rgb.shape[:2]
        self.state.intrinsics = {"K": np.array([[600.0, 0.0, w / 2.0], [0.0, 600.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)}
        self.state.intrinsics_path = "<dummy>"
        self.ui.intri_path.setText(self.state.intrinsics_path)
        self._reset_steps()
        self.reset_results(keep_inputs=True)
        self._set_buttons_enabled(True)
        self._log("Dummy data loaded.")

    def _load_rgb(self, path: str):
        try:
            rgb = _read_image_any(path)
            self.state.rgb_path = path
            self.state.rgb = rgb
            self.ui.rgb_path.setText(path)
            self.ui.view_rgb.set_image_np(rgb)
            self._reset_steps()
            self.reset_results(keep_inputs=True)
            self._set_buttons_enabled(True)
            self._log(f"Loaded RGB: {path} | shape={rgb.shape} dtype={rgb.dtype}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Load RGB failed", str(e))

    def _load_depth(self, path: str):
        try:
            depth = _read_depth_any(path)
            self.state.depth_path = path
            self.state.depth = depth
            self.ui.depth_path.setText(path)
            view = _normalize_depth_for_view(depth)
            self.ui.view_depth.set_image_np(view)
            self._log(f"Loaded Depth: {path} | shape={depth.shape} dtype={depth.dtype}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Load Depth failed", str(e))

    def _load_intrinsics(self, path: str):
        try:
            intr = _load_intrinsics(path)
            self.state.intrinsics_path = path
            self.state.intrinsics = intr
            self.ui.intri_path.setText(path)
            K = intr["K"]
            self._log(f"Loaded intrinsics: fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Load intrinsics failed", str(e))

    # -------- pipeline steps --------

    def run_yolo(self):
        if self.state.rgb is None:
            return
        try:
            self._log("Running YOLOv11-Seg...")
            y = self.client.run_yolo(self.state.rgb_path, self.state.depth_path, self.state.intrinsics)
            self.state.yolo = y

            rgb = self.state.rgb
            mask = y.get("mask")
            bbox = y.get("bbox_xyxy")
            overlay = _overlay_mask_and_bbox(rgb, mask, bbox, alpha=0.55)
            self.ui.view_yolo_overlay.set_image_np(overlay)

            if mask is not None:
                m8 = (mask > 0).astype(np.uint8) * 255
                self.ui.view_yolo_mask.set_image_np(m8)
            safe = {k: v for k, v in y.items() if k != "mask"}
            self.ui.text_yolo.setPlainText(json.dumps(safe, indent=2, default=str))

            self.state.cad_model_path = self._resolve_cad_path(y)
            self._update_summary()
            self.ui.tabs.setCurrentWidget(self.ui.tab_yolo)
            self.state.step_index = max(self.state.step_index, 1)
            self._set_step_label()
            self._log("YOLO done.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "YOLO failed", str(e))

    def run_genpose(self):
        if self.state.rgb is None:
            return
        if self.state.yolo is None:
            QtWidgets.QMessageBox.warning(None, "Need YOLO", "Please run YOLO first.")
            return
        try:
            self._log("Running GenPose++ (size estimation)...")
            s = self.client.run_genpose(self.state.rgb_path, self.state.depth_path, self.state.yolo, self.state.intrinsics)
            self.state.size = s
            s_display = dict(s) if isinstance(s, dict) else {"size_mm": s}
            if "size_mm" in s_display:
                size_disp, converted = self._format_size_display(s_display["size_mm"])
                if size_disp:
                    s_display["size_mm"] = size_disp
                    if converted:
                        s_display["size_unit"] = "mm (converted from m)"
            self.ui.text_size.setPlainText(json.dumps(s_display, indent=2, default=str))
            self._update_summary()
            self.ui.tabs.setCurrentWidget(self.ui.tab_size)
            self.state.step_index = max(self.state.step_index, 2)
            self._set_step_label()
            self._log("GenPose++ done.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "GenPose++ failed", str(e))

    def run_pose(self):
        if self.state.rgb is None:
            return
        if self.state.yolo is None:
            QtWidgets.QMessageBox.warning(None, "Need YOLO", "Please run YOLO first.")
            return
        if self.state.size is None:
            QtWidgets.QMessageBox.warning(None, "Need GenPose++", "Please run GenPose++ first.")
            return
        try:
            self._log("Running FoundationPose (6D pose)...")
            p = self.client.run_foundationpose(
                self.state.rgb_path,
                self.state.depth_path,
                self.state.yolo,
                self.state.size,
                self.state.cad_model_path,
                self.state.intrinsics,
            )
            self.state.pose = p

            t_display, converted_t = self._format_translation_display(p.get("t_mm"))
            pose_txt = {
                "R": np.asarray(p.get("R")).tolist() if p.get("R") is not None else None,
                "t_mm": t_display if t_display is not None else (np.asarray(p.get("t_mm")).reshape(-1).tolist() if p.get("t_mm") is not None else None),
                "pose_source": p.get("pose_source", ""),
                "cad_model": self.state.cad_model_path,
            }
            if converted_t:
                pose_txt["t_unit"] = "mm (converted from m)"
            self.ui.text_pose.setPlainText(json.dumps(pose_txt, indent=2))

            rgb = self.state.rgb
            debug_vis_path = p.get("debug_vis_path")
            if debug_vis_path and os.path.isfile(debug_vis_path):
                try:
                    self.ui.view_pose_overlay.set_image_path(debug_vis_path)
                    overlay = None
                except Exception:
                    overlay = rgb
            elif self.state.intrinsics and p.get("R") is not None and p.get("t_mm") is not None:
                K = self.state.intrinsics["K"]
                t_for_draw = t_display if t_display is not None else p["t_mm"]
                overlay = _draw_axes(rgb, K, p["R"], t_for_draw, axis_len_mm=60.0)
            else:
                overlay = rgb
            if overlay is not None:
                self.ui.view_pose_overlay.set_image_np(overlay)

            self._update_summary()
            self.ui.tabs.setCurrentWidget(self.ui.tab_pose)
            self.state.step_index = max(self.state.step_index, 3)
            self._set_step_label()
            self._log("FoundationPose done.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "FoundationPose failed", str(e))

    def run_all(self):
        self.run_yolo()
        if self.state.yolo is None:
            return
        self.run_genpose()
        if self.state.size is None:
            return
        self.run_pose()

    def run_step(self):
        if self.state.step_index == 0:
            self.run_yolo()
            return
        if self.state.step_index == 1:
            self.run_genpose()
            return
        if self.state.step_index == 2:
            self.run_pose()
            return
        self._log("Pipeline already complete. Reset results to run again.")

    def reset_results(self, keep_inputs: bool = False):
        if keep_inputs:
            rgb = self.state.rgb
            depth = self.state.depth
            intr = self.state.intrinsics
            rgb_path = self.state.rgb_path
            depth_path = self.state.depth_path
            intr_path = self.state.intrinsics_path
        else:
            rgb = depth = intr = None
            rgb_path = depth_path = intr_path = ""

        self.state = PipelineState(
            rgb_path=rgb_path,
            depth_path=depth_path,
            intrinsics_path=intr_path,
            rgb=rgb,
            depth=depth,
            intrinsics=intr,
        )

        if rgb is None:
            self.ui.view_rgb.clear_image("RGB")
            self.ui.rgb_path.clear()
        if depth is None:
            self.ui.view_depth.clear_image("Depth")
            self.ui.depth_path.clear()
        if intr is None:
            self.ui.intri_path.clear()

        self.ui.view_yolo_overlay.clear_image("YOLO Overlay")
        self.ui.view_yolo_mask.clear_image("Mask")
        self.ui.view_pose_overlay.clear_image("Pose Overlay")
        self.ui.text_yolo.clear()
        self.ui.text_size.clear()
        self.ui.text_pose.clear()
        self._update_summary()
        self._reset_steps()
        self._set_buttons_enabled(rgb is not None)
        self._log("Results reset.")

    # -------- helpers --------

    def _resolve_cad_path(self, yolo_result: Dict[str, Any]) -> str:
        cls = yolo_result.get("class_name", "unknown")
        size = self.state.size.get("size_mm") if self.state.size else None
        if size:
            L, W, H = size
            name = f"{cls}_{int(L)}x{int(W)}x{int(H)}.stl"
        else:
            name = f"{cls}.stl"
        return os.path.join("./cad_models", name)


# ----------------------------
# App entry
# ----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)


def main():
    parser = argparse.ArgumentParser(description="Algorithm Pipeline GUI")
    parser.add_argument("--server", default=os.environ.get("WELD_SERVER_URL"))
    parser.add_argument("--timeout", type=float, default=180.0)
    args, _ = parser.parse_known_args()

    client = None
    if args.server:
        try:
            client = SdkAlgorithmClient(args.server, timeout=args.timeout)
            print(f"[GUI] Using server backend: {args.server}")
        except Exception as exc:
            print(f"[GUI] Failed to init server client ({exc}), falling back to dummy.")

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    _ = Controller(win.ui, client=client)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
