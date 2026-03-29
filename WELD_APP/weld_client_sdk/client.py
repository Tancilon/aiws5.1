import json
import urllib.request
from typing import Any, Dict, Optional

import numpy as np


class WeldClientError(RuntimeError):
    pass


class WeldClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _to_jsonable(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        return obj

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(self._to_jsonable(payload)).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except Exception as exc:
            raise WeldClientError(f"Request failed: {exc}") from exc

        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise WeldClientError(f"Invalid JSON response: {exc}") from exc

        if not payload.get("ok"):
            raise WeldClientError(payload.get("error", "Unknown server error"))

        return payload.get("data", {})

    def health(self) -> Dict[str, Any]:
        return self._post("/health", {})

    def category_recognition(self, rgb_path: str, depth_path: Optional[str] = None, intrinsics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "intrinsics": intrinsics,
        }
        return self._post("/category_recognition", payload)

    def dimension_measurement(self, rgb_path: str, depth_path: str, class_name: str, intrinsics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "class_name": class_name,
            "intrinsics": intrinsics,
        }
        return self._post("/dimension_measurement", payload)

    def pose_estimation(self, rgb_path: str, depth_path: str, class_name: str, size_mm: Any, intrinsics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "class_name": class_name,
            "size_mm": size_mm,
            "intrinsics": intrinsics,
        }
        return self._post("/pose_estimation", payload)

    def pipeline(self, rgb_path: str, depth_path: str, intrinsics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "intrinsics": intrinsics,
        }
        return self._post("/pipeline", payload)
