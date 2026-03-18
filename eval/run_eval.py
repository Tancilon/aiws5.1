#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.pose_align_metric import PoseAlignmentMetricEvaluator


METRIC_NOTES = {
    "Acc_cls": "统计类别识别结果与样本标注 class_name 一致的比例，用于评价 YOLOv11 分割阶段的类别判断能力。",
    "Conf_cls": "统计工件型号识别阶段输出的置信度平均值，反映类别识别结果的可信程度。",
    "T_cls": "统计 YOLOv11 分割阶段单样本平均运行时间，反映类别识别阶段的推理效率。",
    "AE_dim": "使用 GenPose2 原始尺寸预测结果与标注尺寸逐维做差并取绝对值，按 [x, y, z] 三个方向报告平均误差。",
    "RE_dim": "使用 GenPose2 原始尺寸预测结果计算尺寸相对误差，按 [x, y, z] 三个方向报告平均值。",
    "ME_dim": "对 GenPose2 原始尺寸预测的三个维度绝对误差求平均后，再对所有有效样本求平均，作为尺寸测量综合误差指标。",
    "Conf_dim": "统计尺寸测量阶段输出的置信度平均值；在 query_mode=True 时对应查询匹配置信度。",
    "Succ_query_dim": "统计查询匹配后的尺寸与样本真实尺寸在可忽略浮点误差下相同的样本比例。",
    "T_dim": "统计 GenPose2 阶段单样本平均运行时间，反映尺寸测量阶段的推理效率。",
    "T_pose": "由于缺少真实工件位姿标注，FoundationPose 阶段不采用位姿精度误差，而以单样本平均运行时间作为主要评价指标。",
    "Succ_pose": "统计 FoundationPose 是否成功输出合法 4x4 位姿矩阵的比例，用于反映姿态估计阶段运行稳定性。",
    "pose_align_cover_rate": "统计渲染模板有效像素中，与观测深度和前景 mask 同时重叠的比例，用于评价位姿与观测轮廓的对齐程度。",
    "pose_align_avg_dist_mm": "在观测与渲染共同可见且深度误差小于 80mm 的像素上，统计平均深度差，单位为毫米。",
    "t_pose_align": "统计新增位姿对齐评测阶段单样本平均运行时间，反映该后处理指标的执行开销。",
    "T_all": "从输入 RGB、Depth 图像开始，到输出类别、尺寸、位姿结果为止，统计整条流程单样本平均运行时间。",
    "Succ_all": "统计整条流程无报错且成功输出完整结果的样本比例，用于评价系统整体稳定性。",
}

METRIC_NAMES_ZH = {
    "Acc_cls": "类别识别准确率",
    "Conf_cls": "类别识别平均置信度",
    "T_cls": "类别识别平均耗时",
    "AE_dim": "尺寸测量绝对误差",
    "RE_dim": "尺寸测量相对误差",
    "ME_dim": "尺寸测量平均误差",
    "Conf_dim": "尺寸测量平均置信度",
    "Succ_query_dim": "查询尺寸匹配成功率",
    "T_dim": "尺寸测量平均耗时",
    "T_pose": "姿态估计平均耗时",
    "Succ_pose": "姿态估计成功率",
    "pose_align_cover_rate": "位姿对齐覆盖率",
    "pose_align_avg_dist_mm": "位姿对齐平均深度差",
    "t_pose_align": "位姿对齐评测平均耗时",
    "T_all": "端到端平均耗时",
    "Succ_all": "端到端成功率",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation for AIWS pipeline.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "aiws_sub.yaml"),
        help="Path to the OmegaConf config file.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(REPO_ROOT / "data" / "aiws5.1-dataset-test"),
        help="Directory containing *_color.png, *_depth.exr and *_meta.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for evaluation outputs. Defaults to eval/results/<timestamp>.",
    )
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="GPU id forwarded to CUDA_VISIBLE_DEVICES on the host side.",
    )
    parser.add_argument(
        "--query-match-rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance used by Succ_query_dim when comparing queried size with ground truth.",
    )
    parser.add_argument(
        "--query-match-atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance used by Succ_query_dim when comparing queried size with ground truth.",
    )
    parser.add_argument(
        "--keep-runtime-artifacts",
        action="store_true",
        help="Keep per-sample runtime directories under the evaluation output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate the first N samples when > 0.",
    )
    return parser.parse_args()


def normalize_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def clean_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_runtime_cfg(base_cfg: OmegaConf, tmp_dir: Path) -> OmegaConf:
    cfg = copy.deepcopy(base_cfg)
    tmp_dir = tmp_dir.resolve()
    cfg.tmp_dir = str(tmp_dir)
    cfg.category_recognition.tmp_output_path = str(tmp_dir)
    cfg.dimension_measurement.data_path = str(tmp_dir / "gendata")
    cfg.dimension_measurement.tmp_output_path = str(tmp_dir)
    cfg.pose_estimation.data_path = str(tmp_dir / "foundationpose")
    cfg.pose_estimation.tmp_output_path = str(tmp_dir)
    return cfg


def discover_samples(dataset_dir: Path) -> List[str]:
    color_ids = {
        path.name.split("_", 1)[0]
        for path in dataset_dir.glob("*_color.png")
        if "_" in path.name
    }
    depth_ids = {
        path.name.split("_", 1)[0]
        for path in dataset_dir.glob("*_depth.exr")
        if "_" in path.name
    }
    meta_ids = {
        path.name.split("_", 1)[0]
        for path in dataset_dir.glob("*_meta.json")
        if "_" in path.name
    }

    shared_ids = sorted(color_ids & depth_ids & meta_ids)
    if not shared_ids:
        raise RuntimeError(f"No complete samples found under {dataset_dir}")

    incomplete = sorted((color_ids | depth_ids | meta_ids) - set(shared_ids))
    if incomplete:
        raise RuntimeError(
            f"Found incomplete sample ids without full color/depth/meta triplets: {incomplete}"
        )

    return shared_ids


def load_meta(meta_path: Path) -> Dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float_list(array: Sequence[float]) -> List[float]:
    return [float(x) for x in array]


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def vector_mean_or_none(values: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return arr.mean(axis=0).astype(float).tolist()


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def serialize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def format_duration(seconds: float) -> str:
    total_milliseconds = int(round(float(seconds) * 1000.0))
    hours, remainder = divmod(total_milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def find_single_file(directory: Path, description: str) -> Path:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"{description} directory not found: {directory}")

    candidates = sorted(path for path in directory.iterdir() if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"{description} not found in {directory}")
    if len(candidates) != 1:
        raise RuntimeError(f"{description} count abnormal in {directory}: {len(candidates)}")
    return candidates[0]


def evaluate_pose_alignment(
    sample_id: str,
    depth_path: Path,
    pose_array: np.ndarray,
    pose_estimation: Any,
    pose_align_metric: PoseAlignmentMetricEvaluator,
) -> Dict[str, float]:
    data_path = Path(pose_estimation.data_path)
    cam_k_path = data_path / "cam_K.txt"
    if not cam_k_path.exists():
        raise FileNotFoundError(f"Pose alignment cam_K not found: {cam_k_path}")

    mesh_path = Path(getattr(pose_estimation, "mesh_file_path", ""))
    if not mesh_path.exists():
        raise FileNotFoundError(f"Pose alignment mesh not found: {mesh_path}")

    mask_path = find_single_file(data_path / "masks", "Pose alignment mask")
    camera_matrix = pose_align_metric.load_camera_intrinsic(cam_k_path)
    result = pose_align_metric.evaluate(
        obs_depth_path=depth_path,
        mesh_path=mesh_path,
        pose=pose_array,
        camera_intrinsics=camera_matrix,
        mask_path=mask_path,
        output_dir=None,
        rendered_depth_filename=f"{sample_id}_pose_align_rendered.exr",
    )
    return {
        "pose_align_cover_rate": float(result["obs_point_cloud_cover_rate"]),
        "pose_align_avg_dist_mm": float(result["avg_dist"]),
    }


def write_per_sample_csv(output_path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "rgb_path",
        "depth_path",
        "meta_path",
        "gt_class_name",
        "gt_dimensions",
        "pred_class_name",
        "pred_class_id",
        "pred_class_confidence",
        "class_correct",
        "cls_success",
        "t_cls",
        "pred_dimensions",
        "pred_dimensions_raw",
        "pred_dimensions_query",
        "pose_input_dimensions",
        "dim_confidence",
        "abs_error_dim",
        "rel_error_dim",
        "mean_abs_error_dim",
        "query_match_dim",
        "dim_success",
        "t_dim",
        "pose",
        "pose_success",
        "t_pose",
        "pose_align_cover_rate",
        "pose_align_avg_dist_mm",
        "pose_align_success",
        "t_pose_align",
        "all_success",
        "t_all",
        "error_stage",
        "error_message",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: serialize_csv_value(record.get(key)) for key in fieldnames})


def write_metrics_csv(output_path: Path, summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["indicator", "name_zh", "value", "note"])
        writer.writeheader()
        for key in (
            "Acc_cls",
            "Conf_cls",
            "T_cls",
            "AE_dim",
            "RE_dim",
            "ME_dim",
            "Conf_dim",
            "Succ_query_dim",
            "T_dim",
            "T_pose",
            "Succ_pose",
            "pose_align_cover_rate",
            "pose_align_avg_dist_mm",
            "t_pose_align",
            "T_all",
            "Succ_all",
        ):
            writer.writerow(
                {
                    "indicator": key,
                    "name_zh": METRIC_NAMES_ZH[key],
                    "value": serialize_csv_value(metrics.get(key)),
                    "note": METRIC_NOTES[key],
                }
            )


def build_summary(
    args: argparse.Namespace,
    dataset_dir: Path,
    output_dir: Path,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    total_samples = len(records)
    cls_attempted = total_samples
    cls_success_records = [r for r in records if r.get("cls_success")]
    cls_conf_records = [r for r in cls_success_records if r.get("pred_class_confidence") is not None]
    dim_attempted_records = [r for r in records if r.get("t_dim") is not None]
    pose_attempted_records = [r for r in records if r.get("t_pose") is not None]
    dim_success_records = [r for r in records if r.get("dim_success")]
    dim_conf_records = [r for r in dim_success_records if r.get("dim_confidence") is not None]
    query_dim_records = [r for r in dim_success_records if r.get("query_match_dim") is not None]
    pose_success_records = [r for r in records if r.get("pose_success")]
    pose_align_attempted_records = [r for r in records if r.get("t_pose_align") is not None]
    pose_align_success_records = [r for r in records if r.get("pose_align_success")]
    all_success_records = [r for r in records if r.get("all_success")]

    acc_cls = sum(1 for r in records if r.get("class_correct")) / total_samples if total_samples else None
    ae_dim = vector_mean_or_none([r["abs_error_dim"] for r in dim_success_records if r.get("abs_error_dim") is not None])
    re_dim = vector_mean_or_none([r["rel_error_dim"] for r in dim_success_records if r.get("rel_error_dim") is not None])
    me_dim = mean_or_none([r["mean_abs_error_dim"] for r in dim_success_records if r.get("mean_abs_error_dim") is not None])
    succ_query_dim = (
        sum(1 for r in query_dim_records if r.get("query_match_dim")) / len(query_dim_records)
        if query_dim_records
        else None
    )
    succ_pose = (
        len(pose_success_records) / len(pose_attempted_records)
        if pose_attempted_records
        else None
    )
    succ_all = len(all_success_records) / total_samples if total_samples else None

    metrics = {
        "Acc_cls": acc_cls,
        "Conf_cls": mean_or_none([float(r["pred_class_confidence"]) for r in cls_conf_records]),
        "T_cls": mean_or_none([r["t_cls"] for r in records if r.get("t_cls") is not None]),
        "AE_dim": ae_dim,
        "RE_dim": re_dim,
        "ME_dim": me_dim,
        "Conf_dim": mean_or_none([float(r["dim_confidence"]) for r in dim_conf_records]),
        "Succ_query_dim": succ_query_dim,
        "T_dim": mean_or_none([r["t_dim"] for r in dim_attempted_records]),
        "T_pose": mean_or_none([r["t_pose"] for r in pose_attempted_records]),
        "Succ_pose": succ_pose,
        "pose_align_cover_rate": mean_or_none(
            [float(r["pose_align_cover_rate"]) for r in pose_align_success_records if r.get("pose_align_cover_rate") is not None]
        ),
        "pose_align_avg_dist_mm": mean_or_none(
            [float(r["pose_align_avg_dist_mm"]) for r in pose_align_success_records if r.get("pose_align_avg_dist_mm") is not None]
        ),
        "t_pose_align": mean_or_none([r["t_pose_align"] for r in pose_align_attempted_records]),
        "T_all": mean_or_none([r["t_all"] for r in records if r.get("t_all") is not None]),
        "Succ_all": succ_all,
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(normalize_path(args.config)),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "gpu_id": args.gpu_id,
        "query_match_rtol": args.query_match_rtol,
        "query_match_atol": args.query_match_atol,
        "sample_count": total_samples,
        "counts": {
            "total_samples": total_samples,
            "cls_attempted": cls_attempted,
            "cls_success": len(cls_success_records),
            "cls_correct": sum(1 for r in records if r.get("class_correct")),
            "cls_confidence_available": len(cls_conf_records),
            "dim_attempted": len(dim_attempted_records),
            "dim_success": len(dim_success_records),
            "dim_confidence_available": len(dim_conf_records),
            "query_dim_evaluated": len(query_dim_records),
            "query_dim_success": sum(1 for r in query_dim_records if r.get("query_match_dim")),
            "pose_attempted": len(pose_attempted_records),
            "pose_success": len(pose_success_records),
            "pose_align_attempted": len(pose_align_attempted_records),
            "pose_align_success": len(pose_align_success_records),
            "all_success": len(all_success_records),
        },
        "metric_denominators": {
            "Acc_cls": total_samples,
            "Conf_cls": len(cls_conf_records),
            "AE_dim": len(dim_success_records),
            "RE_dim": len(dim_success_records),
            "ME_dim": len(dim_success_records),
            "Conf_dim": len(dim_conf_records),
            "Succ_query_dim": len(query_dim_records),
            "Succ_pose": len(pose_attempted_records),
            "pose_align_cover_rate": len(pose_align_success_records),
            "pose_align_avg_dist_mm": len(pose_align_success_records),
            "t_pose_align": len(pose_align_attempted_records),
            "Succ_all": total_samples,
        },
        "metric_names_zh": METRIC_NAMES_ZH,
        "metric_notes": METRIC_NOTES,
        "metrics": metrics,
    }


def evaluate_sample(
    sample_id: str,
    color_path: Path,
    depth_path: Path,
    meta_path: Path,
    cfg: OmegaConf,
    query_match_rtol: float,
    query_match_atol: float,
    category_recognition: Any,
    dimension_measurement: Any,
    pose_estimation: Any,
    pose_align_metric: PoseAlignmentMetricEvaluator,
) -> Dict[str, Any]:
    meta = load_meta(meta_path)
    gt_class_name = meta["annotation"]["class_name"]
    gt_dimensions = np.asarray(meta["annotation"]["dimensions"], dtype=float)

    record: Dict[str, Any] = {
        "sample_id": sample_id,
        "rgb_path": str(color_path),
        "depth_path": str(depth_path),
        "meta_path": str(meta_path),
        "gt_class_name": gt_class_name,
        "gt_dimensions": gt_dimensions.tolist(),
        "pred_class_name": None,
        "pred_class_id": None,
        "pred_class_confidence": None,
        "class_correct": False,
        "cls_success": False,
        "t_cls": None,
        "pred_dimensions": None,
        "pred_dimensions_raw": None,
        "pred_dimensions_query": None,
        "pose_input_dimensions": None,
        "dim_confidence": None,
        "abs_error_dim": None,
        "rel_error_dim": None,
        "mean_abs_error_dim": None,
        "query_match_dim": None,
        "dim_success": False,
        "t_dim": None,
        "pose": None,
        "pose_success": False,
        "t_pose": None,
        "pose_align_cover_rate": None,
        "pose_align_avg_dist_mm": None,
        "pose_align_success": False,
        "t_pose_align": None,
        "all_success": False,
        "t_all": None,
        "error_stage": None,
        "error_message": None,
    }

    total_start = time.perf_counter()

    cls_pred = None
    cls_start = time.perf_counter()
    try:
        cat_result = category_recognition.infer(color_path)
        if not cat_result:
            raise RuntimeError("No category predictions returned")
        cls_pred = cat_result[0]
        record["pred_class_name"] = cls_pred.get("class_name")
        record["pred_class_id"] = cls_pred.get("class_id")
        record["pred_class_confidence"] = cls_pred.get("confidence")
        record["class_correct"] = record["pred_class_name"] == gt_class_name
        record["cls_success"] = True
    except Exception as exc:
        record["error_stage"] = "category_recognition"
        record["error_message"] = str(exc)
    finally:
        record["t_cls"] = time.perf_counter() - cls_start

    pose_input_dimensions = None
    if record["cls_success"] and record["pred_class_name"] is not None:
        dim_start = time.perf_counter()
        try:
            dim_result = dimension_measurement.infer(
                color_path,
                depth_path,
                record["pred_class_name"],
                return_details=True,
            )
            pred_dimensions_raw = np.asarray(dim_result["raw_length"], dtype=float)
            pred_dimensions_query = dim_result.get("query_length")
            if pred_dimensions_query is not None:
                pred_dimensions_query = np.asarray(pred_dimensions_query, dtype=float)
            pose_input_dimensions = np.asarray(dim_result["final_length"], dtype=float)

            if pred_dimensions_raw.shape != (3,):
                raise RuntimeError(f"Unexpected raw dimension shape: {pred_dimensions_raw.shape}")
            if pose_input_dimensions.shape != (3,):
                raise RuntimeError(f"Unexpected pose input dimension shape: {pose_input_dimensions.shape}")

            abs_error = np.abs(pred_dimensions_raw - gt_dimensions)
            rel_error = np.divide(
                abs_error,
                np.abs(gt_dimensions),
                out=np.full_like(abs_error, np.nan),
                where=np.abs(gt_dimensions) > 1e-12,
            )
            record["pred_dimensions"] = pred_dimensions_raw.astype(float).tolist()
            record["pred_dimensions_raw"] = pred_dimensions_raw.astype(float).tolist()
            record["pred_dimensions_query"] = (
                None if pred_dimensions_query is None else pred_dimensions_query.astype(float).tolist()
            )
            record["pose_input_dimensions"] = pose_input_dimensions.astype(float).tolist()
            record["dim_confidence"] = dim_result.get("query_confidence")
            record["abs_error_dim"] = abs_error.astype(float).tolist()
            record["rel_error_dim"] = rel_error.astype(float).tolist()
            record["mean_abs_error_dim"] = float(abs_error.mean())
            if pred_dimensions_query is not None:
                record["query_match_dim"] = bool(
                    np.allclose(
                        pred_dimensions_query,
                        gt_dimensions,
                        rtol=float(query_match_rtol),
                        atol=float(query_match_atol),
                    )
                )
            record["dim_success"] = True
        except Exception as exc:
            record["error_stage"] = "dimension_measurement"
            record["error_message"] = str(exc)
        finally:
            record["t_dim"] = time.perf_counter() - dim_start

    if record["dim_success"] and pose_input_dimensions is not None:
        pose_start = time.perf_counter()
        pose_array = None
        try:
            pose_result = pose_estimation.infer(
                color_path,
                depth_path,
                record["pred_class_name"],
                pose_input_dimensions,
            )
            pose_array = np.asarray(pose_result, dtype=float)
            if pose_array.shape != (4, 4) or not np.isfinite(pose_array).all():
                raise RuntimeError(f"Invalid pose output: shape={pose_array.shape}")
            record["pose"] = pose_array.astype(float).tolist()
            record["pose_success"] = True
        except Exception as exc:
            record["error_stage"] = "pose_estimation"
            record["error_message"] = str(exc)
        finally:
            record["t_pose"] = time.perf_counter() - pose_start

        if record["pose_success"] and pose_array is not None:
            pose_align_start = time.perf_counter()
            try:
                pose_align_result = evaluate_pose_alignment(
                    sample_id=sample_id,
                    depth_path=depth_path,
                    pose_array=pose_array,
                    pose_estimation=pose_estimation,
                    pose_align_metric=pose_align_metric,
                )
                record["pose_align_cover_rate"] = pose_align_result["pose_align_cover_rate"]
                record["pose_align_avg_dist_mm"] = pose_align_result["pose_align_avg_dist_mm"]
                record["pose_align_success"] = True
            except Exception as exc:
                if record["error_stage"] is None:
                    record["error_stage"] = "pose_align"
                    record["error_message"] = str(exc)
            finally:
                record["t_pose_align"] = time.perf_counter() - pose_align_start

    record["t_all"] = time.perf_counter() - total_start
    record["all_success"] = bool(
        record["cls_success"] and record["dim_success"] and record["pose_success"]
    )
    return record


def print_summary(summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    print("[Eval] Completed batch evaluation")
    print(f"[Eval] Samples: {summary['sample_count']}")
    print(f"[Eval] Acc_cls: {metrics['Acc_cls']}")
    print(f"[Eval] Conf_cls: {metrics['Conf_cls']}")
    print(f"[Eval] T_cls: {metrics['T_cls']}")
    print(f"[Eval] AE_dim: {metrics['AE_dim']}")
    print(f"[Eval] RE_dim: {metrics['RE_dim']}")
    print(f"[Eval] ME_dim: {metrics['ME_dim']}")
    print(f"[Eval] Conf_dim: {metrics['Conf_dim']}")
    print(f"[Eval] Succ_query_dim: {metrics['Succ_query_dim']}")
    print(f"[Eval] T_dim: {metrics['T_dim']}")
    print(f"[Eval] T_pose: {metrics['T_pose']}")
    print(f"[Eval] Succ_pose: {metrics['Succ_pose']}")
    print(f"[Eval] pose_align_cover_rate: {metrics['pose_align_cover_rate']}")
    print(f"[Eval] pose_align_avg_dist_mm: {metrics['pose_align_avg_dist_mm']}")
    print(f"[Eval] t_pose_align: {metrics['t_pose_align']}")
    print(f"[Eval] T_all: {metrics['T_all']}")
    print(f"[Eval] Succ_all: {metrics['Succ_all']}")


def main(args: argparse.Namespace) -> int:
    run_started_at = datetime.now()
    run_started_perf = time.perf_counter()
    os.chdir(REPO_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    config_path = normalize_path(args.config)
    dataset_dir = normalize_path(args.dataset_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    cfg = OmegaConf.load(config_path)
    cfg.debug = False
    cfg.clean_hydra_output = False

    output_dir = (
        normalize_path(args.output_dir)
        if args.output_dir
        else (REPO_ROOT / "eval" / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = discover_samples(dataset_dir)
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]

    records: List[Dict[str, Any]] = []
    runtime_root = output_dir / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    pose_align_metric = PoseAlignmentMetricEvaluator(verbose=False)

    def run_one(index: int, sample_id: str) -> Tuple[int, Dict[str, Any]]:
        color_path = dataset_dir / f"{sample_id}_color.png"
        depth_path = dataset_dir / f"{sample_id}_depth.exr"
        meta_path = dataset_dir / f"{sample_id}_meta.json"

        sample_runtime_root = runtime_root / sample_id
        sample_tmp_dir = sample_runtime_root / "tmp_output"
        reset_directory(sample_runtime_root)
        sample_tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            sample_cfg = build_runtime_cfg(cfg, sample_tmp_dir)
            category_recognition = hydra.utils.instantiate(sample_cfg.category_recognition)
            dimension_measurement = hydra.utils.instantiate(sample_cfg.dimension_measurement)
            pose_estimation = hydra.utils.instantiate(sample_cfg.pose_estimation)

            record = evaluate_sample(
                sample_id=sample_id,
                color_path=color_path,
                depth_path=depth_path,
                meta_path=meta_path,
                cfg=sample_cfg,
                query_match_rtol=args.query_match_rtol,
                query_match_atol=args.query_match_atol,
                category_recognition=category_recognition,
                dimension_measurement=dimension_measurement,
                pose_estimation=pose_estimation,
                pose_align_metric=pose_align_metric,
            )
        except Exception as exc:
            meta = load_meta(meta_path)
            record = {
                "sample_id": sample_id,
                "rgb_path": str(color_path),
                "depth_path": str(depth_path),
                "meta_path": str(meta_path),
                "gt_class_name": meta["annotation"]["class_name"],
                "gt_dimensions": meta["annotation"]["dimensions"],
                "pred_class_name": None,
                "pred_class_id": None,
                "pred_class_confidence": None,
                "class_correct": False,
                "cls_success": False,
                "t_cls": None,
                "pred_dimensions": None,
                "pred_dimensions_raw": None,
                "pred_dimensions_query": None,
                "pose_input_dimensions": None,
                "dim_confidence": None,
                "abs_error_dim": None,
                "rel_error_dim": None,
                "mean_abs_error_dim": None,
                "query_match_dim": None,
                "dim_success": False,
                "t_dim": None,
                "pose": None,
                "pose_success": False,
                "t_pose": None,
                "pose_align_cover_rate": None,
                "pose_align_avg_dist_mm": None,
                "pose_align_success": False,
                "t_pose_align": None,
                "all_success": False,
                "t_all": None,
                "error_stage": "runtime_setup",
                "error_message": str(exc),
            }
        finally:
            if not args.keep_runtime_artifacts:
                shutil.rmtree(sample_runtime_root, ignore_errors=True)

        return index, record

    for index, sample_id in enumerate(sample_ids, start=1):
        _, record = run_one(index, sample_id)
        records.append(record)
        status = "ok" if record["all_success"] else f"failed@{record['error_stage']}"
        if record.get("t_pose_align") is not None:
            pose_align_status = "ok" if record.get("pose_align_success") else "failed"
            status = f"{status}, pose_align={pose_align_status}"
        t_all_str = "n/a" if record["t_all"] is None else f"{record['t_all']:.3f}s"
        print(
            f"[Eval] {index}/{len(sample_ids)} sample={sample_id} "
            f"cls={record['pred_class_name']} all_success={record['all_success']} "
            f"t_all={t_all_str} status={status}"
        )

    records.sort(key=lambda item: item["sample_id"])

    if not args.keep_runtime_artifacts:
        shutil.rmtree(runtime_root, ignore_errors=True)

    summary = build_summary(args=args, dataset_dir=dataset_dir, output_dir=output_dir, records=records)
    run_finished_at = datetime.now()
    run_elapsed_seconds = time.perf_counter() - run_started_perf
    summary["started_at"] = run_started_at.isoformat(timespec="seconds")
    summary["finished_at"] = run_finished_at.isoformat(timespec="seconds")
    summary["total_duration_seconds"] = run_elapsed_seconds

    per_sample_json_path = output_dir / "per_sample_results.json"
    per_sample_csv_path = output_dir / "per_sample_results.csv"
    summary_json_path = output_dir / "summary.json"
    metrics_csv_path = output_dir / "summary_metrics.csv"

    with per_sample_json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=json_default)
    write_per_sample_csv(per_sample_csv_path, records)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=json_default)
    write_metrics_csv(metrics_csv_path, summary)

    print_summary(summary)
    print(f"[Eval] Per-sample JSON: {per_sample_json_path}")
    print(f"[Eval] Per-sample CSV: {per_sample_csv_path}")
    print(f"[Eval] Summary JSON: {summary_json_path}")
    print(f"[Eval] Metrics CSV: {metrics_csv_path}")
    print(f"[Eval] Started at: {summary['started_at']}")
    print(f"[Eval] Finished at: {summary['finished_at']}")
    print(f"[Eval] Total duration: {format_duration(run_elapsed_seconds)}")
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args))
