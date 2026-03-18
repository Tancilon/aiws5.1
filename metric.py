r'''
指标计算脚本
计算观测深度图与渲染深度图在深度上的对比指标
包括：
1. 观测点云覆盖率 (Observed Point Cloud Cover Rate)
2. 平均深度差异 (Average Depth Difference)（去除80mm以上的outlier）

# 单个文件（带mask）
python metric.py \
    --obs_depth ../test_photo/depth/001.exr \
    --mesh ../test_photo/mesh/G90.obj \
    --pose ../test_photo/6D\ pose结果/1.txt \
    --cam_K ../test_photo/cam_K.txt \
    --mask ../test_photo/mask/001_mask.png

# 批量处理（带mask目录）
python metric.py \
    --obs_depth ../test_photo/depth \
    --mesh ../test_photo/mesh/G90.obj \
    --pose ../test_photo/6D\ pose结果 \
    --cam_K ../test_photo/cam_K.txt \
    --mask ../test_photo/mask \
    --batch
'''

import numpy as np
import OpenEXR
import Imath
import trimesh
import os
import argparse
import glob
from pathlib import Path
from PIL import Image

PREFERRED_EXR_DEPTH_CHANNELS = ("Z", "Y", "R")


def get_exr_depth_channel(channels):
    """选择深度通道并返回对应像素类型和numpy dtype。"""
    if not channels:
        raise ValueError("No channel is present in the EXR file.")

    channel_name = None
    for preferred_name in PREFERRED_EXR_DEPTH_CHANNELS:
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


def read_exr_to_array(file_path:str):
    """将EXR格式的深度图文件转化为numpy array"""
    input_file = OpenEXR.InputFile(file_path)
    try:
        header = input_file.header()
        dw = header['dataWindow']
        width, height = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        channel_name, pixel_type, dtype = get_exr_depth_channel(header['channels'])
        depth_str = input_file.channel(channel_name, pixel_type)
        depth_data = np.frombuffer(depth_str, dtype=dtype).reshape(height, width)
        return depth_data.astype(np.float32, copy=False)
    finally:
        input_file.close()


def convert_depth_to_mm(depth: np.ndarray, depth_name: str):
    """自动将米单位深度转换为毫米。"""
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]

    if valid_depth.size == 0:
        return depth.astype(np.float32, copy=False)

    is_millimeter = False
    if float(valid_depth.max()) > 10:
        is_millimeter = True
    if float(valid_depth.mean()) > 1000:
        is_millimeter = True

    if is_millimeter:
        return depth.astype(np.float32, copy=False)

    print(f"[INFO] Converting {depth_name} from meters to millimeters.")
    return depth.astype(np.float32, copy=False) * 1000.0


def load_mask(mask_path: str):
    """加载mask图像"""
    mask_img = Image.open(mask_path)
    mask = np.array(mask_img)
    
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    mask = mask > 0
    return mask


def render_depth_trimesh(mesh_path: str, pose: np.ndarray, K: np.ndarray, height: int = 1080, width: int = 1920):
    """使用trimesh的RayMeshIntersector渲染深度图
    这是一个纯CPU实现，不依赖OpenGL
    """
    # 加载mesh并应用pose变换
    mesh = trimesh.load(mesh_path)
    print(f"[DEBUG] Mesh loaded: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
    print(f"[DEBUG] Mesh bounds: {mesh.bounds}")
    
    # 应用pose变换
    mesh.apply_transform(pose)
    
    # 提取相机内参
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"[DEBUG] Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # 创建深度图
    depth = np.zeros((height, width), dtype=np.float32)
    
    # 为每个像素生成射线并计算深度
    print("[INFO] Rendering depth map using ray casting (this may take a while)...")
    
    # 相机原点在世界坐标系的位置
    camera_origin = np.array([0, 0, 0])
    
    # 创建RayMeshIntersector
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    
    # 批量处理以提高效率
    batch_size = 10000
    total_pixels = height * width
    
    for batch_start in range(0, total_pixels, batch_size):
        batch_end = min(batch_start + batch_size, total_pixels)
        
        # 生成像素坐标
        pixel_indices = np.arange(batch_start, batch_end)
        v = pixel_indices // width
        u = pixel_indices % width
        
        # 像素坐标转归一化平面坐标
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # 射线方向（相机坐标系下）
        ray_directions = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)
        ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)
        
        # 射线起点都是相机原点
        ray_origins = np.tile(camera_origin, (len(ray_directions), 1))
        
        # 求交
        locations, index_ray, index_tri = intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )
        
        # 计算深度（Z值）
        if len(locations) > 0:
            depths_batch = locations[:, 2]  # Z坐标即为深度
            depth.flat[batch_start + index_ray] = depths_batch
        
        # 显示进度
        if (batch_start // batch_size) % 10 == 0:
            progress = (batch_end / total_pixels) * 100
            print(f"  Progress: {progress:.1f}%", end='\r')
    
    print(f"\n[DEBUG] Rendered depth: min={depth[depth>0].min() if (depth>0).any() else 0:.4f}, "
          f"max={depth.max():.4f}, "
          f"valid_pixels={(depth>0).sum()}/{depth.size}")
    
    return depth


def cal_depth_metrics(obs_depth_path: str, 
                      rendered_depth_path: str,
                      mask_path: str = None):
    """计算深度指标"""
    obs_depth = convert_depth_to_mm(read_exr_to_array(obs_depth_path), "observed depth")
    obs_mask = obs_depth > 0

    rendered_depth = convert_depth_to_mm(read_exr_to_array(rendered_depth_path), "rendered depth")
    template_mask = rendered_depth > 0

    if mask_path is not None:
        foreground_mask = load_mask(mask_path)
        if foreground_mask.shape != obs_depth.shape:
            raise ValueError(f"Mask shape {foreground_mask.shape} != depth shape {obs_depth.shape}")
        obs_mask = obs_mask & foreground_mask

    inter_mask = obs_mask & template_mask
    template_total = template_mask.sum()
    
    if template_total == 0:
        print("Warning: No valid template pixels found!")
        return {'obs_point_cloud_cover_rate': 0.0, 'avg_dist': 0.0}
    
    obs_point_cloud_cover_rate = inter_mask.sum() / template_total

    err_dist = np.abs(obs_depth - rendered_depth)
    inter_mask_remove_threshold = err_dist < 80
    inter_mask = inter_mask_remove_threshold & inter_mask
    
    if inter_mask.sum() == 0:
        print("Warning: No valid intersection pixels after filtering!")
        avg_dist = 0.0
    else:
        avg_dist = err_dist[inter_mask].sum() / inter_mask.sum()

    result = {
        'obs_point_cloud_cover_rate': obs_point_cloud_cover_rate,
        'avg_dist': avg_dist,
    }

    return result


def load_camera_intrinsic(cam_K_path: str):
    """加载相机内参"""
    K = np.loadtxt(cam_K_path)
    return K


def load_pose(pose_path: str):
    """加载6D pose"""
    pose = np.loadtxt(pose_path)
    return pose


def save_depth_as_exr(depth: np.ndarray, output_path: str):
    """保存深度图为EXR格式"""
    height, width = depth.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    header = OpenEXR.Header(width, height)
    header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    
    out = OpenEXR.OutputFile(output_path, header)
    z_data = depth.astype(np.float32).tobytes()
    out.writePixels({'Z': z_data})
    out.close()


def process_single_pose(obs_depth_path: str, mesh_path: str, pose_path: str, 
                       cam_K_path: str, mask_path: str = None, output_dir: str = None):
    """处理单个pose的指标计算"""
    K = load_camera_intrinsic(cam_K_path)
    pose = load_pose(pose_path)
    
    obs_depth = read_exr_to_array(obs_depth_path)
    height, width = obs_depth.shape
    
    print(f"Rendering depth map with pose: {pose_path}")
    rendered_depth = render_depth_trimesh(mesh_path, pose, K, height, width)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pose_name = Path(pose_path).stem
        rendered_depth_path = os.path.join(output_dir, f"{pose_name}_rendered.exr")
        save_depth_as_exr(rendered_depth, rendered_depth_path)
        print(f"Saved rendered depth to: {rendered_depth_path}")
    else:
        rendered_depth_path = "/tmp/temp_rendered.exr"
        save_depth_as_exr(rendered_depth, rendered_depth_path)
    
    result = cal_depth_metrics(obs_depth_path, rendered_depth_path, mask_path)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='6D Pose指标计算脚本 (trimesh版本)')
    parser.add_argument('--obs_depth', type=str, required=True)
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--pose', type=str, required=True)
    parser.add_argument('--cam_K', type=str, required=True)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    result = process_single_pose(args.obs_depth, args.mesh, args.pose, args.cam_K, args.mask, args.output_dir)
    print(f"\nResults:")
    print(f"Coverage Rate: {result['obs_point_cloud_cover_rate']:.4f}")
    print(f"Avg Distance: {result['avg_dist']:.4f} mm")


if __name__ == "__main__":
    main()
