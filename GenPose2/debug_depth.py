#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试深度数据加载"""

import numpy as np
from cutoop.data_loader import Dataset

# 测试加载深度文件
data_prefix = 'data/AIWS/000/001_'
print(f"加载前缀: {data_prefix}")

# 加载深度
depth_path = data_prefix + 'depth.exr'
print(f"\n加载深度文件: {depth_path}")
depth = Dataset.load_depth(depth_path)

print(f"深度数据类型: {depth.dtype}")
print(f"深度形状: {depth.shape}")

# 统计信息
valid_depth = depth[depth > 0]
print(f"\n有效深度统计:")
print(f"  有效像素数: {len(valid_depth)}")
if len(valid_depth) > 0:
    print(f"  最小值: {np.min(valid_depth):.4f}")
    print(f"  最大值: {np.max(valid_depth):.4f}")
    print(f"  平均值: {np.mean(valid_depth):.4f}")
    print(f"  中位数: {np.median(valid_depth):.4f}")

# 加载mask
mask_path = data_prefix + 'mask.exr'
print(f"\n加载mask文件: {mask_path}")
mask = Dataset.load_mask(mask_path)
print(f"Mask数据类型: {mask.dtype}")
print(f"Mask形状: {mask.shape}")
print(f"Mask唯一值: {np.unique(mask)}")

# 检查mask和depth的重叠
print(f"\n检查mask和depth的重叠:")
for obj_idx in np.unique(mask):
    if obj_idx == 0:
        continue
    object_mask = np.equal(mask, obj_idx)
    obj_depth = depth[object_mask]
    valid_obj_depth = obj_depth[obj_depth > 0]
    print(f"  对象 {obj_idx}:")
    print(f"    mask像素数: {np.sum(object_mask)}")
    print(f"    有效深度像素数: {len(valid_obj_depth)}")
    if len(valid_obj_depth) > 0:
        print(f"    深度范围: {np.min(valid_obj_depth):.4f} - {np.max(valid_obj_depth):.4f}")

# 加载颜色
color_path = data_prefix + 'color.png'
print(f"\n加载颜色文件: {color_path}")
color = Dataset.load_color(color_path)
print(f"颜色形状: {color.shape}")
print(f"颜色数据类型: {color.dtype}")

# 检查形状匹配
print(f"\n形状匹配检查:")
print(f"  depth.shape[:2] = {depth.shape[:2]}")
print(f"  mask.shape[:2] = {mask.shape[:2]}")
print(f"  color.shape[:2] = {color.shape[:2]}")
print(f"  是否匹配: {mask.shape[:2] == depth.shape[:2] == color.shape[:2]}")
