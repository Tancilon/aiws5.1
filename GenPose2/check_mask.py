#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查mask文件"""

import numpy as np
from cutoop.data_loader import Dataset

# 检查mask文件
print("=" * 60)
print("检查 AIWS mask:")
print("=" * 60)

print("\n" + "=" * 60)
print("使用 Dataset.load_mask 读取:")
print("=" * 60)
mask = Dataset.load_mask('data/AIWS/000/001_mask.exr')
print(f"数据类型: {mask.dtype}")
print(f"形状: {mask.shape}")
print(f"唯一值: {np.unique(mask)}")
for val in np.unique(mask):
    count = np.sum(mask == val)
    print(f"  值 {val}: {count} 个像素 ({count/mask.size*100:.2f}%)")

# 对比官方数据集的mask
print("\n" + "=" * 60)
print("检查官方数据集的 mask (Omni6DPose):")
print("=" * 60)
official_mask = Dataset.load_mask('data/Omni6DPose/ROPE/000000/000000_mask.exr')
print(f"数据类型: {official_mask.dtype}")
print(f"形状: {official_mask.shape}")
print(f"唯一值: {np.unique(official_mask)}")
for val in np.unique(official_mask)[:10]:  # 只显示前10个
    count = np.sum(official_mask == val)
    print(f"  值 {val}: {count} 个像素 ({count/official_mask.size*100:.2f}%)")
if len(np.unique(official_mask)) > 10:
    print(f"  ... 还有 {len(np.unique(official_mask)) - 10} 个值")
