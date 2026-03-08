#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直接检查深度文件内容"""

import numpy as np
import OpenEXR
import Imath

def read_exr_raw(filename):
    """直接读取 EXR 深度文件"""
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    channels = header['channels']
    print(f"可用通道: {list(channels.keys())}")
    
    # 尝试读取所有通道
    for channel_name in channels.keys():
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
        data = np.frombuffer(channel_str, dtype=dtype)
        data = data.reshape((height, width))
        
        print(f"\n通道 {channel_name}:")
        print(f"  数据类型: {data.dtype}")
        print(f"  形状: {data.shape}")
        print(f"  最小值: {np.min(data):.6f}")
        print(f"  最大值: {np.max(data):.6f}")
        print(f"  非零值数量: {np.count_nonzero(data)}")
        
        valid = data[data > 0]
        if len(valid) > 0:
            print(f"  有效值统计:")
            print(f"    数量: {len(valid)}")
            print(f"    最小值: {np.min(valid):.6f}")
            print(f"    最大值: {np.max(valid):.6f}")
            print(f"    平均值: {np.mean(valid):.6f}")
        
        # 显示一些样本值
        print(f"  前10个值: {data.flatten()[:10]}")

# 检查转换后的文件
print("=" * 60)
print("检查转换后的深度文件:")
print("=" * 60)
read_exr_raw('data/AIWS/000/001_depth.exr')

print("\n" + "=" * 60)
print("检查备份文件（原始）:")
print("=" * 60)
read_exr_raw('data/AIWS/000/001_depth_backup.exr')
