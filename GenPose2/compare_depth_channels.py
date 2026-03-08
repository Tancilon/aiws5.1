#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比两个深度文件的通道信息"""

import OpenEXR
import Imath

def check_exr_channels(filename):
    """检查 EXR 文件的通道信息"""
    print(f"\n文件: {filename}")
    print("-" * 60)
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    
    channels = header['channels']
    print(f"通道列表: {list(channels.keys())}")
    
    for name, channel in channels.items():
        pt = channel.type
        if pt == Imath.PixelType(Imath.PixelType.FLOAT):
            dtype = "FLOAT (32-bit)"
        elif pt == Imath.PixelType(Imath.PixelType.HALF):
            dtype = "HALF (16-bit)"
        elif pt == Imath.PixelType(Imath.PixelType.UINT):
            dtype = "UINT (32-bit)"
        else:
            dtype = "UNKNOWN"
        
        print(f"  通道 '{name}': {dtype}")
        print(f"    x_sampling: {channel.xSampling}")
        print(f"    y_sampling: {channel.ySampling}")

print("=" * 60)
print("AIWS 深度文件（转换后）")
print("=" * 60)
check_exr_channels('data/AIWS/000/001_depth.exr')

print("\n" + "=" * 60)
print("AIWS 深度文件（原始备份）")
print("=" * 60)
check_exr_channels('data/AIWS/000/001_depth_backup.exr')

print("\n" + "=" * 60)
print("官方 Omni6DPose 深度文件")
print("=" * 60)
check_exr_channels('data/Omni6DPose/ROPE/000000/000000_depth.exr')
