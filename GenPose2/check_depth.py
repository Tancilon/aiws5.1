#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查并转换深度文件单位"""

import numpy as np
import OpenEXR
import Imath
import os
import argparse

def read_exr(filename):
    """读取 EXR 深度文件"""
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 读取深度通道
    # 通常深度存储在 'Y' 或 'R' 通道
    channels = header['channels']
    channel_name = None
    if 'Y' in channels:
        channel_name = 'Y'
    elif 'R' in channels:
        channel_name = 'R'
    else:
        # 获取第一个通道
        channel_name = list(channels.keys())[0]
    
    print(f"使用通道: {channel_name}")
    print(f"可用通道: {list(channels.keys())}")
    
    # 获取像素类型
    pt = channels[channel_name].type
    if pt == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif pt == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    elif pt == Imath.PixelType(Imath.PixelType.UINT):
        dtype = np.uint32
    else:
        dtype = np.float32
    
    # 读取数据
    channel_str = exr_file.channel(channel_name, pt)
    depth = np.frombuffer(channel_str, dtype=dtype)
    depth = depth.reshape((height, width))
    
    return depth, channel_name

def write_exr(filename, depth, channel_name='Y', force_channel_name=None):
    """写入 EXR 深度文件"""
    height, width = depth.shape
    
    # 如果强制指定通道名，使用指定的
    if force_channel_name is not None:
        channel_name = force_channel_name
    
    # 确定像素类型
    if depth.dtype in [np.uint8, np.uint16, np.uint32]:
        depth = depth.astype(np.uint32)
        pixel_type = Imath.PixelType(Imath.PixelType.UINT)
    elif depth.dtype in [np.float16, np.float32, np.float64]:
        depth = depth.astype(np.float32)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    else:
        raise TypeError(f"不支持的数据类型: {depth.dtype}")
    
    header = OpenEXR.Header(width, height)
    header['channels'] = {channel_name: Imath.Channel(pixel_type)}
    
    exr = OpenEXR.OutputFile(filename, header)
    exr.writePixels({channel_name: depth.tobytes()})
    exr.close()

def check_and_convert_depth(filepath, backup=True, convert=False):
    """
    检查深度文件单位，如果是毫米则转换为米
    
    Args:
        filepath: 深度文件路径
        backup: 是否备份原文件
        convert: 是否执行转换
    """
    print(f"\n检查文件: {filepath}")
    print("=" * 60)
    
    if not os.path.exists(filepath):
        print(f"错误: 文件不存在: {filepath}")
        return
    
    # 读取深度数据
    depth, channel_name = read_exr(filepath)
    
    # 统计信息
    valid_depth = depth[depth > 0]
    if len(valid_depth) == 0:
        print("警告: 没有有效的深度值")
        return
    
    print(f"\n深度统计信息:")
    print(f"  形状: {depth.shape}")
    print(f"  数据类型: {depth.dtype}")
    print(f"  最小值: {np.min(valid_depth):.4f}")
    print(f"  最大值: {np.max(valid_depth):.4f}")
    print(f"  平均值: {np.mean(valid_depth):.4f}")
    print(f"  中位数: {np.median(valid_depth):.4f}")
    print(f"  有效像素数: {len(valid_depth)}")
    
    # 判断单位
    # 如果最大值大于10，很可能是毫米单位（假设场景深度小于10米）
    # 如果平均值大于1000，更确定是毫米单位
    is_millimeter = False
    if np.max(valid_depth) > 10:
        is_millimeter = True
        print(f"\n判断: 深度单位可能是 **毫米** (最大值 {np.max(valid_depth):.2f} > 10)")
    else:
        print(f"\n判断: 深度单位可能是 **米** (最大值 {np.max(valid_depth):.2f} <= 10)")
    
    if np.mean(valid_depth) > 1000:
        is_millimeter = True
        print(f"  确认: 平均值 {np.mean(valid_depth):.2f} > 1000, 确定是毫米单位")
    
    # 如果是毫米且需要转换
    if is_millimeter and convert:
        print(f"\n开始转换: 毫米 -> 米 (除以 1000)")
        print(f"  同时将通道从 '{channel_name}' 改为 'Y'（兼容 Dataset.load_depth）")
        
        # 备份原文件
        if backup:
            backup_path = filepath.replace('.exr', '_backup.exr')
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(filepath, backup_path)
                print(f"  已备份到: {backup_path}")
        
        # 转换为米
        depth_meters = depth / 1000.0
        
        # 保存转换后的文件，使用 'Y' 通道
        write_exr(filepath, depth_meters, force_channel_name='Y')
        print(f"  已保存转换后的文件: {filepath}")
        
        # 验证
        depth_verify, channel_verify = read_exr(filepath)
        valid_verify = depth_verify[depth_verify > 0]
        print(f"\n转换后验证:")
        print(f"  通道名: {channel_verify}")
        print(f"  最小值: {np.min(valid_verify):.4f}")
        print(f"  最大值: {np.max(valid_verify):.4f}")
        print(f"  平均值: {np.mean(valid_verify):.4f}")
        print(f"  中位数: {np.median(valid_verify):.4f}")
    elif is_millimeter and not convert:
        print(f"\n提示: 使用 --convert 参数来执行转换")
    else:
        print(f"\n无需转换，深度单位已经是米")

def main():
    parser = argparse.ArgumentParser(description='检查并转换深度文件单位')
    parser.add_argument('filepath', type=str, help='深度文件路径')
    parser.add_argument('--convert', action='store_true', help='执行转换（从毫米到米）')
    parser.add_argument('--no-backup', action='store_true', help='不备份原文件')
    
    args = parser.parse_args()
    
    check_and_convert_depth(
        args.filepath, 
        backup=not args.no_backup,
        convert=args.convert
    )

if __name__ == '__main__':
    main()
