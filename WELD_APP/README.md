# WELD_APP

## 子项目定位
WELD_APP 是在现有算法流水线基础上提供“GUI → SDK → 服务端”的调用链：
```
GUI 按钮触发 request → weld_client_sdk 发起请求 → weld_vision_server 接收参数并调用算法 → 返回结果
```

## 目录结构
- `weld_client_cmd/`：GUI 与交互逻辑（本地展示与按钮控制）
- `weld_client_sdk/`：HTTP SDK（GUI 与服务端通信）
- `weld_vision_server/`：算法服务端（加载配置、串联 Category/Dimension/Pose）

## 快速启动
> 运行前进入深圳主机/home/dq/mnt/localgit/aiws项目目录

### 1) 启动服务端
```
python WELD_APP/weld_vision_server/server.py \
  --host 127.0.0.1 --port 8000 \
  --config config/aiws_sub.yaml
```

### 2) 启动 GUI
```
python WELD_APP/weld_client_cmd/gui.py --server http://127.0.0.1:8000
```

> 不传 `--server` 时，GUI 使用 Dummy 后端（仅用于界面演示）。

## GUI 使用说明
1. 选择 RGB / Depth / Intrinsics 文件（或选择文件夹自动检测）
2. 点击 `Run Next Step` 或逐步运行（YOLO → GenPose++ → FoundationPose）
3. 结果会显示在对应 Tab

## 数据与文件格式
### RGB
支持：`png/jpg/jpeg/bmp/tif/tiff/npy/npz`

### Depth
支持：`png/tif/tiff/npy/npz/exr`
- GenPose++ 推荐使用 **EXR 深度**
- 如果你使用 EXR，确认深度单位（算法内部会做米/毫米检查）

### Intrinsics（相机内参）
支持：`json/yml/yaml`，格式两种写法之一：
```
# 方式 1：K 矩阵
{"K": [[fx, 0, cx],[0, fy, cy],[0, 0, 1]]}

# 方式 2：分量
{"fx": 1171.2, "fy": 1170.4, "cx": 970.1, "cy": 542.5, "width": 1920, "height": 1080}
```
> 注意：GenPose++ 需要 `width/height` 字段，GUI/服务端已自动补齐，但推荐在内参文件里显式提供。

## 结果展示说明
- **YOLOv11-Seg**：
  - GUI 会显示服务端 `tmp_output_path` 下 `_mask` 掩码（由服务端返回 `mask_path`）
- **GenPose++**：
  - 前端显示的 `size_mm` 会自动将米转换为毫米（仅展示）
- **FoundationPose**：
  - GUI 会优先展示服务端生成的 `foundation_debug/track_vis` 第一张图
  - 若未生成，则回退为 GUI 内部的姿态可视化

## 与 workspace.py 的一致性
服务端使用 `config/aiws_sub.yaml`（与 `workspace.py` 同源配置）：
- `category_recognition` → YOLOv11-Seg
- `dimension_measurement` → GenPose2
- `pose_estimation` → FoundationPose

## 环境依赖提示
算法侧通过已有 conda 环境执行：
- `yolo11`（CategoryRecognition）
- `genpose2`（DimensionMeasurement）
- `foundationpose`（PoseEstimation）

GUI 依赖：
- `PyQt5`
- `numpy`
- `opencv-python`（读取 png/tiff/depth 可视化；若仅显示 debug_vis，可无 cv2）
- `OpenEXR/Imath`（可选，用于读取 EXR）

## 常见问题
- **GenPose++ 报错 KeyError: 'width'**
  - 内参缺少 width/height，或未正确加载内参文件
- **GUI 无法启动 / Qt xcb 报错**
  - 需在有图形界面的会话下运行，确保系统依赖完整
- **FoundationPose 显示灰图**
  - 已改为直接显示服务端输出的 `track_vis` 原图

