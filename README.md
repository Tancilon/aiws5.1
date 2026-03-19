# AIWS 5.1

本仓库当前采用如下部署方式：

- 宿主机 `conda aiws` 环境只负责根流程编排
- `YOLOv11`、`GenPose2`、`FoundationPose` 三个算法阶段分别运行在各自的 Docker 镜像中

这样做的目的，是把算法依赖和宿主机编排依赖解耦，方便迁移到新机器。

## 目录说明

- [environment-aiws.yml](environment-aiws.yml)
  - 宿主机 `aiws` 编排环境的最小依赖清单
- [scripts/build_aiws_stack.sh](scripts/build_aiws_stack.sh)
  - 一键构建脚本
- [scripts/download_weights.sh](scripts/download_weights.sh)
  - 单独下载运行权重脚本
- [scripts/run_aiws.sh](scripts/run_aiws.sh)
  - 根流程运行脚本
- [config/aiws_sub.yaml](config/aiws_sub.yaml)
  - 默认配置，包含镜像名、相机内参和临时目录等

## 前置条件

新机器上需要提前具备以下基础环境：

1. `conda`
2. `docker`
3. 如果要跑 GPU 推理，需要宿主机已经配置好 `nvidia-container-toolkit`

建议先验证 Docker GPU 是否正常：

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

如果这条命令不能正常输出 GPU 信息，不建议继续执行后续构建。

## 一键构建

在仓库根目录执行：

```bash
bash scripts/build_aiws_stack.sh
```

该脚本默认会完成三件事：

1. 创建或更新宿主机 `aiws` conda 环境
2. 从 Hugging Face 下载运行期权重
3. 构建三张算法镜像

如果本地没有离线 bundle，那么第 2、3 步默认依赖网络：

- 权重从 Hugging Face 下载
- Docker 镜像通过 `docker build` 在线拉取基础镜像和安装依赖

默认镜像名：

- `yolov11-seg:infer`
- `genpose2-env:test`
- `foundationpose-env:test`

默认权重仓库：

- `tancilon/aiws5.1-weights`

如果你没有登录 Hugging Face，或者不想在这一步下载权重，可以显式关闭：

```bash
AIWS_DOWNLOAD_WEIGHTS=0 bash scripts/build_aiws_stack.sh
```

## 离线构建

如果一台“源机器”已经准备好了三张 Docker 镜像和全部运行权重，可以先把它们导出到仓库内的离线 bundle 目录：

```bash
bash scripts/export_aiws_env_bundle.sh
```

默认会导出到：

```bash
assets/env_files/
```

然后把整个仓库连同 `assets/env_files` 一起交给其他团队。目标机器上可直接执行：

```bash
AIWS_ENV_BUNDLE_MODE=require \
AIWS_SKIP_CONDA_SETUP=1 \
bash scripts/build_aiws_stack.sh
```

含义如下：

- `AIWS_ENV_BUNDLE_MODE=require`：只使用 `assets/env_files` 中的离线镜像包和权重，不再回退到网络下载或在线 `docker build`
- `AIWS_SKIP_CONDA_SETUP=1`：跳过宿主机 `aiws` conda 环境创建；建议在目标机器已经具备该环境时使用

如果你希望“有离线包就用，没有就回退联网构建”，可改用默认模式：

```bash
bash scripts/build_aiws_stack.sh
```

或者显式指定：

```bash
AIWS_ENV_BUNDLE_MODE=auto bash scripts/build_aiws_stack.sh
```

## 使用代理构建

如果新机器访问外网需要代理，例如本地 Clash，可先设置：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5h://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1,::1

bash scripts/build_aiws_stack.sh
```

如果权重仓库是私有的，还需要先登录 Hugging Face，或者提前设置 `HF_TOKEN`。

## 自定义镜像名

如果后续你要把镜像推到私有仓库，或者使用不同 tag，可以在构建前覆盖环境变量：

```bash
export AIWS_YOLO_IMAGE=myrepo/yolov11:latest
export AIWS_GENPOSE2_IMAGE=myrepo/genpose2:latest
export AIWS_FOUNDATIONPOSE_IMAGE=myrepo/foundationpose:latest

bash scripts/build_aiws_stack.sh
```

运行时同样使用这三个环境变量，所以构建和运行可以保持一致。

## 权重下载

运行期真实需要的权重包括：

- `YOLOv11` 运行权重
- `GenPose2` 三个 checkpoint
- `FoundationPose` 两个 checkpoint
- `GenPose2` 依赖的 `dinov2` torch.hub 缓存

### 1. 安装并登录 Hugging Face CLI

在 Linux 上，推荐直接使用 Hugging Face 官方安装脚本：

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login
```

如果你已经在使用 `conda`，也可以安装到当前环境：

```bash
conda install -c conda-forge huggingface_hub
hf auth login
```

`brew install huggingface-cli` 只适用于已经安装了 Homebrew 的机器。

### 2. 新机器下载权重

在新机器上登录 Hugging Face 后执行：

```bash
export AIWS_HF_WEIGHTS_REPO=tancilon/aiws5.1-weights
bash scripts/download_weights.sh
```

这个脚本会把权重恢复到项目当前代码实际使用的位置，并把 `dinov2` 所需的 torch cache 恢复到：

```bash
$HOME/.cache/torch/hub
```

下载完成后，再执行环境构建：

```bash
bash scripts/build_aiws_stack.sh
```

实际上，当前 [build_aiws_stack.sh](scripts/build_aiws_stack.sh) 已经默认把“下载权重”接进去了，所以新机器通常只需要：

```bash
hf auth login
bash scripts/build_aiws_stack.sh
```

如果当前机器上的 `hf` CLI 在代理环境下不稳定，构建脚本默认会切到兼容性更高的 `legacy-python` 下载后端。

## 运行示例

构建完成后，直接运行：

```bash
bash scripts/run_aiws.sh aiws_sub "data/AIWS/F120/1_color.png" "data/AIWS/F120/1_depth.exr" 0
```

说明：

- `aiws_sub` 是配置名，对应 [aiws_sub.yaml](config/aiws_sub.yaml)
- 第 2 个参数是 RGB 图
- 第 3 个参数是深度图
- 第 4 个参数是 GPU 编号

## 当前运行模型

根流程执行时：

- 宿主机使用 `conda run -n aiws python workspace.py`
- 算法阶段全部通过 Docker 启动

也就是说，宿主机不再承担 `YOLOv11`、`GenPose2`、`FoundationPose` 的算法依赖。

## 常见问题

### 1. 构建脚本是否会安装 Docker 或 NVIDIA runtime

不会。

[build_aiws_stack.sh](scripts/build_aiws_stack.sh) 只负责：

- `aiws` conda 环境
- 三张 Docker 镜像

系统级安装和 GPU runtime 配置需要你提前完成。

### 2. 如何确认镜像名被根流程正确读取

可以先设置环境变量，再运行：

```bash
export AIWS_YOLO_IMAGE=myrepo/yolov11:latest
export AIWS_GENPOSE2_IMAGE=myrepo/genpose2:latest
export AIWS_FOUNDATIONPOSE_IMAGE=myrepo/foundationpose:latest
```

根配置 [aiws_sub.yaml](config/aiws_sub.yaml) 会从这三个环境变量读取镜像名。

