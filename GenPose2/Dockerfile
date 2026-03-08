FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04@sha256:28cb396884380adc15a4bda23e9654610cbc4e20a3292d7e7679a717db5f90d2

ARG DEBIAN_FRONTEND=noninteractive
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy
ARG no_proxy

ENV TZ=Etc/UTC \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    ALL_PROXY=${ALL_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy:-${HTTP_PROXY}} \
    https_proxy=${https_proxy:-${HTTPS_PROXY}} \
    all_proxy=${all_proxy:-${ALL_PROXY}} \
    no_proxy=${no_proxy:-${NO_PROXY}} \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    for attempt in 1 2 3 4 5; do \
      apt-get update -o Acquire::Retries=5 && break; \
      echo "apt-get update failed on attempt ${attempt}, retrying..."; \
      sleep 5; \
    done && \
    apt-get install -y -o Acquire::Retries=5 --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopenexr-dev \
    libsm6 \
    libusb-1.0-0 \
    libxext6 \
    libxrender1 \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

WORKDIR /opt/genpose2

COPY environment.yml /tmp/environment.yml

SHELL ["/bin/bash", "-lc"]

RUN micromamba create -y -n genpose2 -f /tmp/environment.yml && \
    micromamba clean --all --yes

ENV PATH=/opt/micromamba/envs/genpose2/bin:/opt/micromamba/condabin:$PATH \
    LD_LIBRARY_PATH=/opt/micromamba/envs/genpose2/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONPATH=/opt/genpose2:${PYTHONPATH} \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118

COPY . /opt/genpose2

RUN python -m pip install --upgrade pip wheel setuptools==80.9.0 && \
    python -m pip install \
      torch==2.1.0+cu118 \
      torchvision==0.16.0+cu118 \
      torchaudio==2.1.0+cu118 && \
    awk '!/^(torch|torchvision|torchaudio|pointnet2)==/' requirements.txt > /tmp/requirements.docker.txt && \
    python -m pip install -r /tmp/requirements.docker.txt && \
    python -m pip install setuptools==80.9.0 && \
    cd networks/pts_encoder/pointnet2_utils/pointnet2 && \
    python setup.py install

ENV LD_LIBRARY_PATH=/opt/micromamba/envs/genpose2/lib:/opt/micromamba/envs/genpose2/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN python - <<'PY'
import OpenEXR
import cutoop
import torch
import pointnet2_cuda
import pyrealsense2

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("docker_env_import_check: ok")
PY

CMD ["bash"]
