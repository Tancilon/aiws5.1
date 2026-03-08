FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

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
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    ALL_PROXY=${ALL_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy:-${HTTP_PROXY}} \
    https_proxy=${https_proxy:-${HTTPS_PROXY}} \
    all_proxy=${all_proxy:-${ALL_PROXY}} \
    no_proxy=${no_proxy:-${NO_PROXY}} \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    for attempt in 1 2 3 4 5; do \
      apt-get update -o Acquire::Retries=5 && break; \
      echo "apt-get update failed on attempt ${attempt}, retrying..."; \
      sleep 5; \
    done && \
    apt-get install -y -o Acquire::Retries=5 --no-install-recommends \
      bash \
      bzip2 \
      ca-certificates \
      curl \
      git \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN curl -L --retry 5 --retry-all-errors --retry-delay 5 \
      https://micro.mamba.pm/api/micromamba/linux-64/latest \
      -o /tmp/micromamba.tar.bz2 && \
    tar -xvjf /tmp/micromamba.tar.bz2 -C /usr/local/bin --strip-components=1 bin/micromamba && \
    rm -f /tmp/micromamba.tar.bz2

WORKDIR /opt/yolov11

COPY requirements.txt /tmp/requirements.txt

SHELL ["/bin/bash", "-lc"]

RUN micromamba create -y -n yolo11 python=3.9 pip && \
    micromamba clean --all --yes

ENV PATH=/opt/micromamba/envs/yolo11/bin:/opt/micromamba/condabin:$PATH

COPY . /opt/yolov11

RUN python -m pip install --upgrade pip wheel setuptools==80.9.0 && \
    python -m pip install -r /tmp/requirements.txt

RUN python /opt/yolov11/docker/import_check.py

CMD ["bash"]
