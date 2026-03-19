# Offline Env Bundle

`assets/env_files` stores the offline AIWS runtime bundle.

Expected generated layout:

```text
assets/env_files/
  README.md
  bundle-manifest.txt
  environment-aiws.yml
  docker-images/
    yolov11-seg--infer.tar
    genpose2-env--test.tar
    foundationpose-env--test.tar
  weights/
    yolov11-seg-aiws/runs/segment/train2/weights/best.pt
    GenPose2/results/ckpts/...
    aiws_alignment-feat-model-free/weights/...
    torch-hub/checkpoints/dinov2_vits14_pretrain.pth
    torch-hub/facebookresearch_dinov2_main.tar.gz
```

Generate the bundle on a source machine:

```bash
bash scripts/export_aiws_env_bundle.sh
```

Restore from the bundle on a target machine:

```bash
AIWS_ENV_BUNDLE_MODE=require \
AIWS_SKIP_CONDA_SETUP=1 \
bash scripts/build_aiws_stack.sh
```

Notes:

- `AIWS_ENV_BUNDLE_MODE=require` forces the build script to use only local bundle assets.
- The offline bundle currently covers Docker images and runtime weights. The host-side `aiws` conda environment is not packed into this directory.
