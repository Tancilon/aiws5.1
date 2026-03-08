import torch
from ultralytics import YOLO

model = YOLO("ckpt/yolo11n-seg.pt")

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("ultralytics_model_ready:", model.model is not None)
