from ultralytics import YOLO

model = YOLO("ckpt/yolo11n-seg.pt")
model.train(
    data="data/aiws-dataset-yolo/data.yaml",
    imgsz=640,
    epochs=100,
    device=0
)
