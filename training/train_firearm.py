from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="training/firearm_data.yaml",
    epochs=3,
    imgsz=640
)