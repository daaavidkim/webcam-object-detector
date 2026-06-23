from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="training/firearm_data.yaml",
    epochs=25,
    imgsz=640
)

for result in results:
    for box in result.boxes:

        confidence = float(box.conf[0])

        print(
            f"FIREARM DETECTED "
            f"{confidence:.2f}"
        )