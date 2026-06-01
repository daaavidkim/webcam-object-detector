from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            print(
                f"{result.names[class_id]} "
                f"{confidence:.2f}"
            )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Object Detector",
        annotated_frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()