from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train-2/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Firearm Detector", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()