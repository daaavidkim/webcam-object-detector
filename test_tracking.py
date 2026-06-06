from ultralytics import YOLO
import cv2
from datetime import datetime


# Tracking test - Goal
# See whether yolo can track IDs of different people or obj
# To replace presence tracking and instead tracking IDs

# Result
# Tracking did work however unstable during testing
# The main application currently uses presence-based logic instead
# because object presence was more reliable than tracking IDs.


# Different Yolo model 
# Slower but usually more accurate than yolo11n
model = YOLO("yolo11s.pt")

# relevant obj
SECURITY_OBJ = {
    "person",
    "car",
    "truck",
    "motorcycle"
}

# moved confidence threshold to make less shaky detections
MIN_CON = 0.60
# longer exit timeout because frame cuts could be causing more IDs
EXIT_TIMEOUT = 5

# storing actively trackied obj
TRACKED_OBJECTS = {}

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            if box.id is None:
                continue
            
            # New track id
            track_id = int(box.id[0])

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            label = result.names[class_id]

            if (
                label not in SECURITY_OBJ
                or confidence < MIN_CON
            ):
                continue

            # New obj
            if track_id not in TRACKED_OBJECTS:

                TRACKED_OBJECTS[track_id] = {
                    "label": label,
                    "entry_time": datetime.now(),
                    "last_seen": datetime.now()
                }

                print(
                    f"{label.upper()} #{track_id} ENTERED "
                    f"{confidence:.2f}"
                )

            # Existing obj
            else:

                TRACKED_OBJECTS[track_id]["last_seen"] = datetime.now()

    # obj exits
    for track_id in list(TRACKED_OBJECTS.keys()):

        obj = TRACKED_OBJECTS[track_id]

        time_since_seen = (
            datetime.now()
            - obj["last_seen"]
        ).total_seconds()

        #check it left long enough to be considered exit
        if time_since_seen >= EXIT_TIMEOUT:

            duration = (
                datetime.now()
                - obj["entry_time"]
            ).total_seconds()

            print(
                f"{obj['label'].upper()} "
                f"#{track_id} EXITED "
                f"({duration:.1f}s)"
            )

            del TRACKED_OBJECTS[track_id]

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Tracking Test",
        annotated_frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()