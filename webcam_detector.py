from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime
import json

with open("config.json", "r") as file:
    config = json.load(file)

# Load YOLO model
model = YOLO("yolo11n.pt")
firearm_model = YOLO("runs/detect/train-3/weights/best.pt")


# Security-relevant objects
SECURITY_OBJ = set(config["security_objects"])
MIN_CON = config["min_confidence"]
EXIT_TIMEOUT = config["exit_timeout"]

# Create folders if they do not exist
os.makedirs("captures/entries", exist_ok=True)
os.makedirs("captures/exits", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("recordings", exist_ok=True)
os.makedirs("captures/firearms", exist_ok=True)

csv_file = "logs/events.csv"

# Create CSV file only once
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "event",
            "object",
            "duration_seconds"
        ])

# Object state
OBJECT_STATE = {obj: False for obj in SECURITY_OBJ}
ENTRY_TIMES = {obj: None for obj in SECURITY_OBJ}
LAST_SEEN = {obj: None for obj in SECURITY_OBJ}

# Video recording state
RECORDING = False
VIDEO_WRITER = None
VIDEO_START_TIME = None
FIREARM_PRESENT = False
LAST_FIREARM_SEEN = None
FIREARM_TIMEOUT = 3
FIREARM_MIN_CON = 0.70

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("Failing to read webcam.")
        break

    results = model(
        frame,
        classes=[0, 2, 3, 7],
        verbose=False
    )
    firearm_results = firearm_model(
        frame,
        verbose=False
    )

    detected_objects = set()

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = result.names[class_id]

            if label in SECURITY_OBJ and confidence >= MIN_CON:
                detected_objects.add(label)
                LAST_SEEN[label] = datetime.now()
    
    firearm_detected = False
    firearm_confidence = 0

    for result in firearm_results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            label = result.names[int(box.cls[0])]

            if label == "firearm" and confidence >= FIREARM_MIN_CON:
                firearm_detected = True
                firearm_confidence = max(firearm_confidence, confidence)
                LAST_FIREARM_SEEN = datetime.now()

    if firearm_detected and not FIREARM_PRESENT:
        FIREARM_PRESENT = True
        firearm_time = datetime.now()

        print(f"FIREARM DETECTED ({firearm_confidence:.2f})")

        filename = (
            f"captures/firearms/"
            f"firearm_{firearm_time.strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(filename, frame)

        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                firearm_time,
                "FIREARM_DETECTED",
                "firearm",
                ""
            ])

    if FIREARM_PRESENT and LAST_FIREARM_SEEN is not None:
        time_since_firearm = (
            datetime.now() - LAST_FIREARM_SEEN
        ).total_seconds()

        if time_since_firearm >= FIREARM_TIMEOUT:
            FIREARM_PRESENT = False

    # Object entries
    for obj in detected_objects:

        if not OBJECT_STATE[obj]:
            OBJECT_STATE[obj] = True

            entry_time = datetime.now()
            ENTRY_TIMES[obj] = entry_time

            # Start recording when the first object enters
            if not RECORDING:
                RECORDING = True
                VIDEO_START_TIME = entry_time

                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                video_filename = (
                    f"recordings/"
                    f"event_{VIDEO_START_TIME.strftime('%Y%m%d_%H%M%S')}.mp4"
                )

                VIDEO_WRITER = cv2.VideoWriter(
                    video_filename,
                    fourcc,
                    20.0,
                    (width, height)
                )

                print(f"Started recording: {video_filename}")

            print(
                f"{obj.upper()} ENTERED at "
                f"{entry_time.strftime('%H:%M:%S')}"
            )

            filename = (
                f"captures/entries/"
                f"{obj}_{entry_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            )

            cv2.imwrite(filename, frame)

            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    entry_time,
                    "ENTER",
                    obj,
                    ""
                ])

    # Object exits
    for obj in SECURITY_OBJ:

        if OBJECT_STATE[obj]:

            time_since_seen = (
                datetime.now() - LAST_SEEN[obj]
            ).total_seconds()

            if time_since_seen >= EXIT_TIMEOUT:
                exit_time = datetime.now()

                duration = (
                    exit_time - ENTRY_TIMES[obj]
                ).total_seconds()

                print(
                    f"{obj.upper()} EXITED at "
                    f"{exit_time.strftime('%H:%M:%S')} "
                    f"(Present for {duration:.1f}s)"
                )

                filename = (
                    f"captures/exits/"
                    f"{obj}_{exit_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                )

                cv2.imwrite(filename, frame)

                with open(csv_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        exit_time,
                        "EXIT",
                        obj,
                        f"{duration:.1f}"
                    ])

                OBJECT_STATE[obj] = False
                ENTRY_TIMES[obj] = None

    # Write video frame if currently recording
    if RECORDING and VIDEO_WRITER is not None:
        VIDEO_WRITER.write(frame)

    # Stop recording when all objects have exited
    if RECORDING and not any(OBJECT_STATE.values()):
        RECORDING = False

        VIDEO_WRITER.release()
        VIDEO_WRITER = None
        VIDEO_START_TIME = None

        print("Recording stopped.")

    # Display bounding boxes
    annotated_frame = results[0].plot()

    for result in firearm_results:
        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence >= FIREARM_MIN_CON:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    f"FIREARM {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    active_objects = [
        obj for obj, state in OBJECT_STATE.items()
        if state
    ]

    cv2.putText(
        annotated_frame,
        f"Recording: {'ON' if RECORDING else 'OFF'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255) if RECORDING else (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Objects: {', '.join(active_objects) if active_objects else 'None'}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Firearm: {'DETECTED' if FIREARM_PRESENT else 'None'}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255) if FIREARM_PRESENT else (255, 255, 255),
        2
    )

    if RECORDING and VIDEO_START_TIME:
        duration = int(
            (datetime.now() - VIDEO_START_TIME).total_seconds()
        )

        minutes = duration // 60
        seconds = duration % 60

        cv2.putText(
            annotated_frame,
            f"Duration: {minutes:02d}:{seconds:02d}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    cv2.imshow(
        "Security Camera",
        annotated_frame
    )

    # ESC to quit
    if cv2.waitKey(1) == 27:
        break

# Clean up
if VIDEO_WRITER is not None:
    VIDEO_WRITER.release()

cap.release()
cv2.destroyAllWindows()