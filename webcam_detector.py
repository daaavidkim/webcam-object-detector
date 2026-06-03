from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime

# loads yolo model
model = YOLO("yolo11n.pt")

# since we only need necessary objects for security cam
# create watch list with only necessary objects
# only obj we check for

SECURITY_OBJ = {
    "person",
    "car",
    "truck",
    "motorcycle"
}

MIN_CON = 0.50
EXIT = 2
#how long obj must be gone for it to be recorded **this is for frame lag
#or in case camera doesn't catch obj for a frame


#check folder exists
os.makedirs("captures/entries", exist_ok=True)
os.makedirs("captures/exits", exist_ok=True)
os.makedirs("logs", exist_ok=True)
csv_file = "logs/events.csv"

#only want to create csv file once
#first time (not false)
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        write = csv.writer(file)
        write.writerow([
            "timestamp",
            "event",
            "object",
            "duration_seconds"
        ])

# obj state

OBJECT_STATE = {
    "person" : False,
    "car" : False,
    "truck" : False,
    "motorcycle" : False
}

ENTRY_TIMES = { 
    "person" : None,
    "car" : None,
    "truck" : None,
    "motorcycle" : None
}

LAST_SEEN = {
    "person" : None,
    "car" : None,
    "truck" : None,
    "motorcycle" : None
}

# opens webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("Failing to read webcam.")
        break

    results = model(frame)

    detected_objects = set()

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = result.names[class_id]


            #checks if in security_obj and that the model is confident
            if(label in SECURITY_OBJ and confidence >= MIN_CON):
                detected_objects.add(label)
                LAST_SEEN[label] = datetime.now()

    #obj entry

    for obj in detected_objects:

        if not OBJECT_STATE[obj]:
            OBJECT_STATE[obj] = True

            entry_time = datetime.now()
            ENTRY_TIMES[obj] = entry_time

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

    #obj exits
    for obj in SECURITY_OBJ:
        if OBJECT_STATE[obj]:

            time_since_seen = (datetime.now() - LAST_SEEN[obj]).total_seconds()

            if time_since_seen >= EXIT:
                exit_time = datetime.now()

                duration = (exit_time - ENTRY_TIMES[obj]).total_seconds()

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


    #boxes
    annotated_frame = results[0].plot()
    # captures everythingokay
    cv2.imshow(
        "Security Camera",
        annotated_frame
    )
    #esc = quit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()