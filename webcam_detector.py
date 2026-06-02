from ultralytics import YOLO
import cv2

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

OBJ_STATE = {
    "person": False,
    "car" : False,
    "truck" : False,
    "motorcycle" : False,
}


MIN_CON = 0.50

# opens webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("Failing to read webcam.")
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = result.names[class_id]


            #checks if in security_obj and that the model is confident
            if(label in SECURITY_OBJ and confidence > MIN_CON):
                print(f"Deteced: {label} "
                      f"({confidence:.2f})")

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