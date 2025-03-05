import numpy as np
import cv2
import time
import winsound
import csv
import os
from collections import defaultdict

# Load model and classes
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
frame_count = 0
start_time = time.time()
object_tracker = defaultdict(int)
log_file = 'detection_log.csv'

if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Object", "Count"])

while True:
    ret, image = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    height, width = image.shape[:2]
    frame_count += 1
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0 / 127.5, (300, 300), 127.5)
    net.setInput(blob)
    detected_objects = net.forward()

    object_counts = {}
    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0, 0, i, 2]
        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])
            label = classes[class_index]

            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            object_counts[label] = object_counts.get(label, 0) + 1
            object_tracker[label] += 1

            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
            cv2.putText(image, label, (upper_left_x, upper_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        colors[class_index], 2)

            if label == "person":
                winsound.Beep(1000, 200)

            frame_name = f"detected_{label}_{i}.jpg"
            cv2.imwrite(frame_name, image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(image, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    fps = frame_count / (time.time() - start_time)
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Live Object Detection", image)

    print(object_counts)

    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        for obj, count in object_counts.items():
            writer.writerow([timestamp, obj, count])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()