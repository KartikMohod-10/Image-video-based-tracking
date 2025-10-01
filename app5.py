# Multi-Object Tracking with Trail Visualization using YOLOv8
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# Load video and model
cap = cv2.VideoCapture('Persons_Walking.mp4')
model = YOLO("yolov8n.pt")

# Initialize tracking structures
trail = defaultdict(lambda: deque(maxlen=30))  # Stores movement trail

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run tracking on the frame (class 0 = person)
    results = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated_frame = frame.copy()

    # Check if tracking IDs are available
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Update trail
            trail[oid].append((cx, cy))

            # Draw bounding box, ID, and centroid
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f'ID:{int(oid)}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

            # Draw trail
            for i in range(1, len(trail[oid])):
                cv2.line(annotated_frame, trail[oid][i - 1], trail[oid][i], (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
