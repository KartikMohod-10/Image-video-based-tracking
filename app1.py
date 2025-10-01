# Simple object Detection 
import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
image = cv2.imread("cat.jpg")
results = model(image)
annotated_image = results[0].plot()
cv2.imshow("Annotated Image",annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()