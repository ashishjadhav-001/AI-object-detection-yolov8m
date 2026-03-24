from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model= YOLO('yolov8n.pt')

image= cv2.imread("data/test.jpg")

results= model(image, conf=0.5, iou=0.5)

for r in results:
    annotated_frame=r.plot()

annotated_frame=cv2.cvtColor(annotated_frame,cv2.COLOR_BGR2RGB)
    
plt.imshow(annotated_frame)
plt.axis("off")
plt.show()