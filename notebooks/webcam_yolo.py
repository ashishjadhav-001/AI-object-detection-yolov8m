from ultralytics import YOLO
import cv2

model=YOLO('yolov8n.pt')

cap=cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = model(frame, conf=0.5)
    
    for r in results:
        annotated_frame = r.plot()
        
        # count objects
        boxes = r.boxes
        count = len(boxes)
        
        # display count
        cv2.putText(annotated_frame, f"Objects: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("YOLO Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()