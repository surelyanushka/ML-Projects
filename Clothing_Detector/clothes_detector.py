import cv2
from ultralytics import YOLO

#Add your own model here
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.25, verbose=False)[0]

    annotated = results.plot()

    cv2.imshow("YOLOv8 Real-Time Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
