import cv2
import time
from ultralytics import YOLO

model = YOLO(r"C:\Users\prasa\runs\detect\train3\weights\best.pt")
cap = cv2.VideoCapture(0)
pTime = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. Advanced Prediction with Tracking
    # persist=True enables 'tracking IDs' so the AI remembers which player is which
    results = model.track(source=frame, persist=True, conf=0.4, show=False)

    # 2. Extract Data for Analytics
    for r in results:
        annotated_frame = r.plot() # Standard boxes
        player_count = 0
        
        if r.boxes.cls is not None:
            # Count only the 'batsman' class (usually index 0 or 1 in your data.yaml)
            player_count = (r.boxes.cls == 0).sum().item()

    # 3. Add Professional UI Overlays
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Players Active: {player_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Cricket Analytics Engine", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()