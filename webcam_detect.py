import cv2
import time
from ultralytics import YOLO

# Load your custom model
model = YOLO(r"weights/best.pt")

# Initialize camera and timing
cap = cv2.VideoCapture(0)
p_time = 0

print("Launching Analytics Engine... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO tracking (persist=True helps keep consistent IDs on players)
    results = model.track(source=frame, persist=True, conf=0.4, show=False)

    # 1. Create the Status Bar Background (Semi-transparent black bar)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    alpha = 0.6  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 2. Process Detections
    for r in results:
        annotated_frame = r.plot()  # Draw bounding boxes
        
        # Count objects
        batsman_count = 0
        bat_count = 0
        if r.boxes.cls is not None:
            classes = r.boxes.cls.cpu().numpy()
            # Note: Replace '0' and '1' with your actual class IDs from data.yaml
            batsman_count = (classes == 0).sum()
            bat_count = (classes == 1).sum()

    # 3. Calculate Performance (FPS)
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # 4. Draw Status Text
    # Status Indicator
    cv2.circle(annotated_frame, (30, 30), 10, (0, 255, 0), -1) # Green dot
    cv2.putText(annotated_frame, "SYSTEM ACTIVE", (50, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Live Counts
    status_text = f"Batsmen: {batsman_count} | Bats: {bat_count} | FPS: {int(fps)}"
    cv2.putText(annotated_frame, status_text, (frame.shape[1] - 450, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the final frame
    cv2.imshow("Cricket Detection System - Pro Version", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()