from ultralytics import YOLO
import cv2
import os

# === CONFIG ===
MODEL_PATH = "best.pt"
VIDEO_PATH = "15sec_input_720p.mp4"
OUTPUT_PATH = "output/final_tracking.mp4"
CONF_THRESHOLD = 0.5

# === Load Model ===
print("Loading model...")
model = YOLO(MODEL_PATH)

# === Load Video ===
print("Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Could not open video at {VIDEO_PATH}")
    exit()

# === Get Video Properties ===
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Video Loaded: {width}x{height} @ {fps} FPS")

# === Prepare Output Folder ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Run YOLO Detection ===
    results = model(frame)[0]
    detections = results.boxes

    # === Draw Boxes ===
    for box in detections:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if conf < CONF_THRESHOLD or cls != 0:  # class 0 is person
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # === Write and Show ===
    out.write(frame)
    cv2.imshow("Tracking Preview", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

# === Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Done. Output saved to {OUTPUT_PATH}")
