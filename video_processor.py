import cv2
import json
from datetime import datetime
from collections import Counter
from ultralytics import YOLO

# Load YOLOv8 small model
model = YOLO("yolov8n.pt")  # Downloads automatically if not available

def process_video(video_path, json_path):
    events = []
    object_counts = Counter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # YOLO detection
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        names = results[0].names

        for box in boxes:
            cls_id = int(box.cls)
            label = names[cls_id]
            object_counts[label] += 1
            events.append({
                "timestamp": str(datetime.now()),
                "frame": frame_id,
                "event": f"{label.capitalize()} detected at frame {frame_id}"
            })

    cap.release()

    # Convert counts to summary
    summary_events = [{"object": obj, "count": count} for obj, count in object_counts.items()]

    with open(json_path, "w") as f:
        json.dump(summary_events, f, indent=4)
    
    return summary_events
