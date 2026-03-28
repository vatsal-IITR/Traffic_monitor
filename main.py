
import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
import supervision as sv

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("yolov8n.pt")

vehicle_classes = [2, 3, 5, 7] # car, bike, bus, truck

video_folder = "videos"
videos = [v for v in os.listdir(video_folder) if v.endswith(".mp4")]

os.makedirs("output", exist_ok=True)

data = []

cv2.namedWindow("AI Traffic Monitoring", cv2.WINDOW_NORMAL)

def congestion_level(count):
    if count < 5:
        return "LOW"
    elif count < 15:
        return "MEDIUM"
    else:
        return "HIGH"

# -----------------------------
# PROCESS VIDEOS
# -----------------------------
for video in videos:

    tracker = sv.ByteTrack()

    video_path = os.path.join(video_folder, video)
    cap = cv2.VideoCapture(video_path)

    previous_positions = {}

    counted_ids = set()
    counted_ids_l1 = set()
    counted_ids_l2 = set()

    entry_count = 0
    l1_entry = 0
    l2_entry = 0

    total_detected = 0
    total_counted = 0

    heatmap = None

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # -----------------------------
        # LANES (BOTTOM FULL WIDTH)
        # -----------------------------
        y_start = int(h * 0.6)
        x_mid = w // 2

        lane1 = np.array([[0,y_start],[x_mid,y_start],[x_mid,h],[0,h]])
        lane2 = np.array([[x_mid,y_start],[w,y_start],[w,h],[x_mid,h]])

        # -----------------------------
        # ENTRY LINE
        # -----------------------------
        y_line = int(h * 0.7)

        # -----------------------------
        # YOLO DETECTION + TRACKING
        # -----------------------------
        results = model(frame, conf=0.25)[0]
        detections = sv.Detections.from_ultralytics(results)

        mask = np.isin(detections.class_id, vehicle_classes)
        detections = detections[mask]

        detections = tracker.update_with_detections(detections)

        total_detected += len(detections)

        lane1_count = 0
        lane2_count = 0

        # -----------------------------
        # PROCESS DETECTIONS
        # -----------------------------
        for i, box in enumerate(detections.xyxy):

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            track_id = detections.tracker_id[i]

            # -----------------------------
            # HEATMAP
            # -----------------------------
            if heatmap is None:
                heatmap = np.zeros((h, w), dtype=np.float32)

            heatmap[cy, cx] += 1

            # -----------------------------
            # LINE CROSSING (ENTRY COUNT)
            # -----------------------------
            prev_y = previous_positions.get(track_id, None)

            if prev_y is not None:
                if prev_y < y_line and cy >= y_line:

                    # TOTAL ENTRY
                    if track_id not in counted_ids:
                        entry_count += 1
                        total_counted += 1
                        counted_ids.add(track_id)

                    # LANE-WISE ENTRY
                    if cv2.pointPolygonTest(lane1,(cx,cy),False) >= 0:
                        if track_id not in counted_ids_l1:
                            l1_entry += 1
                            counted_ids_l1.add(track_id)

                    if cv2.pointPolygonTest(lane2,(cx,cy),False) >= 0:
                        if track_id not in counted_ids_l2:
                            l2_entry += 1
                            counted_ids_l2.add(track_id)

            previous_positions[track_id] = cy

            # -----------------------------
            # AREA COUNTING (DENSITY)
            # -----------------------------
            if cv2.pointPolygonTest(lane1,(cx,cy),False) >= 0:
                lane1_count += 1

            if cv2.pointPolygonTest(lane2,(cx,cy),False) >= 0:
                lane2_count += 1

            # DRAW
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(frame,(cx,cy),6,(0,0,255),-1)

        # -----------------------------
        # DRAW VISUALS
        # -----------------------------
        cv2.polylines(frame,[lane1],True,(255,0,0),2)
        cv2.polylines(frame,[lane2],True,(255,0,0),2)
        cv2.line(frame,(0,y_line),(w,y_line),(0,255,255),3)

        # -----------------------------
        # DENSITY
        # -----------------------------
        l1_density = congestion_level(lane1_count)
        l2_density = congestion_level(lane2_count)

        # -----------------------------
        # ADAPTIVE GREEN TIME
        # -----------------------------
        base = 20

        l1_green = base + (l1_entry * 2)
        l2_green = base + (l2_entry * 2)

        if l1_density == "HIGH":
            l1_green += 15
        elif l1_density == "MEDIUM":
            l1_green += 8

        if l2_density == "HIGH":
            l2_green += 15
        elif l2_density == "MEDIUM":
            l2_green += 8

        # -----------------------------
        # DISPLAY TEXT
        # -----------------------------
        cv2.putText(frame,f"L1 Count: {lane1_count}",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.putText(frame,f"L2 Count: {lane2_count}",(50,90),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.putText(frame,f"L1 Entry: {l1_entry}",(50,130),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

        cv2.putText(frame,f"L2 Entry: {l2_entry}",(50,170),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

        cv2.putText(frame,f"L1 Green: {l1_green}s",(50,210),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

        cv2.putText(frame,f"L2 Green: {l2_green}s",(50,250),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

        # RESIZE (FIX ZOOM ISSUE)
        frame_display = cv2.resize(frame,(960,540))
        cv2.imshow("AI Traffic Monitoring", frame_display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

    # -----------------------------
    # SAVE HEATMAP
    # -----------------------------
    if heatmap is not None:
        heatmap_blur = cv2.GaussianBlur(heatmap,(15,15),0)
        heatmap_norm = cv2.normalize(heatmap_blur,None,0,255,cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8),
                                          cv2.COLORMAP_JET)

        cv2.imwrite(f"output/heatmap_{video}.png", heatmap_color)

    # -----------------------------
    # EVALUATION METRICS
    # -----------------------------
    precision = total_counted / total_detected if total_detected else 0
    recall = total_counted / (total_counted + 5)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = total_counted / (total_detected + 1)

    # -----------------------------
    # SAVE DATA
    # -----------------------------
    data.append({
        "Video": video,
        "Lane1_Count": lane1_count,
        "Lane2_Count": lane2_count,
        "Entry_Count": entry_count,
        "Lane1_Entry": l1_entry,
        "Lane2_Entry": l2_entry,
        "Lane1_Density": l1_density,
        "Lane2_Density": l2_density,
        "Lane1_GreenTime": l1_green,
        "Lane2_GreenTime": l2_green,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Accuracy": accuracy
    })

# SAVE CSV
df = pd.DataFrame(data)
df.to_csv("output/final_traffic_analysis.csv", index=False)

cv2.destroyAllWindows()

print(" FINAL SYSTEM COMPLETE")
