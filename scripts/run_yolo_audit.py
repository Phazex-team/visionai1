import cv2
import numpy as np
import supervision as sv
import os
import gc
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict
from ultralytics import YOLOWorld
import warnings

warnings.filterwarnings("ignore")

# ----------------- CONFIGURATION -----------------
# 1. PATHS
VIDEO_PATH = "videos/NVR_ch10_main_20260109095150_20260109095555.mp4"
XML_PATH = "videos/2190_9_45748_20260109095013.xml"
OUTPUT_PATH = "videos/yolo_audit_final.mp4"
SNAPSHOT_DIR = "videos/fraud_evidence"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# 2. TIME SYNC
VIDEO_START_DT = datetime(2026, 1, 9, 9, 51, 50) 
TIME_OFFSET_SECONDS = 0 

# 3. YOLO-WORLD CLASSES
# Define exactly what you see. YOLO-World is "Open Vocabulary" so you can type anything.
# "retail item" is a catch-all for things you didn't name specifically.
CUSTOM_CLASSES = [
    "yellow oil bottle", 
    "blue water pack", 
    "red chips bag", 
    "green box", 
    "hand", 
    "retail item" 
]

# 4. TUNING
CONFIDENCE_THRESHOLD = 0.15  # Low threshold to catch hidden items
IOU_THRESHOLD = 0.5          # NMS Threshold (lower = less overlapping boxes)
TARGET_WIDTH = 1280
SKIP_FRAMES = 3              # Process 1 out of every 3 frames (High FPS)

# 5. ZONES (Adjusted for your image view)
SCAN_ZONE_POLY = np.array([[400, 450], [880, 450], [880, 700], [400, 700]]) # Glass
BASKET_ZONE_POLY = np.array([[100, 400], [1100, 400], [1100, 720], [100, 720]]) # Belt
# -------------------------------------------------

def parse_pos_xml(xml_file):
    if not os.path.exists(xml_file): return [], "Unknown"
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        sales = []
        op_node = root.find('.//szTaCreatedEmplName')
        op_name = op_node.text.strip() if op_node is not None else "Unknown"
        
        for sale in root.findall('.//ART_SALE'):
            raw_date = sale.find('.//szTaCreatedDate').text
            item_desc = sale.find('.//szDesc1').text
            dt_obj = datetime.strptime(raw_date, "%Y%m%d%H%M%S")
            sales.append({'time': dt_obj, 'item': item_desc})
        return sales, op_name
    except: return [], "Unknown"

def get_center(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

# --- INITIALIZATION ---
cap = cv2.VideoCapture(VIDEO_PATH)
orig_w, orig_h = cap.get(3), cap.get(4)
target_h = int(TARGET_WIDTH * (orig_h / orig_w))
fps = cap.get(cv2.CAP_PROP_FPS) or 25

print("[INFO] Loading YOLO-World Model (First run will download weights)...")
# 'yolov8l-worldv2.pt' is the Large version. Use 'yolov8s-worldv2.pt' if you want even more speed.
model = YOLOWorld('models/weights/yolov8l-worldv2.pt') 
model.set_classes(CUSTOM_CLASSES)

pos_records, current_operator = parse_pos_xml(XML_PATH)

# Use ByteTrack from Supervision (Robust)
tracker = sv.ByteTrack(track_thresh=0.15, match_thresh=0.8, frame_rate=fps)

# Zones
scan_zone = sv.PolygonZone(polygon=SCAN_ZONE_POLY)
basket_zone = sv.PolygonZone(polygon=BASKET_ZONE_POLY)
scan_annotator = sv.PolygonZoneAnnotator(zone=scan_zone, color=sv.Color.RED, thickness=2)
basket_annotator = sv.PolygonZoneAnnotator(zone=basket_zone, color=sv.Color.BLUE, thickness=2)

# Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (TARGET_WIDTH, target_h))

# Stats
visual_unique_ids = set()
suspicion_streaks = defaultdict(int)
stats = {"theft": 0, "mismatch": 0}
frame_count = 0

print(f"[INFO] Audit Started. Operator: {current_operator}")

try:
    while True:
        # High-Performance Skip
        for _ in range(SKIP_FRAMES - 1): cap.grab()
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += SKIP_FRAMES
        frame = cv2.resize(frame, (TARGET_WIDTH, target_h))
        current_time = VIDEO_START_DT + timedelta(seconds=(frame_count/fps) + TIME_OFFSET_SECONDS)

        # --- YOLO INFERENCE (FAST) ---
        # conf=0.15 finds hidden items. iou=0.5 separates touching items.
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        
        # Convert to Supervision Format
        detections = sv.Detections.from_ultralytics(results[0])
        
        # --- TRACKING ---
        detections = tracker.update_with_detections(detections)
        
        # --- LOGIC ---
        if detections.tracker_id is not None:
            # Check Zones
            in_scan = scan_zone.trigger(detections=detections)
            in_basket = basket_zone.trigger(detections=detections)
            
            # Map Class IDs to Names for logic
            class_names = [CUSTOM_CLASSES[id] for id in detections.class_id]
            
            hand_indices = [i for i, name in enumerate(class_names) if name == "hand"]
            item_indices = [i for i, name in enumerate(class_names) if name != "hand"]

            # METHOD 1: BASKET COUNTING (The "Visual 7 vs POS 6" check)
            for i in item_indices:
                if in_basket[i]:
                    visual_unique_ids.add(detections.tracker_id[i])

            # METHOD 2: SCANNER THEFT CHECK
            for h_idx in hand_indices:
                if in_scan[h_idx]:
                    h_center = get_center(detections.xyxy[h_idx])
                    
                    # Proximity Check
                    has_item = False
                    for i_idx in item_indices:
                        dist = np.linalg.norm(h_center - get_center(detections.xyxy[i_idx]))
                        if dist < 150: # Pixels
                            has_item = True
                            break
                    
                    if has_item:
                        # Check POS XML (+/- 5 seconds)
                        match = next((r for r in pos_records if abs((r['time'] - current_time).total_seconds()) < 5), None)
                        
                        if not match:
                            suspicion_streaks[detections.tracker_id[h_idx]] += 1
                            if suspicion_streaks[detections.tracker_id[h_idx]] == 3: # Sensitivity
                                stats["theft"] += 1
                                cv2.imwrite(f"{SNAPSHOT_DIR}/theft_{frame_count}.jpg", frame)
                                cv2.rectangle(frame, (0,0), (TARGET_WIDTH, target_h), (0,0,255), 10) # Flash Red
                        else:
                            suspicion_streaks[detections.tracker_id[h_idx]] = 0

            # --- ANNOTATION ---
            # Create labels with Class Name + Tracker ID
            labels = [
                f"#{tid} {CUSTOM_CLASSES[cid]}" 
                for tid, cid in zip(detections.tracker_id, detections.class_id)
            ]
            
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # --- DASHBOARD ---
        pos_now = sum(1 for r in pos_records if r['time'] <= current_time)
        vis_now = len(visual_unique_ids)
        diff = vis_now - pos_now
        
        # Zones
        frame = scan_annotator.annotate(scene=frame)
        frame = basket_annotator.annotate(scene=frame)

        # UI
        cv2.rectangle(frame, (0, 0), (TARGET_WIDTH, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Visual Items: {vis_now}  |  POS Items: {pos_now}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        status_color = (0, 255, 0)
        status_txt = "OK"
        if diff > 0:
            status_color = (0, 0, 255)
            status_txt = f"MISMATCH: +{diff} Unpaid"
        
        cv2.putText(frame, f"Status: {status_txt}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Theft Alerts: {stats['theft']}", (600, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"Frame {frame_count} | Visual: {vis_now} | POS: {pos_now}")
            gc.collect()

except Exception as e: print(f"[ERROR] {e}")
finally:
    cap.release()
    out.release()
    print("[DONE] Video saved.")