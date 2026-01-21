import cv2
import numpy as np
from ultralytics import YOLO

import os

os.chdir('/workspace/dino')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration with absolute paths
INPUT_VIDEO = os.path.join(BASE_DIR, "videos/NVR_ch10_main_20260109095150_20260109095555.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "comparison_results/smart_output.mp4")


   
CONFIDENCE_THRESHOLD = 0.5           

# Color Settings (HSV) - Example: Green to Blue
LOWER_HSV = np.array([35, 50, 50])   
UPPER_HSV = np.array([85, 255, 255]) 
NEW_HUE_VALUE = 110                  

def process_video():
    print("Loading AI Model...")
    # Using the X (Extra Large) model since you are on a powerful GPU
    model = YOLO('yolov8x.pt') 

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Cannot open {INPUT_VIDEO}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames. Please wait...")

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Run Object Detection
        results = model(frame, stream=True, classes=[0], verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Boundary checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # Extract Person ROI
                person_roi = frame[y1:y2, x1:x2]

                # Convert to HSV and Mask
                hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, LOWER_HSV, UPPER_HSV)
                
                # Clean noise
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                if cv2.countNonZero(mask) > 0:
                    h, s, v = cv2.split(hsv_roi)
                    h[mask > 0] = NEW_HUE_VALUE 
                    
                    new_hsv_roi = cv2.merge((h, s, v))
                    modified_roi = cv2.cvtColor(new_hsv_roi, cv2.COLOR_HSV2BGR)

                    # Blend modified ROI back into original frame
                    person_roi_bg = cv2.bitwise_and(person_roi, person_roi, mask=cv2.bitwise_not(mask))
                    person_roi_fg = cv2.bitwise_and(modified_roi, modified_roi, mask=mask)
                    combined_roi = cv2.add(person_roi_bg, person_roi_fg)

                    frame[y1:y2, x1:x2] = combined_roi

        # Write to file
        out.write(frame)

        # Print progress every 50 frames
        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Done! Saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    process_video()