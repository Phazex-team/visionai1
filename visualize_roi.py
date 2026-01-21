#!/usr/bin/env python3
"""Visualize ROI and zones on video frame"""
import cv2
import yaml
import numpy as np

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

video_path = config['video_path']
roi = config['optimization']['roi_bounds']
zones = config.get('zones', {})

# Read first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read video")
    exit(1)

h, w = frame.shape[:2]
print(f'Video resolution: {w}x{h}')
print(f'ROI bounds: {roi}')

# Draw ROI rectangle (GREEN, thick)
x1, y1, x2, y2 = roi
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
cv2.putText(frame, f'ROI: {x2-x1}x{y2-y1}', (x1+10, y1+40), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

# Zone colors
zone_colors = {
    'counter': (255, 0, 0),    # Blue
    'scanner': (0, 0, 255),    # Red  
    'trolley': (255, 255, 0),  # Cyan
    'exit': (0, 165, 255)      # Orange
}

# Draw zones
for zone_name, points in zones.items():
    if points:
        pts = np.array(points, np.int32)
        color = zone_colors.get(zone_name, (128, 128, 128))
        cv2.polylines(frame, [pts], True, color, 3)
        cx = int(np.mean([p[0] for p in points]))
        cy = int(np.mean([p[1] for p in points]))
        cv2.putText(frame, zone_name.upper(), (cx-50, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# Save
output_path = 'roi_visualization.jpg'
cv2.imwrite(output_path, frame)
print(f'Saved: {output_path}')
