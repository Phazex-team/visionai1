#!/usr/bin/env python3
"""Quick zone check script"""
import cv2
import json
import numpy as np
import os

os.chdir('/workspace/dino')

# Load zones
with open('zones_config.json') as f:
    config = json.load(f)

zones = config.get('zones', {})
print(f"Zones found: {list(zones.keys())}")

# Open video
cap = cv2.VideoCapture('videos/NVR_ch10_main_20260109095150_20260109095555.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {width}x{height} @ {fps}fps ({frames} frames)")

# Check each zone
print("\nZone details:")
for name, points in zones.items():
    pts = np.array(points)
    if len(pts) > 0:
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        print(f"  {name}: {len(points)} points, bounds=({min_x},{min_y})-({max_x},{max_y})")

# Read frame and draw zones
ret, frame = cap.read()
if ret:
    colors = {
        'counter': (0, 255, 0),
        'scanner': (255, 0, 0),
        'pos': (0, 165, 255),
        'exit': (0, 0, 255),
        'customer_area': (255, 255, 0),
        'trolley': (255, 0, 255),
        'basket': (128, 0, 128),
        'baby_seat': (0, 128, 128),
    }
    
    for name, points in zones.items():
        pts = np.array(points, np.int32)
        color = colors.get(name, (128, 128, 128))
        cv2.polylines(frame, [pts], True, color, 2)
        
        # Label
        M = cv2.moments(pts)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(frame, name.upper(), (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite('zones_visualization.jpg', frame)
    print(f"\nSaved: zones_visualization.jpg")

cap.release()
