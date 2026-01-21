#!/usr/bin/env python3
"""Quick YOLOE test"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

import cv2
import numpy as np

print("="*60)
print("YOLOE Quick Test")
print("="*60)

# Import
print("\n1. Importing YOLOEModel...")
from model_yoloe import YOLOEModel, COCO_CLASSES, RETAIL_RELEVANT_CLASSES
print(f"   COCO classes: {len(COCO_CLASSES)}")
print(f"   Retail relevant: {len(RETAIL_RELEVANT_CLASSES)}")

# Load model
print("\n2. Loading model...")
model_paths = [
    "models/weights/yoloe-11m-seg.pt",
    "models/weights/yolov8l.pt", 
    "models/weights/yolov8m.pt"
]

model = None
for path in model_paths:
    if os.path.exists(path):
        print(f"   Trying: {path}")
        model = YOLOEModel(model_name=path, device="cuda")
        if model.model_ready:
            print(f"   Loaded: {path}")
            break
        else:
            print(f"   Failed to load")

if not model or not model.model_ready:
    print("   ERROR: No model loaded")
    sys.exit(1)

# Configure
model.filter_retail = False  # Don't filter
model.set_thresholds(confidence=0.1, iou=0.5)

# Load test frame
print("\n3. Loading test frame...")
img_path = "zone_reference_frame.jpg"
if not os.path.exists(img_path):
    # Try video
    cap = cv2.VideoCapture("videos/NVR_ch10_main_20260109095150_20260109095555.mp4")
    ret, img = cap.read()
    cap.release()
else:
    img = cv2.imread(img_path)

if img is None:
    print("   ERROR: Could not load image")
    sys.exit(1)

print(f"   Shape: {img.shape}")

# Predict
print("\n4. Running inference...")
result = model.predict(img)

print(f"\n5. Results:")
print(f"   Boxes: {len(result['boxes'])}")
print(f"   Labels: {len(result['labels'])}")
print(f"   Inference time: {result['inference_time']:.1f}ms")

if len(result['labels']) > 0:
    print("\n   Detections:")
    for i, (label, score) in enumerate(zip(result['labels'][:20], result['scores'][:20])):
        print(f"   {i+1}. {label} ({score:.2f})")
else:
    print("\n   No detections!")
    
print("\n" + "="*60)
print("Test complete")
