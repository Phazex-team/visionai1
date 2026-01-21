#!/usr/bin/env python3
"""Test face masking on a small number of frames to verify it works."""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from face_masking import FaceMasker, FaceMaskingConfig

print("Starting face mask quick test...", flush=True)

# Use Haar detector config
config = FaceMaskingConfig(
    enabled=True,
    detector_type='haar',
    detection_interval_frames=1,  # Detect every frame for test
    persistence_frames=5,
    mask_type='blur',
    blur_strength=51,
    async_enabled=False,
    enable_profile_detection=True
)

print(f"Creating masker with Haar detector...", flush=True)
masker = FaceMasker(config=config)

# Process video - only first 100 frames
video_path = '/workspace/dino/videos/NVR_ch10_main_20260109095150_20260109095555.mp4'
output_path = '/workspace/dino/videos/test_masked_100frames.mp4'

print(f"Opening video: {video_path}", flush=True)
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height} @ {fps}fps", flush=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

total_faces = 0
for frame_num in range(1, 101):
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {frame_num}", flush=True)
        break
    
    masked, count = masker.process_frame(frame, frame_num=frame_num)
    writer.write(masked)
    total_faces += count
    
    if frame_num % 10 == 0:
        print(f"Frame {frame_num}: {count} faces detected", flush=True)

cap.release()
writer.release()
masker.stop()

print(f"\nProcessed 100 frames, total face detections: {total_faces}", flush=True)
print(f"Output saved to: {output_path}", flush=True)

# Verify output
print("\nVerifying output...", flush=True)
cap_check = cv2.VideoCapture(output_path)
if cap_check.isOpened():
    frame_count = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Output has {frame_count} frames", flush=True)
    cap_check.release()
else:
    print("ERROR: Could not open output file!", flush=True)

print("Done!", flush=True)
