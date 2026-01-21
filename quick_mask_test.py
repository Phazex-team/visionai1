#!/usr/bin/env python3
"""Quick test to process just one video with face masking."""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from face_masking import FaceMasker, FaceMaskingConfig

# Use Haar detector config
config = FaceMaskingConfig(
    enabled=True,
    detector_type='haar',
    detection_interval_frames=2,
    persistence_frames=15,
    mask_type='blur',
    blur_strength=51,
    async_enabled=False,
    enable_profile_detection=True
)

print(f"Config: detector={config.detector_type}, interval={config.detection_interval_frames}")

# Create masker
masker = FaceMasker(config=config)

# Process video
video_path = Path('/workspace/dino/videos/NVR_ch10_main_20260109095150_20260109095555.mp4')
output_path = video_path.parent / (video_path.stem + '_masked.mp4')

print(f"Input: {video_path}")
print(f"Output: {output_path}")

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps}fps, {total} frames")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

frame_num = 0
total_faces = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_num += 1
    masked, count = masker.process_frame(frame, frame_num=frame_num)
    writer.write(masked)
    total_faces += count
    
    if frame_num % 500 == 0:
        print(f"Progress: {frame_num}/{total} ({100*frame_num//total}%), faces this batch: {count}")

cap.release()
writer.release()
masker.stop()

print(f"Done! Processed {frame_num} frames, total face detections: {total_faces}")
print(f"Output: {output_path}")
