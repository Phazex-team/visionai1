#!/usr/bin/env python3
import cv2

cap = cv2.VideoCapture('videos/NVR_ch10_main_20260109095150_20260109095555.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"Width: {width}")
print(f"Height: {height}")
print(f"FPS: {fps}")
print(f"Total frames: {frames}")

# Write to file
with open('/tmp/video_info.txt', 'w') as f:
    f.write(f"Width: {width}\n")
    f.write(f"Height: {height}\n")
    f.write(f"FPS: {fps}\n")
    f.write(f"Total frames: {frames}\n")
