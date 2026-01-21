#!/usr/bin/env python3
"""Show ROI scaling calculation"""
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

video_w, video_h = 2688, 1520
roi = config['optimization']['roi_bounds']
roi_x1, roi_y1, roi_x2, roi_y2 = roi
roi_w, roi_h = roi_x2 - roi_x1, roi_y2 - roi_y1
max_dim = config['optimization']['max_dim']
scale = max_dim / max(roi_w, roi_h)
scaled_w, scaled_h = int(roi_w * scale), int(roi_h * scale)

print("="*65)
print("ROI SCALING CALCULATION")
print("="*65)
print()
print(f"STEP 1: Original Video")
print(f"        {video_w} x {video_h} pixels")
print()
print(f"STEP 2: ROI Crop (enable_roi_crop: true)")
print(f"        Bounds: ({roi_x1}, {roi_y1}) → ({roi_x2}, {roi_y2})")
print(f"        Cropped size: {roi_w} x {roi_h} pixels")
print()
print(f"STEP 3: max_dim Resize (max_dim: {max_dim})")
print(f"        Scale factor = {max_dim} / max({roi_w}, {roi_h})")
print(f"        Scale factor = {max_dim} / {max(roi_w, roi_h)} = {scale:.4f}")
print()
print(f"        Final processing size: {scaled_w} x {scaled_h} pixels")
print(f"        (This is what the model sees)")
print()
print("="*65)
print("COORDINATE MAPPING (automatic)")
print("="*65)
print()
print("Detection output → Original video coords:")
print()
print(f"  x_original = (x_detected ÷ {scale:.4f}) + {roi_x1}")
print(f"  y_original = (y_detected ÷ {scale:.4f}) + {roi_y1}")
print()
print("Example: If model detects box at (100, 200) in scaled image:")
print(f"  x_original = (100 ÷ {scale:.4f}) + {roi_x1} = {int(100/scale) + roi_x1}")
print(f"  y_original = (200 ÷ {scale:.4f}) + {roi_y1} = {int(200/scale) + roi_y1}")
print()
print("="*65)
