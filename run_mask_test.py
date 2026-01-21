#!/usr/bin/env python3
"""Quick test to process just one video with face masking - with proper error handling."""

import cv2
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from face_masking import FaceMasker, FaceMaskingConfig

def process_video(input_path, output_path):
    """Process a single video with face masking."""
    
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

    print(f"Config: detector={config.detector_type}, interval={config.detection_interval_frames}", flush=True)

    # Create masker
    masker = FaceMasker(config=config)

    print(f"Input: {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {input_path}", flush=True)
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total} frames", flush=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"ERROR: Cannot create writer", flush=True)
        cap.release()
        return False

    frame_num = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            masked, count = masker.process_frame(frame, frame_num=frame_num)
            writer.write(masked)
            
            if frame_num % 200 == 0:
                pct = 100 * frame_num // max(1, total)
                print(f"Progress: {frame_num}/{total} ({pct}%)", flush=True)
                
    except Exception as e:
        print(f"ERROR during processing: {e}", flush=True)
        traceback.print_exc()
    finally:
        print(f"Releasing resources...", flush=True)
        cap.release()
        writer.release()
        masker.stop()

    print(f"Done! Processed {frame_num} frames", flush=True)
    print(f"Output: {output_path}", flush=True)
    return True

if __name__ == '__main__':
    input_path = Path('/workspace/dino/videos/NVR_ch10_main_20260109095150_20260109095555.mp4')
    output_path = input_path.parent / (input_path.stem + '_masked.mp4')
    
    success = process_video(input_path, output_path)
    print(f"Success: {success}", flush=True)
