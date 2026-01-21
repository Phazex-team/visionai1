#!/usr/bin/env python3
"""
Batch process videos to apply face masking.
Processes all MP4 files in the videos folder except output files.
"""

import cv2
import os
import sys
import shutil
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from face_masking import FaceMasker, FaceMaskingConfig, reset_face_masker
from config_models import ApplicationConfig

def main():
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    app_config = ApplicationConfig.load_from_file(str(config_path))
    
    face_config = app_config.face_masking
    
    print(f"[BatchFaceMask] Using config:")
    print(f"  detector_type={face_config.detector_type}  async_enabled={face_config.async_enabled}")
    print(f"  detection_interval_frames={face_config.detection_interval_frames}  persistence_frames={face_config.persistence_frames}")
    print(f"  mask_type={face_config.mask_type}  blur_strength={face_config.blur_strength}")
    
    # Find videos
    videos_dir = Path(__file__).parent / 'videos'
    video_files = sorted(videos_dir.glob('*.mp4'))
    
    # Filter out output and masked videos
    video_files = [
        v for v in video_files 
        if not v.name.endswith('_output.mp4') and not v.name.endswith('_masked.mp4')
    ]
    
    print(f"[BatchFaceMask] Found {len(video_files)} videos to process")
    
    for video_path in video_files:
        output_name = video_path.stem + '_masked.mp4'
        output_path = video_path.parent / output_name
        temp_path = video_path.parent / (output_name + '.__tmp__.mp4')
        
        print(f"[BatchFaceMask] Processing: {video_path.name} -> {output_name}")
        
        # Reset face masker for fresh tracking
        reset_face_masker()
        
        # Create masker with config
        masker = FaceMasker(config=face_config)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[BatchFaceMask] ERROR: Cannot open {video_path}")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create writer - use temp file then rename for atomic write
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"[BatchFaceMask] ERROR: Cannot create writer for {output_name}")
            cap.release()
            continue
        
        frame_num = 0
        last_pct = -1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Process frame
            masked_frame, face_count = masker.process_frame(frame, frame_num=frame_num)
            writer.write(masked_frame)
            
            # Progress
            pct = int(100 * frame_num / max(1, total_frames))
            if pct != last_pct and pct % 5 == 0:
                print(f"  {video_path.name}: {pct}% ({frame_num}/{total_frames})")
                last_pct = pct
        
        # Cleanup
        cap.release()
        writer.release()
        masker.stop()
        
        # Atomic rename
        if temp_path.exists():
            shutil.move(str(temp_path), str(output_path))
            print(f"[BatchFaceMask] Wrote: {output_name}")
        else:
            print(f"[BatchFaceMask] ERROR: Temp file not found: {temp_path}")
    
    print("[BatchFaceMask] All videos processed!")

if __name__ == '__main__':
    main()
