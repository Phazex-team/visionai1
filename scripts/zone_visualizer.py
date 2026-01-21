#!/usr/bin/env python3
"""
Zone Visualizer - Visualize and verify zone definitions on video frames
Helps ensure counter, exit, basket, pos, customer, cashier areas are correctly defined
"""
import cv2
import numpy as np
import json
import os
import sys

# Zone colors (BGR format)
ZONE_COLORS = {
    'counter': (0, 255, 0),       # Green
    'scanner': (255, 0, 0),       # Blue
    'pos': (0, 165, 255),         # Orange
    'exit': (0, 0, 255),          # Red
    'customer_area': (255, 255, 0),  # Cyan
    'trolley': (255, 0, 255),     # Magenta
    'basket': (128, 0, 128),      # Purple
    'baby_seat': (0, 128, 128),   # Olive
    'cashier': (255, 128, 0),     # Light Blue
    'packing_area': (0, 255, 255), # Yellow
}


def load_zones(config_path: str = 'zones_config.json') -> dict:
    """Load zones from config file"""
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return {}
    
    with open(config_path) as f:
        config = json.load(f)
    
    return config.get('zones', {})


def draw_zones_on_frame(frame: np.ndarray, zones: dict, alpha: float = 0.3) -> np.ndarray:
    """
    Draw all zones on a frame with semi-transparent fill and labels
    
    Args:
        frame: Input frame
        zones: Dict of zone_name -> list of polygon points
        alpha: Transparency (0-1)
    
    Returns:
        Frame with zones overlaid
    """
    overlay = frame.copy()
    output = frame.copy()
    
    for zone_name, points in zones.items():
        if not points:
            continue
        
        color = ZONE_COLORS.get(zone_name, (128, 128, 128))
        pts = np.array(points, np.int32)
        
        # Draw filled polygon
        cv2.fillPoly(overlay, [pts], color)
        
        # Draw polygon outline
        cv2.polylines(output, [pts], True, color, 2)
        
        # Add label at centroid
        M = cv2.moments(pts)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Draw label background
            label = zone_name.upper()
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output, (cx - 5, cy - label_h - 5), (cx + label_w + 5, cy + 5), color, -1)
            cv2.putText(output, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Blend overlay with output
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output


def visualize_zones_on_video(video_path: str, config_path: str = 'zones_config.json', output_path: str = None):
    """
    Create a video or image with zones visualized
    
    Args:
        video_path: Path to input video
        config_path: Path to zones config JSON
        output_path: Optional path for output image/video
    """
    # Load zones
    zones = load_zones(config_path)
    if not zones:
        print("❌ No zones found in config")
        return
    
    print(f"✅ Loaded {len(zones)} zones: {list(zones.keys())}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps}fps ({total_frames} frames)")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read frame")
        cap.release()
        return
    
    # Draw zones on frame
    annotated = draw_zones_on_frame(frame, zones)
    
    # Add legend
    y_offset = 30
    for zone_name, color in ZONE_COLORS.items():
        if zone_name in zones:
            cv2.rectangle(annotated, (10, y_offset - 15), (30, y_offset + 5), color, -1)
            cv2.putText(annotated, zone_name, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
    
    # Save output
    if output_path is None:
        output_path = 'zones_visualization.jpg'
    
    cv2.imwrite(output_path, annotated)
    print(f"✅ Saved zone visualization: {output_path}")
    
    cap.release()
    return annotated


def check_zone_coverage(zones: dict, frame_width: int, frame_height: int):
    """
    Check if zones cover expected areas and report issues
    """
    print("\n" + "="*60)
    print("ZONE COVERAGE CHECK")
    print("="*60)
    
    required_zones = ['counter', 'scanner', 'exit', 'customer_area', 'trolley']
    optional_zones = ['pos', 'basket', 'baby_seat', 'cashier']
    
    # Check required zones
    print("\n📋 Required Zones:")
    for zone in required_zones:
        if zone in zones and zones[zone]:
            pts = np.array(zones[zone])
            area = cv2.contourArea(pts)
            print(f"  ✅ {zone}: {len(zones[zone])} points, area={area:.0f}px²")
        else:
            print(f"  ❌ {zone}: MISSING!")
    
    # Check optional zones
    print("\n📋 Optional Zones:")
    for zone in optional_zones:
        if zone in zones and zones[zone]:
            pts = np.array(zones[zone])
            area = cv2.contourArea(pts)
            print(f"  ✅ {zone}: {len(zones[zone])} points, area={area:.0f}px²")
        else:
            print(f"  ⚪ {zone}: not defined")
    
    # Check for zone overlaps
    print("\n📋 Zone Overlaps (expected for some zones):")
    zone_names = list(zones.keys())
    for i, zone1 in enumerate(zone_names):
        for zone2 in zone_names[i+1:]:
            pts1 = np.array(zones[zone1], np.int32)
            pts2 = np.array(zones[zone2], np.int32)
            
            # Create masks
            mask1 = np.zeros((frame_height, frame_width), dtype=np.uint8)
            mask2 = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.fillPoly(mask1, [pts1], 255)
            cv2.fillPoly(mask2, [pts2], 255)
            
            # Check overlap
            overlap = cv2.bitwise_and(mask1, mask2)
            overlap_area = np.sum(overlap > 0)
            
            if overlap_area > 0:
                print(f"  ⚠️ {zone1} ↔ {zone2}: {overlap_area}px² overlap")


def create_zone_template(video_path: str, output_path: str = 'zones_template.json'):
    """
    Create a template zones config based on video dimensions
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create template with proportional zones
    template = {
        "video": os.path.basename(video_path),
        "frame_size": [width, height],
        "zones": {
            "counter": [
                [int(width * 0.3), int(height * 0.2)],
                [int(width * 0.7), int(height * 0.2)],
                [int(width * 0.7), int(height * 0.5)],
                [int(width * 0.3), int(height * 0.5)]
            ],
            "scanner": [
                [int(width * 0.4), int(height * 0.25)],
                [int(width * 0.6), int(height * 0.25)],
                [int(width * 0.6), int(height * 0.4)],
                [int(width * 0.4), int(height * 0.4)]
            ],
            "pos": [
                [int(width * 0.35), int(height * 0.3)],
                [int(width * 0.5), int(height * 0.3)],
                [int(width * 0.5), int(height * 0.45)],
                [int(width * 0.35), int(height * 0.45)]
            ],
            "exit": [
                [int(width * 0.7), int(height * 0.3)],
                [int(width * 0.95), int(height * 0.3)],
                [int(width * 0.95), int(height * 0.8)],
                [int(width * 0.7), int(height * 0.8)]
            ],
            "customer_area": [
                [int(width * 0.5), int(height * 0.4)],
                [int(width * 0.9), int(height * 0.4)],
                [int(width * 0.9), int(height * 0.9)],
                [int(width * 0.5), int(height * 0.9)]
            ],
            "trolley": [
                [int(width * 0.1), int(height * 0.5)],
                [int(width * 0.4), int(height * 0.5)],
                [int(width * 0.4), int(height * 0.9)],
                [int(width * 0.1), int(height * 0.9)]
            ],
            "basket": [
                [int(width * 0.15), int(height * 0.55)],
                [int(width * 0.35), int(height * 0.55)],
                [int(width * 0.35), int(height * 0.85)],
                [int(width * 0.15), int(height * 0.85)]
            ],
            "cashier": [
                [int(width * 0.25), int(height * 0.1)],
                [int(width * 0.45), int(height * 0.1)],
                [int(width * 0.45), int(height * 0.3)],
                [int(width * 0.25), int(height * 0.3)]
            ]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✅ Created zone template: {output_path}")
    print(f"   Video dimensions: {width}x{height}")
    print(f"   Edit the zones in the file to match your video layout")
    
    return template


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Zone Visualizer')
    parser.add_argument('--video', '-v', default='videos/NVR_ch10_main_20260109095150_20260109095555.mp4',
                        help='Path to video file')
    parser.add_argument('--config', '-c', default='zones_config.json',
                        help='Path to zones config JSON')
    parser.add_argument('--output', '-o', default='zones_visualization.jpg',
                        help='Output image path')
    parser.add_argument('--template', '-t', action='store_true',
                        help='Create a template zones config')
    parser.add_argument('--check', action='store_true',
                        help='Check zone coverage')
    
    args = parser.parse_args()
    
    if args.template:
        create_zone_template(args.video)
        return
    
    # Visualize zones
    visualize_zones_on_video(args.video, args.config, args.output)
    
    # Check zone coverage
    if args.check:
        zones = load_zones(args.config)
        cap = cv2.VideoCapture(args.video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        check_zone_coverage(zones, width, height)


if __name__ == '__main__':
    main()
