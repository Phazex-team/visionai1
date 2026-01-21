#!/usr/bin/env python3
"""
Main Comparison Script - Compare Different Detection Models
Supports: GroundingDINO (ViT), YOLOWorld (CNN), OWLv2 (ViT), YOLOE (CNN) and more
"""
import sys
import os
import glob
import argparse
import numpy as np
from datetime import datetime

# ===== GPU MEMORY MANAGEMENT =====
# Set environment variables BEFORE importing torch/cuda libraries
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from model_grounding_dino import GroundingDINOModel
from model_yolo_world import YOLOWorldModel
from model_comparator import ModelComparator
from fraud_detector import FraudDetector
from zone_manager import ZoneManager

# Try importing newer models
try:
    from model_owlv2 import OWLv2Model
    HAS_OWLV2 = True
except ImportError:
    HAS_OWLV2 = False

try:
    from model_yoloe import YOLOEModel
    HAS_YOLOE = True
except ImportError:
    HAS_YOLOE = False

# Get the base directory (workspace root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration with absolute paths
VIDEO_PATH = os.path.join(BASE_DIR, "videos/NVR_ch10_main_20260109095150_20260109095555.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_results")

DETECTION_CLASSES = [
    "yellow oil bottle",
    "blue water pack",
    "red chips bag",
    "green box",
    "plastic bottle",
    "retail item"
]

def create_models(device="cuda"):
    """Create all models for comparison"""
    models = []
    
    # Vision Transformer Based Model
    # TEMPORARILY DISABLED FOR TESTING
    #try:
    #    dino_model = GroundingDINOModel(
    #        model_path=os.path.join(BASE_DIR, "models/GroundingDINO_SwinT_OGC.py"),
    #        weights_path=os.path.join(BASE_DIR, "weights/groundingdino_swint_ogc.pth"),
    #        device=device
    #    )
    #    dino_model.set_classes(DETECTION_CLASSES)
    #    dino_model.set_thresholds(box_threshold=0.25, text_threshold=0.20)
    #    models.append(dino_model)
    #    print("✅ GroundingDINO (ViT) loaded")
    #except Exception as e:
    #    print(f"❌ GroundingDINO error: {e}")
    
    # Traditional CNN Based Model
    try:
        yolo_model = YOLOWorldModel(
            model_name="models/weights/yolov8l-worldv2.pt",
            device=device
        )
        yolo_model.set_classes(DETECTION_CLASSES)
        yolo_model.set_thresholds(confidence=0.15, iou=0.5)
        models.append(yolo_model)
        print("✅ YOLOWorld (CNN) loaded")
    except Exception as e:
        print(f"❌ YOLOWorld error: {e}")
    
    # OWLv2 - Faster ViT Alternative
    if HAS_OWLV2:
        try:
            owlv2_model = OWLv2Model(device=device)
            owlv2_model.set_classes(DETECTION_CLASSES)
            owlv2_model.set_threshold(threshold=0.1)
            models.append(owlv2_model)
            print("✅ OWLv2 (ViT) loaded")
        except Exception as e:
            print(f"❌ OWLv2 error: {e}")
    
    # YOLOE - Extended YOLO (Real-Time Seeing Anything)
    # Using yoloe-11m-seg.pt (proper YOLOE model, not yolov9e)
    if HAS_YOLOE:
        try:
            yoloe_model = YOLOEModel(
                model_name="models/weights/yoloe-11m-seg.pt",
                device=device
            )
            yoloe_model.set_classes(DETECTION_CLASSES)
            yoloe_model.set_thresholds(confidence=0.15, iou=0.5)
            models.append(yoloe_model)
            print("✅ YOLOE (yoloe-11m-seg - CNN) loaded")
        except Exception as e:
            print(f"❌ YOLOE error: {e}")
    
    return models


def run_comparison(models, video_path, num_frames=None, skip_frames=1):
    """Run model comparison"""
    
    if not models:
        print("[ERROR] No models loaded. Exiting.")
        return None, None
    
    # Validate video path
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        print(f"[HELP] Available videos in /workspace/dino/videos/:")
        videos = glob.glob(os.path.join(BASE_DIR, "videos/*.mp4"))
        for v in videos:
            print(f"       {os.path.basename(v)}")
        return None, None
    
    comparator = ModelComparator(models, video_path, output_dir=OUTPUT_DIR)
    
    print(f"\n[SETUP] Video: {video_path}")
    print(f"[SETUP] Models: {[m.name for m in models]}")
    print(f"[SETUP] Output: {OUTPUT_DIR}")
    
    # Run comparison
    results = comparator.run_comparison(num_frames=num_frames, skip_frames=skip_frames)
    
    # Print results
    comparator.print_comparison_table()
    
    # Show rankings
    rankings = comparator.get_model_rankings()
    print("\n⚡ SPEED RANKINGS (Inference Time):")
    for i, (name, time_ms) in enumerate(rankings, 1):
        print(f"  {i}. {name}: {time_ms:.2f}ms")
    
    # Export results
    comparator.export_results("comparison_results.json")
    
    # Create side-by-side video
    try:
        num_frames_video = min(100, num_frames) if num_frames else 100
        print(f"\n[VIDEO] Attempting full-resolution video...")
        comparator.create_side_by_side_video("comparison_side_by_side.mp4", num_frames=num_frames_video)
    except Exception as e:
        print(f"[WARNING] Full-resolution video failed: {e}")
        print(f"[VIDEO] Trying scaled-down version...")
        try:
            comparator.create_side_by_side_video_simple("comparison_side_by_side_scaled.mp4", 
                                                       num_frames=num_frames_video, 
                                                       scale=0.5)
        except Exception as e2:
            print(f"[WARNING] Could not create any comparison video: {e2}")
    
    return comparator, results


def test_fraud_detection_compatibility(models):
    """Test if models work with fraud detection pipeline"""
    print(f"\n{'='*70}")
    print("Testing Fraud Detection Compatibility")
    print(f"{'='*70}\n")
    
    for model in models:
        print(f"\n[TEST] {model.name}")
        print(f"  Type: {model.model_type}")
        print(f"  Device: {model.device}")
        
        # Create mock frame
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Test prediction
        result = model.predict(mock_frame)
        print(f"  ✅ Inference time: {result['inference_time']:.2f}ms")
        print(f"  ✅ Output format compatible: boxes={len(result['boxes'])}, scores={len(result['scores'])}, labels={len(result['labels'])}")


def main():
    parser = argparse.ArgumentParser(description="Compare detection models for fraud detection")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames to process")
    parser.add_argument("--skip", type=int, default=1, help="Skip every n frames")
    parser.add_argument("--video", default=VIDEO_PATH, help="Video path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--test-fraud", action="store_true", help="Test fraud detection compatibility")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"FRAUD DETECTION MODEL COMPARISON")
    print(f"Vision Transformer (ViT) vs Traditional (CNN)")
    print(f"{'='*70}\n")
    
    # Load models
    print("[LOAD] Initializing models...")
    models = create_models(device=args.device)
    
    if not models:
        print("[ERROR] Failed to load any models")
        return 1
    
    # Test fraud detection compatibility
    if args.test_fraud:
        test_fraud_detection_compatibility(models)
    
    # Run comparison
    try:
        comparator, results = run_comparison(
            models,
            args.video,
            num_frames=args.frames,
            skip_frames=args.skip
        )
        
        if comparator is None or results is None:
            print("[ERROR] Comparison could not start")
            return 1
        
        print(f"\n{'='*70}")
        print("✅ COMPARISON COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {OUTPUT_DIR}/comparison_results.json")
        
    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
