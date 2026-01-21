#!/usr/bin/env python3
"""
Safe Test Script - Test model comparison with error reporting
"""
import os
import sys

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

print("="*70)
print("FRAUD DETECTION - MODEL COMPARISON TEST")
print("="*70)

# Test 1: File checks
print("\n[TEST 1] Checking required files...")
try:
    video_path = os.path.join(BASE_DIR, "videos/NVR_ch10_main_20260109095150_20260109095555.mp4")
    dino_config = os.path.join(BASE_DIR, "models/GroundingDINO_SwinT_OGC.py")
    dino_weights = os.path.join(BASE_DIR, "weights/groundingdino_swint_ogc.pth")
    
    checks = {
        "Video": video_path,
        "GroundingDINO Config": dino_config,
        "GroundingDINO Weights": dino_weights,
    }
    
    all_exist = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
        if not exists:
            print(f"      Missing: {path}")
            all_exist = False
    
    if not all_exist:
        print("\n[ERROR] Some files are missing. Cannot proceed.")
        sys.exit(1)
        
except Exception as e:
    print(f"[ERROR] File check failed: {e}")
    sys.exit(1)

# Test 2: Import checks
print("\n[TEST 2] Testing Python imports...")
try:
    from model_interface import DetectionModel
    print("  ✓ model_interface")
    
    from model_grounding_dino import GroundingDINOModel
    print("  ✓ model_grounding_dino")
    
    from model_yolo_world import YOLOWorldModel
    print("  ✓ model_yolo_world")
    
    from model_comparator import ModelComparator
    print("  ✓ model_comparator")
    
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load models (no inference, just loading)
print("\n[TEST 3] Loading models...")
try:
    print("  Loading GroundingDINO...")
    dino_model = GroundingDINOModel(
        model_path=dino_config,
        weights_path=dino_weights,
        device="cuda"
    )
    if dino_model.model is None:
        print("  ⚠ GroundingDINO loaded with None model - there may be an error")
    else:
        print("  ✓ GroundingDINO loaded successfully")
    
except Exception as e:
    print(f"  ✗ GroundingDINO load failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("  Loading YOLOWorld...")
    yolo_model = YOLOWorldModel(model_name="yolov8l-worldv2.pt", device="cuda")
    if yolo_model.model is None:
        print("  ⚠ YOLOWorld loaded with None model - there may be an error or model needs download")
    else:
        print("  ✓ YOLOWorld loaded successfully")
except Exception as e:
    print(f"  ✗ YOLOWorld load failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("BASIC TESTS COMPLETE")
print("="*70)
print("\nTo run full comparison, use:")
print("  cd /workspace/dino")
print("  python scripts/run_comparison.py --frames 10 --skip 1")
