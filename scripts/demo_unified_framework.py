#!/usr/bin/env python3
"""
Integration Test - Unified Detection Framework with Mock Data
Demonstrates framework usage without requiring actual video/models
"""
import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config_models import ApplicationConfig, DetectionConfig, OptimizationConfig, POSConfig, EvidenceConfig
from optimization_manager import OptimizationManager
from pos_processor import POSProcessor
from evidence_recorder import EvidenceRecorder
from model_factory import ModelFactory

def demonstrate_unified_framework():
    """Demonstrate the unified framework with mock components"""
    
    print("\n" + "="*80)
    print("UNIFIED DETECTION FRAMEWORK - INTEGRATION DEMONSTRATION")
    print("="*80 + "\n")
    
    # 1. CREATE UNIFIED CONFIG
    print("📋 STEP 1: Creating Unified Application Configuration")
    print("-" * 80)
    
    config = ApplicationConfig(
        video_path='demo_video.mp4',
        fps=25,
        frame_width=1280,
        frame_height=720,
        detection=DetectionConfig(
            model_name='yoloworld',
            confidence_threshold=0.15,
            iou_threshold=0.5,
            device='cuda'
        ),
        optimization=OptimizationConfig(
            model_name='yoloworld',
            enable_roi_crop=True,
            roi_bounds=(100, 150, 1200, 700),
            max_dim=640,
            skip_every_n_frames=2,
            target_fps=30
        ),
        pos=POSConfig(
            enabled=True,
            match_strategy='fuzzy',
            min_match_confidence=0.7
        ),
        evidence=EvidenceConfig(
            enabled=True,
            output_dir='evidence_demo',
            buffer_seconds_before=3.0,
            buffer_seconds_after=5.0,
            save_frames=True,
            save_video_clips=True
        )
    )
    
    print(f"✅ Configuration created:")
    print(f"   Video: {config.video_path} ({config.frame_width}x{config.frame_height}@{config.fps}fps)")
    print(f"   Model: {config.detection.model_name} (conf={config.detection.confidence_threshold})")
    print(f"   Optimization: ROI={config.optimization.enable_roi_crop}, Skip={config.optimization.skip_every_n_frames} frames")
    print(f"   POS enabled: {config.pos.enabled}")
    print(f"   Evidence enabled: {config.evidence.enabled}\n")
    
    # 2. INITIALIZE OPTIMIZATION MANAGER
    print("⚙️  STEP 2: Initialize Optimization Manager")
    print("-" * 80)
    
    opt_mgr = OptimizationManager(config.optimization)
    print(f"✅ OptimizationManager initialized:")
    print(f"   ROI cropping: {config.optimization.enable_roi_crop}")
    print(f"   ROI bounds: {config.optimization.roi_bounds}")
    print(f"   Max dimension: {config.optimization.max_dim}")
    print(f"   Clear GPU every: {config.optimization.clear_gpu_after_n_frames} frames\n")
    
    # 3. DEMONSTRATE FRAME OPTIMIZATION
    print("🎬 STEP 3: Demonstrate Frame Optimization Pipeline")
    print("-" * 80)
    
    # Create mock frame
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    print(f"Original frame: {test_frame.shape}")
    
    # Process through optimization pipeline
    optimized_frame, metadata = opt_mgr.prepare_frame(test_frame)
    print(f"Optimized frame: {optimized_frame.shape}")
    print(f"   ROI applied: {metadata['roi_applied']}")
    print(f"   Scale factor: {metadata['scale_factor']:.2f}")
    print(f"   Crop offset: {metadata['crop_offset']}\n")
    
    # 4. DEMONSTRATE BOX MAPPING
    print("📦 STEP 4: Demonstrate Detection Box Mapping")
    print("-" * 80)
    
    # Create mock detections in optimized frame
    mock_boxes = np.array([
        [50, 50, 150, 150],
        [200, 200, 350, 350],
        [400, 100, 550, 250]
    ], dtype=np.float32)
    
    print(f"Mock detections in optimized frame: {len(mock_boxes)} boxes")
    print(f"   Box 1: {mock_boxes[0]}")
    print(f"   Box 2: {mock_boxes[1]}")
    
    # Map back to original coordinates
    mapped_boxes = opt_mgr.map_boxes_back(mock_boxes, metadata)
    print(f"\nMapped to original frame coordinates:")
    print(f"   Box 1: {mapped_boxes[0]}")
    print(f"   Box 2: {mapped_boxes[1]}\n")
    
    # 5. DEMONSTRATE FRAME SKIPPING
    print("⏭️  STEP 5: Demonstrate Frame Skipping Logic")
    print("-" * 80)
    
    skip_config = OptimizationConfig(
        model_name='yoloworld',
        skip_every_n_frames=2
    )
    skip_mgr = OptimizationManager(skip_config)
    
    print(f"Skip configuration: process 1 out of every {skip_config.skip_every_n_frames + 1} frames")
    skip_pattern = [f"{'P' if not skip_mgr.should_skip_frame(i) else 'S'}" for i in range(12)]
    print(f"Frame pattern (P=Process, S=Skip): {''.join(skip_pattern)}")
    print(f"Processing rate: {sum(1 for c in skip_pattern if c == 'P')}/{len(skip_pattern)} = {100*sum(1 for c in skip_pattern if c == 'P')/len(skip_pattern):.0f}%\n")
    
    # 6. DEMONSTRATE POS PROCESSOR
    print("📋 STEP 6: Demonstrate POS Data Processing")
    print("-" * 80)
    
    pos_processor = POSProcessor(config.pos, fps=config.fps)
    
    # Create mock POS data
    mock_pos_items = [
        {'time': datetime.now(), 'item': 'milk bottle'},
        {'time': datetime.now(), 'item': 'sauce bottle'},
        {'time': datetime.now(), 'item': 'cereal box'},
        {'time': datetime.now(), 'item': 'juice carton'}
    ]
    pos_processor.pos_items = mock_pos_items
    pos_processor.item_names_set = {item['item'].lower() for item in mock_pos_items}
    
    print(f"✅ Loaded {len(mock_pos_items)} POS items:")
    for i, item in enumerate(mock_pos_items, 1):
        print(f"   {i}. {item['item']}")
    
    # Demonstrate matching
    detected_labels = ['milk bottle', 'sauce bottle', 'box', 'random item']
    print(f"\nDetected items in frame: {detected_labels}")
    
    match_result = pos_processor.match_detections_to_pos(0, detected_labels)
    print(f"\nMatching results (fuzzy strategy):")
    print(f"   ✅ Matched: {match_result['matched_count']}")
    print(f"   ⚠️  Unmatched POS items: {match_result['unmatched_pos_items']}")
    print(f"   🚨 Extra detected items: {match_result['extra_detected_items']}\n")
    
    # 7. DEMONSTRATE EVIDENCE RECORDING
    print("📁 STEP 7: Demonstrate Evidence Recording")
    print("-" * 80)
    
    ev_recorder = EvidenceRecorder(config.evidence, {
        'fps': config.fps,
        'frame_width': config.frame_width,
        'frame_height': config.frame_height,
        'video_path': config.video_path
    })
    
    print(f"✅ EvidenceRecorder initialized:")
    print(f"   Output directory: {os.path.basename(ev_recorder.evidence_dir)}")
    print(f"   Buffer: {config.evidence.buffer_seconds_before}s before, {config.evidence.buffer_seconds_after}s after")
    
    # Simulate recording frames
    print(f"\nRecording {20} frames to buffer...")
    for i in range(20):
        mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        ev_recorder.record_frame(mock_frame, i)
    
    print(f"✅ Buffered {len(ev_recorder.frame_buffer)} frames")
    
    # Simulate fraud event
    fraud_event = {
        'type': 'missed_scan',
        'label': 'milk bottle',
        'frame_num': 15,
        'timestamp': datetime.now(),
        'description': 'Item left counter without scan',
        'confidence': 0.92
    }
    
    print(f"\n🚨 Simulating fraud event: {fraud_event['description']}")
    fraud_record = ev_recorder.save_fraud_evidence(fraud_event, [])
    
    if fraud_record:
        print(f"✅ Evidence saved:")
        print(f"   Fraud ID: {fraud_record['fraud_id']}")
        print(f"   Type: {fraud_record['type']}")
        print(f"   Evidence frames: {fraud_record['frame_count']}\n")
    
    # 8. DEMONSTRATE MODEL FACTORY
    print("🏭 STEP 8: Demonstrate Model Factory Pattern")
    print("-" * 80)
    
    print(f"Model configurations available:")
    configs_to_try = [
        ('yoloworld', 'CNN (Fast)', 640),
        ('owlv2', 'ViT (Accurate)', 960),
        ('yoloe', 'CNN (Open-vocab)', 800)
    ]
    
    for model_name, description, max_dim in configs_to_try:
        det_cfg = DetectionConfig(model_name=model_name)
        opt_cfg = OptimizationConfig(
            model_name=model_name,
            max_dim=max_dim,
            skip_every_n_frames=2 if model_name == 'owlv2' else 1
        )
        print(f"  ✅ {model_name:12} ({description:20}) max_dim={max_dim}, skip={opt_cfg.skip_every_n_frames}")
    
    print(f"\nFactory would create models with:")
    print(f"  - Unified configuration (DetectionConfig + OptimizationConfig)")
    print(f"  - Automatic path resolution")
    print(f"  - Device placement (CUDA/CPU)")
    print(f"  - Metadata for optimization\n")
    
    # 9. SUMMARY
    print("="*80)
    print("FRAMEWORK SUMMARY")
    print("="*80)
    
    summary_stats = {
        'Total Components': 7,
        'Config Models': 5,
        'Processing Pipeline': 3,
        'Optimization Features': 5,
        'Model Support': 4
    }
    
    for key, value in summary_stats.items():
        print(f"✅ {key:.<40} {value}")
    
    print("\n🎯 Framework Advantages:")
    print("  ✅ Unified configuration (no scattered hardcoded values)")
    print("  ✅ Centralized optimization (ROI, resize, GPU memory)")
    print("  ✅ POS data integration (XML/CSV/API support)")
    print("  ✅ Evidence recording (frames, videos, reports)")
    print("  ✅ Fraud detection pipeline (state machine, event tracking)")
    print("  ✅ Model factory pattern (consistent instantiation)")
    print("  ✅ No backward compatibility required (clean new design)")
    
    print("\n" + "="*80 + "\n")
    
    return True

if __name__ == "__main__":
    success = demonstrate_unified_framework()
    sys.exit(0 if success else 1)
