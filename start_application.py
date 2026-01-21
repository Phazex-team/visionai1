#!/usr/bin/env python3
"""
Unified Detection Framework - Startup Guide & CLI Entry Point

Usage:
    python3 start_application.py --config config.yaml --video video.mp4
    python3 start_application.py --web  # Start Flask web UI
    python3 start_application.py --demo # Run demo with mock data
"""

import argparse
import sys
import os
from pathlib import Path

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from config_models import ApplicationConfig, DetectionConfig, OptimizationConfig, POSConfig, EvidenceConfig
from detection_pipeline import DetectionPipeline


def start_cli_mode(config_path: str, video_path: str = None):
    """Start detection pipeline in CLI mode for a single video"""
    print("\n" + "="*100)
    print("🚀 STARTING UNIFIED DETECTION FRAMEWORK - CLI MODE")
    print("="*100 + "\n")
    
    try:
        # Load config
        print(f"📖 Loading configuration from: {config_path}")
        config = ApplicationConfig.load_from_file(config_path)
        
        # Override video path if provided
        if video_path:
            config.video_path = video_path
            print(f"📝 Video path overridden to: {video_path}")
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Model: {config.detection.model_name}")
        print(f"   Video: {config.video_path}")
        print(f"   Device: {config.detection.device}")
        print(f"   POS Enabled: {config.pos.enabled}")
        print(f"   Evidence Enabled: {config.evidence.enabled}")
        
        # Validate video exists
        if not os.path.exists(config.video_path):
            print(f"❌ ERROR: Video file not found: {config.video_path}")
            return 1
        
        # Create and run pipeline
        print(f"\n⚙️  Initializing detection pipeline...")
        pipeline = DetectionPipeline(config)
        
        print(f"▶️  Starting video processing...\n")
        stats = pipeline.run_video(max_frames=None)
        
        # Print results
        print(f"\n{'='*100}")
        print("✅ PROCESSING COMPLETE")
        print(f"{'='*100}")
        print(f"📊 Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        return 1
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


def start_batch_mode(config_path: str, video_dir: str, pattern: str = "*.mp4"):
    """Process all videos in a directory"""
    print("\n" + "="*100)
    print("🚀 STARTING UNIFIED DETECTION FRAMEWORK - BATCH MODE")
    print("="*100 + "\n")
    
    import glob
    
    # Find all video files matching pattern
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"❌ ERROR: Directory not found: {video_dir}")
        return 1
    
    # Get all matching videos, exclude _output files
    video_files = sorted([
        f for f in video_dir.glob(pattern)
        if not f.name.endswith('_output.mp4') and f.is_file()
    ])
    
    if not video_files:
        print(f"❌ ERROR: No video files found matching '{pattern}' in {video_dir}")
        return 1
    
    print(f"📁 Found {len(video_files)} video(s) to process:")
    for i, vf in enumerate(video_files, 1):
        print(f"   {i}. {vf.name}")
    print()
    
    results = []
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*100}")
        print(f"📹 PROCESSING VIDEO {i}/{len(video_files)}: {video_file.name}")
        print(f"{'='*100}")
        
        try:
            # Load config fresh for each video
            config = ApplicationConfig.load_from_file(config_path)
            config.video_path = str(video_file)
            
            # Create and run pipeline
            pipeline = DetectionPipeline(config)
            stats = pipeline.run_video(max_frames=None)
            
            results.append({
                'video': video_file.name,
                'status': 'success',
                'stats': stats
            })
            print(f"✅ Completed: {video_file.name}")
            
        except Exception as e:
            print(f"❌ Failed: {video_file.name} - {e}")
            results.append({
                'video': video_file.name,
                'status': 'failed',
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*100}")
    print("📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*100}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"   • {r['video']}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"   • {r['video']}: {r['error']}")
    
    return 0 if not failed else 1


def start_web_mode(config_path: str = None):
    """Start Flask web UI"""
    print("\n" + "="*100)
    print("🌐 STARTING UNIFIED DETECTION FRAMEWORK - WEB MODE")
    print("="*100 + "\n")
    
    try:
        from flask_preview_server_new import app, run_detection_loop
        
        # Load config
        if config_path:
            print(f"📖 Loading configuration from: {config_path}")
            config = ApplicationConfig.load_from_file(config_path)
        else:
            print(f"📖 Using default configuration")
            config = ApplicationConfig(
                video_path='',
                detection=DetectionConfig(model_name='yoloworld'),
                optimization=OptimizationConfig(model_name='yoloworld'),
                pos={'enabled': False},
                evidence={'enabled': True}
            )
        
        print(f"✅ Configuration loaded")
        print(f"   Model: {config.detection.model_name}")
        print(f"   Device: {config.detection.device}")
        
        print(f"\n🌐 Starting Flask web server...")
        print(f"\n   ╔════════════════════════════════════════════════╗")
        print(f"   ║  🌐 WEB INTERFACE READY                        ║")
        print(f"   ║  Open your browser and visit:                 ║")
        print(f"   ║  http://localhost:5000                        ║")
        print(f"   ║                                                ║")
        print(f"   ║  Features:                                     ║")
        print(f"   ║  • Upload video file for processing           ║")
        print(f"   ║  • Real-time detection overlay                ║")
        print(f"   ║  • Live fraud detection alerts                ║")
        print(f"   ║  • Evidence recording and reports             ║")
        print(f"   ║  • Statistics and metrics                     ║")
        print(f"   ║                                                ║")
        print(f"   ║  Press Ctrl+C in terminal to stop             ║")
        print(f"   ╚════════════════════════════════════════════════╝\n")
        
        # Run Flask app
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


def start_demo_mode():
    """Start demo with mock data"""
    print("\n" + "="*100)
    print("🎮 STARTING UNIFIED DETECTION FRAMEWORK - DEMO MODE")
    print("="*100 + "\n")
    
    try:
        from demo_unified_framework import demonstrate_unified_framework
        
        print("📚 Running integration demonstration with mock data...\n")
        demonstrate_unified_framework()
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


def start_test_mode():
    """Run test suite"""
    print("\n" + "="*100)
    print("🧪 STARTING UNIFIED DETECTION FRAMEWORK - TEST MODE")
    print("="*100 + "\n")
    
    try:
        from test_unified_framework import run_all_tests
        
        print("🧪 Running comprehensive test suite...\n")
        run_all_tests()
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


def create_example_config(output_path: str = 'config.yaml'):
    """Create example configuration file"""
    print(f"\n📝 Creating example configuration file: {output_path}\n")
    
    try:
        config = ApplicationConfig(
            video_path='videos/NVR_ch10_main_20260109095150_20260109095555.mp4',
            detection=DetectionConfig(
                model_name='yoloworld',
                confidence_threshold=0.5,
                iou_threshold=0.45,
                device='cuda'
            ),
            optimization=OptimizationConfig(
                model_name='yoloworld',
                roi_bounds=[100, 100, 1820, 980],
                max_dim=1024,
                skip_every_n_frames=2,
                target_fps=15
            ),
            pos={
                'enabled': True,
                'data_format': 'xml',
                'xml_path': 'videos/2190_9_45748_20260109095013.xml',
                'match_strategy': 'fuzzy',
                'min_match_confidence': 0.75
            },
            evidence={
                'enabled': True,
                'record_mode': 'fraud_with_buffer',
                'buffer_seconds_before': 5,
                'buffer_seconds_after': 10,
                'save_frames': True,
                'save_video_clips': True,
                'video_codec': 'mp4v'
            },
            zones={ 
                'counter': [[100, 100], [1820, 100], [1820, 400], [100, 400]],
                'scanner': [[300, 100], [600, 100], [600, 300], [300, 300]],
                'trolley': [[100, 400], [1820, 400], [1820, 980], [100, 980]],
                'exit': [[1600, 500], [1820, 500], [1820, 800], [1600, 800]]
            },
            fps=30.0
        )
        
        config.save_to_file(output_path)
        print(f"✅ Example configuration created: {output_path}")
        print(f"\n📋 Configuration contents:")
        print(f"   - Video: {config.video_path}")
        print(f"   - Model: {config.detection.model_name}")
        print(f"   - Device: {config.detection.device}")
        print(f"   - POS: {config.pos.get('enabled', False)}")
        print(f"   - Evidence: {config.evidence.get('enabled', False)}")
        print(f"\nEdit this file and run:")
        print(f"   python3 start_application.py --config {output_path} --video your_video.mp4")
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Unified Detection Framework - Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with configuration file
  python3 start_application.py --config config.yaml --video video.mp4
  
  # Process all videos in a folder
  python3 start_application.py --config config.yaml --video-dir videos/
  
  # Process only specific pattern
  python3 start_application.py --config config.yaml --video-dir videos/ --pattern "NVR_ch10*.mp4"
  
  # Start web UI
  python3 start_application.py --web
  
  # Run demo
  python3 start_application.py --demo
  
  # Run tests
  python3 start_application.py --test
  
  # Create example config
  python3 start_application.py --create-config
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML/JSON)')
    parser.add_argument('--video', type=str, help='Override video path from config')
    parser.add_argument('--video-dir', type=str, help='Process all videos in a directory')
    parser.add_argument('--pattern', type=str, default='*.mp4', help='File pattern for --video-dir (default: *.mp4)')
    parser.add_argument('--web', action='store_true', help='Start Flask web UI mode')
    parser.add_argument('--demo', action='store_true', help='Run demo mode with mock data')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    parser.add_argument('--create-config', action='store_true', help='Create example configuration file')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.create_config:
        return create_example_config()
    elif args.web:
        return start_web_mode(args.config)
    elif args.demo:
        return start_demo_mode()
    elif args.test:
        return start_test_mode()
    elif args.video_dir and args.config:
        return start_batch_mode(args.config, args.video_dir, args.pattern)
    elif args.config:
        return start_cli_mode(args.config, args.video)
    else:
        print("\n⚠️  No mode specified. Use --help for options:\n")
        parser.print_help()
        print("\n🚀 Quick Start:")
        print("   1. Create config:  python3 start_application.py --create-config")
        print("   2. Run demo:       python3 start_application.py --demo")
        print("   3. Run tests:      python3 start_application.py --test")
        print("   4. Web UI:         python3 start_application.py --web")
        print("   5. Process video:  python3 start_application.py --config config.yaml --video video.mp4\n")
        return 0


if __name__ == '__main__':
    sys.exit(main())
