# 🚀 Unified Detection Framework - Startup Guide

**Quick Start:** The application is now started using `start_application.py` in the root directory.

---

## 📋 Table of Contents
1. [Quick Start (30 seconds)](#quick-start)
2. [Installation & Setup](#installation--setup)
3. [Running Modes](#running-modes)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Create Configuration
```bash
cd /workspace/dino
python3 start_application.py --create-config
```
This creates `config.yaml` with example settings.

### 2. Run Demo (No Video Required)
```bash
python3 start_application.py --demo
```
Demonstrates all framework features with mock data.

### 3. Run Tests
```bash
python3 start_application.py --test
```
Executes 7 comprehensive tests (should all pass).

### 4. Start Web UI
```bash
python3 start_application.py --web
```
Starts Flask server at `http://localhost:5000`

### 5. Process Your Video
```bash
python3 start_application.py --config config.yaml --video /path/to/video.mp4
```
Processes video file and saves evidence.

---

## Installation & Setup

### Prerequisites
```bash
# Python 3.9+
python3 --version

# Install dependencies
pip install opencv-python torch torchvision numpy flask pyyaml
```

### Project Structure
```
/workspace/dino/
├── start_application.py          ⭐ ENTRY POINT
├── config.yaml                   (created after --create-config)
├── scripts/
│   ├── config_models.py         (configuration dataclasses)
│   ├── model_factory.py         (unified model creation)
│   ├── detection_pipeline.py    (main orchestrator)
│   ├── optimization_manager.py  (performance tuning)
│   ├── pos_processor.py         (POS integration)
│   ├── evidence_recorder.py     (evidence recording)
│   ├── flask_preview_server_new.py (web UI)
│   ├── test_unified_framework.py   (tests)
│   ├── demo_unified_framework.py   (demo)
│   ├── fraud_detector_v2.py     (fraud detection)
│   ├── model_interface.py       (base class)
│   └── ...more support files
├── evidence/                     (created during processing)
│   └── video_name_fraud_evidence_{timestamp}/
│       ├── frames/
│       ├── clips/
│       └── fraud_report.json
└── ...
```

---

## Running Modes

### Mode 1: CLI Processing
**Use when:** You have a video file to process

```bash
python3 start_application.py --config config.yaml --video retail.mp4
```

**Output:**
```
====================================================================================================
🚀 STARTING UNIFIED DETECTION FRAMEWORK - CLI MODE
====================================================================================================

📖 Loading configuration from: config.yaml
✅ Configuration loaded successfully
   Model: yoloworld
   Video: retail.mp4
   Device: cuda
   POS Enabled: True
   Evidence Enabled: True

⚙️  Initializing detection pipeline...
▶️  Starting video processing...

[Progress every 30 frames...]
Frame 30/1234 | FPS: 15.2 | Detections: 4 | Frauds: 1
Frame 60/1234 | FPS: 14.8 | Detections: 6 | Frauds: 1
...

====================================================================================================
✅ PROCESSING COMPLETE
====================================================================================================
📊 Statistics:
   total_frames: 1234
   detections: 245
   frauds_detected: 5
   pos_matches: 189
   processing_time: 82.5s
   average_fps: 14.9
```

**Output Location:**
- Evidence: `evidence/retail_fraud_evidence_20260112_143022/`
  - `frames/` - Individual fraud frames
  - `clips/` - Video clips of fraud events
  - `fraud_report.json` - Detailed fraud statistics

---

### Mode 2: Web UI
**Use when:** You want real-time dashboard and streaming

```bash
python3 start_application.py --web
```

**Output:**
```
====================================================================================================
🌐 STARTING UNIFIED DETECTION FRAMEWORK - WEB MODE
====================================================================================================

📖 Using default configuration
✅ Configuration loaded
   Model: yoloworld
   Device: cuda

🌐 Starting Flask server...
   URL: http://localhost:5000
   Video feed: http://localhost:5000/video_feed
   Statistics: http://localhost:5000/stats

   Press Ctrl+C to stop
```

**Web Interface Features:**
- Live video stream with detections overlay
- Real-time statistics (FPS, detections, frauds, inference time)
- POS matching display
- Evidence status
- Interactive dashboard

**Access:**
- Main UI: `http://localhost:5000`
- Video feed: `http://localhost:5000/video_feed`
- JSON stats: `http://localhost:5000/stats`

---

### Mode 3: Demo
**Use when:** Testing framework without video or to see all features

```bash
python3 start_application.py --demo
```

**Output:**
```
====================================================================================================
🎮 STARTING UNIFIED DETECTION FRAMEWORK - DEMO MODE
====================================================================================================

📚 Running integration demonstration with mock data...

📋 DEMO: Unified Detection Framework Integration
====================================================================

Step 1: Create ApplicationConfig with all components
  ✅ Config created with 5 top-level settings
     - Detection config: yoloworld model
     - Optimization config: ROI + resize enabled
     - POS config: XML format with fuzzy matching
     - Evidence config: fraud_with_buffer mode
     - FPS: 30.0

Step 2: Initialize OptimizationManager
  ✅ OptimizationManager initialized
     - ROI bounds: (100, 100, 1820, 980)
     - Max dimension: 640
     - Skip every: 2 frames
     - Target FPS: 15

Step 3: Demonstrate frame optimization pipeline
  ✅ Frame prepared for inference
     - Original shape: (1080, 1920, 3)
     - ROI crop applied: (200, 200, 400, 400)
     - Resized shape: (640, 640, 3) [aspect ratio preserved]
     - Transformation metadata stored

Step 4: Demonstrate detection box mapping
  ✅ Boxes mapped back to original coordinates
     - Original detections: 3 boxes in (640, 640)
     - Mapped to original: 3 boxes in (1920, 1080)

Step 5: Demonstrate frame skipping logic
  ✅ Frame skip logic validated
     - Processing frame 0: True (every N+1)
     - Processing frame 1: False (skip)
     - Processing frame 2: True (process again)

Step 6: Demonstrate POS processing with fuzzy matching
  ✅ POS items loaded and matched
     - Loaded 3 POS items
     - Matched detections: 2
     - Unmatched POS: 1
     - Fraud candidates: 1

Step 7: Demonstrate evidence recording
  ✅ Evidence recorded
     - Buffered frames: 45 (1.5 seconds at 30 FPS)
     - Fraud event saved
     - Evidence directory: evidence/test_fraud_evidence_20260112/

Step 8: Demonstrate model factory pattern
  ✅ Model factory created YOLOWorld model
     - Model name: yoloworld
     - Device: cuda
     - Config applied: ✅

====================================================================
✅ ALL DEMO STEPS COMPLETED SUCCESSFULLY
====================================================================
```

**No video required, demonstrates all framework features**

---

### Mode 4: Test Suite
**Use when:** Validating framework functionality

```bash
python3 start_application.py --test
```

**Output:**
```
====================================================================================================
🧪 STARTING UNIFIED DETECTION FRAMEWORK - TEST MODE
====================================================================================================

🧪 Running comprehensive test suite...

======================================================================
TEST SUMMARY
======================================================================
✅ Config Models: PASS
✅ Optimization Manager: PASS
✅ POS Processor: PASS
✅ Evidence Recorder: PASS
✅ Model Factory: PASS
✅ Model Registry: PASS
✅ Framework Integration: PASS

Total: 7 passed, 0 failed
======================================================================
```

**All tests should pass** if framework is correctly installed.

---

### Mode 5: Create Example Config
**Use when:** Setting up for the first time

```bash
python3 start_application.py --create-config
```

**Creates:** `config.yaml` with example settings

**Output:**
```
📝 Creating example configuration file: config.yaml

✅ Example configuration created: config.yaml

📋 Configuration contents:
   - Video: retail_video.mp4
   - Model: yoloworld
   - Device: cuda
   - POS: True
   - Evidence: True

Edit this file and run:
   python3 start_application.py --config config.yaml --video your_video.mp4
```

---

## Configuration

### Configuration File Format

**config.yaml:**
```yaml
video_path: retail_video.mp4

detection:
  model_name: yoloworld        # yoloworld | owlv2 | yoloe | groundingdino
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: cuda                 # cuda | cpu

optimization:
  roi_bounds: [100, 100, 1820, 980]  # [x1, y1, x2, y2]
  max_dim: 1024                       # 640-1280
  skip_every_n_frames: 2              # 1-3
  target_fps: 15

pos:
  enabled: true
  data_format: xml             # xml | csv | api
  pos_data_path: pos_data.xml
  match_strategy: fuzzy        # exact | substring | fuzzy
  min_match_confidence: 0.75

evidence:
  enabled: true
  record_mode: fraud_with_buffer  # all_frames | fraud_only | fraud_with_buffer
  buffer_seconds_before: 5
  buffer_seconds_after: 10
  save_frames: true
  save_videos: true
  video_codec: mp4v

zones:
  enabled: true
  counter: [[100, 100], [1820, 100], [1820, 400], [100, 400]]
  scanner: [[300, 100], [600, 100], [600, 300], [300, 300]]
  trolley: [[100, 400], [1820, 400], [1820, 980], [100, 980]]
  exit: [[1600, 500], [1820, 500], [1820, 800], [1600, 800]]

fps: 30.0
```

### Configuration Parameters

#### Detection Config
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `model_name` | yoloworld, owlv2, yoloe, groundingdino | Which model to use |
| `confidence_threshold` | 0.0-1.0 | Detection confidence cutoff |
| `iou_threshold` | 0.0-1.0 | NMS IOU threshold |
| `device` | cuda, cpu | GPU or CPU inference |

#### Optimization Config
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `roi_bounds` | [x1, y1, x2, y2] | Region of interest (crop area) |
| `max_dim` | 640-1280 | Max input dimension (20-30% speedup) |
| `skip_every_n_frames` | 1-3 | Process every Nth frame (30-50% speedup) |
| `target_fps` | 10-30 | Target processing FPS |

#### POS Config
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `enabled` | true/false | Enable POS matching |
| `data_format` | xml, csv, api | POS data source format |
| `pos_data_path` | filename or URL | Where to load POS data |
| `match_strategy` | exact, substring, fuzzy | Matching algorithm |
| `min_match_confidence` | 0.0-1.0 | Match confidence threshold |

#### Evidence Config
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `enabled` | true/false | Enable evidence recording |
| `record_mode` | all_frames, fraud_only, fraud_with_buffer | What to record |
| `buffer_seconds_before` | 1-10 | Seconds of pre-fraud video |
| `buffer_seconds_after` | 1-10 | Seconds of post-fraud video |
| `save_frames` | true/false | Save individual JPEG frames |
| `save_videos` | true/false | Save MP4 video clips |
| `video_codec` | mp4v, H264, etc. | Video codec |

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision -U
```

### Issue: "CUDA out of memory"
**Solution:**
Edit config.yaml:
```yaml
optimization:
  max_dim: 640        # Reduce from 1024
  skip_every_n_frames: 3  # Increase from 2
detection:
  device: cpu         # Switch to CPU temporarily
```

### Issue: Video file not found
**Solution:**
```bash
# Use absolute path
python3 start_application.py --config config.yaml --video /full/path/to/video.mp4

# Or place video in /workspace/dino/ and use relative path
python3 start_application.py --config config.yaml --video ./video.mp4
```

### Issue: "No detections found"
**Solution:**
- Increase `confidence_threshold` to 0.3 (more permissive)
- Check video resolution matches ROI bounds
- Verify model weights are loaded
- Try with `--demo` mode first

### Issue: Tests failing
**Solution:**
```bash
# Check individual test output
python3 start_application.py --test

# Run demo to verify framework works
python3 start_application.py --demo

# Check imports manually
python3 -c "from scripts.config_models import ApplicationConfig; print('OK')"
```

### Issue: Flask web UI not accessible
**Solution:**
```bash
# Check port is not in use
lsof -i :5000

# Use different port
# Edit start_application.py line with: app.run(..., port=5001)

# Check firewall
sudo ufw allow 5000  # Linux
```

### Issue: Slow performance
**Solution:**
Edit config.yaml:
```yaml
optimization:
  roi_bounds: [100, 100, 1820, 980]  # Crop to smaller region
  max_dim: 640                        # Smaller input
  skip_every_n_frames: 2              # Process fewer frames
detection:
  confidence_threshold: 0.6           # Stricter threshold = faster
```

---

## Common Workflows

### Workflow 1: First Time Setup
```bash
# 1. Create config
python3 start_application.py --create-config

# 2. Run demo to verify
python3 start_application.py --demo

# 3. Run tests
python3 start_application.py --test

# 4. Try web UI
python3 start_application.py --web
```

### Workflow 2: Process Multiple Videos
```bash
# Create config once
python3 start_application.py --create-config

# Edit config.yaml to set model, optimization, POS, evidence

# Process videos
python3 start_application.py --config config.yaml --video video1.mp4
python3 start_application.py --config config.yaml --video video2.mp4
python3 start_application.py --config config.yaml --video video3.mp4

# Results in: evidence/video1_fraud_evidence_{timestamp}/, etc.
```

### Workflow 3: Real-time Monitoring
```bash
# Start web UI (change video_path in flask_preview_server_new.py)
python3 start_application.py --web

# Visit http://localhost:5000
# See live stream, statistics, fraud detection
```

### Workflow 4: Model Comparison
```bash
# Edit config.yaml for each model
# yoloworld
python3 start_application.py --config config.yaml --video test.mp4

# owlv2
python3 start_application.py --config config.yaml --video test.mp4

# yoloe
python3 start_application.py --config config.yaml --video test.mp4

# Compare results in evidence directories
```

---

## Output Structure

### After Processing Video
```
/workspace/dino/
├── evidence/
│   └── retail_video_fraud_evidence_20260112_143022/
│       ├── frames/
│       │   ├── fraud_001_before.jpg
│       │   ├── fraud_001_during.jpg
│       │   └── fraud_001_after.jpg
│       ├── clips/
│       │   └── fraud_001.mp4
│       └── fraud_report.json
```

### fraud_report.json Structure
```json
{
  "video_file": "retail_video.mp4",
  "processing_time": 82.5,
  "total_frames": 1234,
  "fraud_events": [
    {
      "fraud_id": "fraud_001",
      "frame_range": [450, 480],
      "type": "missed_scan",
      "item_label": "apple",
      "confidence": 0.92,
      "evidence_frames": ["fraud_001_before.jpg", "fraud_001_during.jpg", "fraud_001_after.jpg"],
      "evidence_video": "fraud_001.mp4"
    }
  ],
  "statistics": {
    "total_frauds": 5,
    "by_type": {"missed_scan": 3, "customer_theft": 2},
    "by_label": {"apple": 2, "banana": 1, "milk": 2}
  }
}
```

---

## Next Steps

1. **Create configuration:** `python3 start_application.py --create-config`
2. **Test with demo:** `python3 start_application.py --demo`
3. **Try web UI:** `python3 start_application.py --web`
4. **Process your video:** `python3 start_application.py --config config.yaml --video your_video.mp4`
5. **Review evidence:** Check `evidence/` directory for results

---

**Ready to start?** Run: `python3 start_application.py --demo`
