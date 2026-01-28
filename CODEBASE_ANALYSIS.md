# TillGuardAI Video Processing Codebase Analysis

## Executive Summary

This analysis documents a sophisticated retail fraud detection system that processes video evidence to identify and record scanning discrepancies at point-of-sale terminals.

## System Architecture

### Core Components

#### 1. **Detection Pipeline** (`detection_pipeline.py`)
- **Purpose**: Main orchestrator for the complete detection workflow
- **Key Responsibilities**:
  - Model initialization and factory management
  - Performance optimization (ROI, resize, GPU memory)
  - POS data loading and temporal matching
  - Fraud detection coordination
  - Evidence recording management

- **Key Classes**:
  - `DetectionPipeline`: Main orchestrator
  - Coordinates: ModelFactory, OptimizationManager, POSProcessor, FraudDetectorV2, EvidenceRecorder

#### 2. **Fraud Detector V2** (`fraud_detector_v2.py`)
- **Purpose**: Advanced item tracking and fraud state machine
- **Key Features**:
  - Item state machine (NEW → SCANNED or FRAUD)
  - Movement tracking and touch detection
  - Zone-based analysis (counter, scanner, trolley, packing area)
  - Fraud type classification:
    - MISSED_SCAN: Item left without scanning
    - QUICK_PASS: Item briefly visible then disappears
    - CASHIER_SKIP: Cashier intentionally bypasses item
    - HAND_CONCEALMENT: Item hidden in hand/pocket
    - POCKET_CONCEALMENT: Item hidden in clothing
    - TROLLEY_HIDDEN: Hidden in trolley compartment
    - PACKING_AREA_UNSCANNED: Item in packing area without scan

- **Item States**:
  ```
  NEW → TOUCHED → ON_COUNTER → [SCANNED or FRAUD]
                            → LEFT_COUNTER → [SCANNED or FRAUD_CONFIRMED]
  ```

- **Key Classes**:
  - `ItemState`: Enum of item lifecycle states
  - `FraudType`: Classification of fraud patterns
  - `TrackedItemV2`: Individual item tracking with movement history

#### 3. **Evidence Recorder** (`evidence_recorder.py`)
- **Purpose**: Professional documentation of fraud events
- **Key Features**:
  - Frame buffering (before/after capture)
  - Video clip generation with annotations
  - Face masking for privacy compliance
  - Detailed fraud reporting with JSON export
  - Detection visualization with bounding boxes

- **Output Structure**:
  ```
  evidence/
  ├── fraud_report.json
  ├── frames/
  │   ├── fraud_001_frame_000123_001.jpg
  │   └── ...
  └── clips/
      ├── fraud_001.mp4
      └── ...
  ```

#### 4. **Model Factory & Registry** (`model_factory.py`)
- **Purpose**: Dynamic model instantiation and management
- **Supported Models**:
  - YOLOv8 (object detection)
  - YOLOv8-Pose (human pose detection)
  - SORT/DeepSORT (object tracking)
  - Face detection (MediaPipe/YOLO-Face)

- **Key Classes**:
  - `ModelFactory`: Creates model instances
  - `ModelRegistry`: Manages loaded models and their configurations

#### 5. **Optimization Manager** (`optimization_manager.py`)
- **Purpose**: Performance optimization and resource management
- **Optimization Techniques**:
  - ROI (Region of Interest) cropping
  - Frame resizing and quality adjustment
  - GPU memory management
  - Adaptive frame skipping
  - Batch processing support

#### 6. **POS Processor** (`pos_processor.py`)
- **Purpose**: Point-of-sale data integration
- **Capabilities**:
  - Receipt parsing
  - Item-timestamp matching
  - Scan event correlation
  - POS-video timeline synchronization

### Supporting Modules

#### Face Masking (`face_masking.py`)
- Privacy-preserving face blur/masking
- Async MediaPipe integration
- Configurable blur strength
- Temporal face tracking for consistency

#### Zone Management (`zone_manager.py`, `zone_visualizer.py`)
- Defines counter, scanner, trolley, packing areas
- Zone annotation and visualization
- Zone-based fraud detection rules

#### Smart Color Changer (`smart_color_changer.py`)
- Advanced color normalization
- Lighting condition adaptation
- Improves detection accuracy across varied conditions

## Data Flow

```
Video Input (MP4/AVI)
    ↓
Frame Extraction (OpenCV)
    ↓
Optional: ROI Cropping & Resize (OptimizationManager)
    ↓
Object Detection (YOLOv8)
    ↓
Object Tracking (SORT/DeepSORT)
    ↓
Item State Analysis (FraudDetectorV2)
    ↓
POS Data Correlation (POSProcessor)
    ↓
Fraud Classification & Scoring
    ↓
Evidence Recording (EvidenceRecorder)
    ├─ Frame Buffering
    ├─ Annotation Overlay
    ├─ Video Clip Generation
    ├─ Face Masking (Privacy)
    └─ Report Generation
    ↓
Output: Annotated Evidence + JSON Report
```

## Configuration System

### Config Sections

1. **Detection Config**
   - Model type and weights path
   - Confidence thresholds
   - Class definitions

2. **Optimization Config**
   - ROI parameters
   - GPU settings
   - Processing speed vs. accuracy tradeoff

3. **Evidence Config**
   - Buffer times (before/after fraud)
   - Output directory structure
   - Face masking settings
   - Video codec selection

4. **POS Config**
   - Receipt file format
   - Item-timestamp matching rules
   - Timezone handling

## Key Processing Features

### Real-time Item Tracking
- Tracks every item visible in frame
- Assigns unique track IDs
- Records movement patterns
- Detects state transitions

### Fraud Detection Triggers
1. **Grace Period**: Ignores items already on counter at startup
2. **Touch Detection**: Monitors item movement when picked up
3. **Scan Verification**: Checks POS scan events
4. **Exit Confirmation**: Confirms item left counter without scan
5. **Fraud Classification**: Determines fraud type and severity

### Evidence Quality
- Before/after frame capture (default: 3 seconds before, 2 seconds after)
- Professional annotation with:
  - Bounding boxes with confidence scores
  - Item labels and IDs
  - Timestamp overlays
  - Fraud type indicators
- Face masking for privacy compliance
- Multiple codec support (mp4v, h264, x264)

## Performance Characteristics

### Typical Performance
- **FPS**: 10-30 fps (depends on model and optimization)
- **Latency**: 100-500ms per frame (with buffering)
- **Memory Usage**: 2-6 GB GPU RAM
- **Storage**: ~500MB per hour of video (annotated evidence)

### Optimization Strategies
1. **ROI Cropping**: Process only checkout area
2. **Frame Skipping**: Skip frames for non-critical periods
3. **GPU Batching**: Process multiple frames in parallel
4. **Model Quantization**: Reduce model size/memory
5. **Resolution Scaling**: Lower resolution in low-activity zones

## Report Generation

### Output Formats

#### 1. JSON Report (`fraud_report.json`)
```json
{
  "timestamp": "2026-01-28T12:00:00",
  "video": "output.mp4",
  "total_frauds": 2,
  "fraud_records": [
    {
      "fraud_id": 0,
      "type": "missed_scan",
      "label": "Malt Drink",
      "frame_num": 250,
      "timestamp": "...",
      "description": "Item left counter without scan",
      "confidence": 0.95,
      "evidence_frames": [100, 110, 120, ...],
      "video_file": "clips/fraud_000.mp4",
      "roi_zone": "counter_exit",
      "roi_confirmed": true
    }
  ],
  "summary": {
    "by_type": {"missed_scan": 2},
    "by_label": {"Malt Drink": 1, "Kinder Joy": 1}
  }
}
```

#### 2. HTML Report
Professional visualization with:
- Summary statistics dashboard
- Interactive discrepancy tables
- Timeline visualization
- Quantity indicators with visual boxes
- Recommendations section
- Print-friendly styling

#### 3. Annotated Video
Professional evidence video with:
- Alert banners for discrepancies
- Item quantity indicators
- Bounding boxes for detections
- Timestamp overlays
- Color-coded severity indicators

## New Scan Discrepancy Evidence Tools

### ScanDiscrepancyAnnotator (`scan_discrepancy_annotator.py`)
Specialized tool for creating professional evidence videos from merchant video files:

**Features**:
- Adds visual markers for missed items
- Generates timestamp annotations
- Creates professional alert banners
- Produces quantity comparison visualizations
- Supports flexible frame range specification

**Output**:
- Annotated MP4 video
- JSON report with frame numbers and timing
- Professional documentation

### ProfessionalEvidenceReport (`evidence_report_generator.py`)
Creates HTML and text reports for evidence documentation:

**Features**:
- Professional HTML design with responsive layout
- Color-coded severity indicators
- Timeline visualization
- Evidence file tracking
- Print-friendly formatting

**Output**:
- HTML report (professional formatting)
- Text report (plain text documentation)
- Both formats are admissible as evidence

### Integrated Pipeline (`process_evidence.py`)
Complete end-to-end processing:

```python
process_scan_discrepancy_evidence(
    video_path="videos/output.mp4",
    discrepancies_data=[
        {
            'item_name': 'Malt Drink',
            'actual_quantity': 2,
            'scanned_quantity': 1,
            'frame_range': (0, 0.3),
            'description': 'Quantity discrepancy'
        }
    ],
    output_base_dir="evidence"
)
```

## Integration Points

### POS Integration
- Reads receipt data (JSON/XML)
- Matches items with detection timestamps
- Verifies scan events against video timeline
- Generates discrepancy reports

### Video Input
- Supports MP4, AVI, MOV, WebM
- Auto-detects frame rate and resolution
- Handles various codecs and quality levels

### Output Destinations
- Local filesystem storage
- Evidence directories with structured layout
- JSON export for SIEM/analytics systems
- HTML reports for stakeholder review
- Video evidence for legal proceedings

## Usage Recommendations

1. **For Real-time Monitoring**:
   - Use `DetectionPipeline` with optimizations enabled
   - Configure appropriate buffer times
   - Enable GPU acceleration

2. **For Evidence Documentation**:
   - Use `ScanDiscrepancyAnnotator` for merchant videos
   - Generate both HTML and JSON reports
   - Archive evidence with metadata

3. **For Integration**:
   - Parse JSON fraud reports
   - Implement webhook callbacks
   - Store evidence with blockchain timestamp (optional)

## Security & Privacy Considerations

1. **Face Masking**: Automatic privacy protection
2. **Audit Logging**: All processing logged with timestamps
3. **Evidence Integrity**: Original video preserved, annotations added to copies
4. **Access Control**: Evidence directories should be restricted
5. **Data Retention**: Configure based on retention policies

## Future Enhancement Opportunities

1. **Blockchain Timestamping**: Immutable evidence timestamps
2. **Multi-camera Correlation**: Track items across multiple angles
3. **Advanced Analytics**: ML-based fraud pattern recognition
4. **Real-time Alerts**: Webhook notifications during processing
5. **Cloud Integration**: AWS/Azure evidence storage
6. **3D Tracking**: Depth-based item tracking for complex scenarios

---

**Document Version**: 1.0  
**Date**: January 28, 2026  
**System**: TillGuardAI v2.0
