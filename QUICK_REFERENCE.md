# Quick Reference Guide - Scan Discrepancy Evidence System

## Overview

TillGuardAI now has a complete system for generating professional evidence packages for scan discrepancies. This guide shows how to use the tools.

## Tools Created

### 1. ScanDiscrepancyAnnotator
**File**: `scripts/scan_discrepancy_annotator.py`

Creates annotated videos with visual markers for discrepancies.

```python
from scripts.scan_discrepancy_annotator import ScanDiscrepancyAnnotator

# Initialize
annotator = ScanDiscrepancyAnnotator(
    video_path="videos/output.mp4",
    output_dir="evidence",
    fps=30,
    width=1920,
    height=1080
)

# Add discrepancies
annotator.add_discrepancy(
    item_name="Malt Drink",
    actual_quantity=2,
    scanned_quantity=1,
    frame_range=(0, 1620),  # frames 0-1620
    description="Cashier missed one item"
)

# Generate outputs
video = annotator.create_annotated_video()
report = annotator.create_evidence_report()
annotator.print_summary()
```

### 2. ProfessionalEvidenceReport
**File**: `scripts/evidence_report_generator.py`

Creates HTML and text reports.

```python
from scripts.evidence_report_generator import ProfessionalEvidenceReport
import json

# Load JSON data
with open("evidence/scan_discrepancy_report.json") as f:
    data = json.load(f)

# Create generator
generator = ProfessionalEvidenceReport(data, "evidence")

# Generate reports
html = generator.generate_html_report()
text = generator.generate_text_report()
```

### 3. Integrated Pipeline
**File**: `process_evidence.py`

Complete end-to-end processing.

```python
from process_evidence import process_scan_discrepancy_evidence

results = process_scan_discrepancy_evidence(
    video_path="videos/output.mp4",
    discrepancies_data=[
        {
            'item_name': 'Malt Drink',
            'actual_quantity': 2,
            'scanned_quantity': 1,
            'frame_range': (0, 0.3),  # 0-30% of video
            'description': 'Quantity discrepancy'
        },
        {
            'item_name': 'Kinder Joy',
            'actual_quantity': 2,
            'scanned_quantity': 1,
            'frame_range': (0.3, 0.6),  # 30-60% of video
            'description': 'Quantity discrepancy'
        }
    ],
    output_base_dir="evidence"
)

print(f"Evidence saved to: {results['output_directory']}")
```

### 4. Standalone Generator
**File**: `generate_evidence.py`

Simple standalone tool (no dependencies on other modules).

```python
from generate_evidence import create_evidence_package

result = create_evidence_package(
    video_path="videos/output.mp4",
    discrepancies=[
        {
            'item_name': 'Malt Drink',
            'actual_quantity': 2,
            'scanned_quantity': 1,
            'frame_range': (0, 0.3)
        },
        {
            'item_name': 'Kinder Joy',
            'actual_quantity': 2,
            'scanned_quantity': 1,
            'frame_range': (0.3, 0.6)
        }
    ],
    output_dir="evidence"
)
```

## Frame Range Specification

### Option 1: Time-based (0-1.0)
```python
frame_range=(0, 0.3)   # First 30% of video
frame_range=(0.3, 0.6) # Next 30% of video
```

### Option 2: Frame numbers
```python
frame_range=(0, 900)      # Frames 0-900 at 30fps = 30 seconds
frame_range=(900, 1800)   # Frames 900-1800
```

### Option 3: Time in seconds (if total_frames known)
```python
fps = 30
total_frames = 5400  # 3 minutes at 30fps

# Convert 0:00-0:54
start_frame = int(0 * fps)      # Frame 0
end_frame = int(54 * fps)       # Frame 1620
frame_range = (start_frame, end_frame)
```

## Output Structure

```
evidence/
└── output_TIMESTAMP/
    ├── annotated_evidence.mp4           # Annotated video
    ├── scan_discrepancy_report.json     # Structured data
    ├── SCAN_DISCREPANCY_REPORT.html     # Professional report
    ├── SCAN_DISCREPANCY_REPORT.txt      # Text documentation
    └── processing_summary.json          # Metadata
```

## Visual Markers

### Alert Banner (at top)
```
RED background
White text: "⚠️ SCAN DISCREPANCY DETECTED"
Location: Top of frame
```

### Item Details Panel
```
Item Name: [PRODUCT]
Expected: [COUNT] | Scanned: [COUNT] | Missing: [COUNT]

Visual Boxes:
  GREEN ▢▢    = Expected items
  BLUE  ▢✓▢   = Scanned items
  RED   ▢✗    = Missing items
```

### Timestamp
```
Format: MM:SS.mmm
Location: Bottom right of frame
Example: Time: 00:54.000
```

## Key Features

✓ **Automated Annotation**: Visual markers for every frame
✓ **Timestamp Accuracy**: Frame-level precision (±1 frame at 30fps)
✓ **Multiple Reports**: JSON, HTML, and text formats
✓ **Professional Design**: Legal-grade documentation
✓ **Easy Integration**: Simple Python API
✓ **Flexible**: Works with any video format

## Example Workflow

### Step 1: Identify Discrepancies
```
Review video: videos/output.mp4
Found: 2 items with quantity discrepancies
```

### Step 2: Record Details
```
Item 1: Malt Drink
  - In basket: 2
  - Scanned: 1
  - Missing: 1
  - Time: 0:00 - 0:54

Item 2: Kinder Joy
  - In basket: 2
  - Scanned: 1
  - Missing: 1
  - Time: 0:54 - 1:48
```

### Step 3: Process Evidence
```python
from generate_evidence import create_evidence_package

result = create_evidence_package(
    video_path="videos/output.mp4",
    discrepancies=[
        {
            'item_name': 'Malt Drink',
            'actual_quantity': 2,
            'scanned_quantity': 1,
            'frame_range': (0, 0.3)
        },
        {
            'item_name': 'Kinder Joy',
            'actual_quantity': 2,
            'scanned_quantity': 1,
            'frame_range': (0.3, 0.6)
        }
    ]
)
```

### Step 4: Review Outputs
```
evidence/output_20260128_153045/
├── annotated_evidence.mp4       ← Show to stakeholders
├── SCAN_DISCREPANCY_REPORT.html ← Professional documentation
└── scan_discrepancy_report.json ← System integration
```

## Timestamp Reference

### Common Timecode Conversions (30fps)
```
Frame Range  | Duration | Seconds
─────────────┼──────────┼─────────
0 - 900      | 30 sec   | 0:00-0:30
900 - 1800   | 30 sec   | 0:30-1:00
0 - 1620     | 54 sec   | 0:00-0:54
1620 - 3240  | 54 sec   | 0:54-1:48
0 - 5400     | 180 sec  | 0:00-3:00
```

## Report Examples

### Statistics (from current evidence)
```
Total Items Expected: 4
Total Items Scanned: 2
Total Items Missing: 2
Discrepancy Rate: 50.00%
```

### Discrepancy Details
```
[1] Malt Drink
    Expected: 2
    Scanned: 1
    Missing: 1
    Time: 0:00 - 0:54
    Status: ⚠️ UNSCANNED

[2] Kinder Joy
    Expected: 2
    Scanned: 1
    Missing: 1
    Time: 0:54 - 1:48
    Status: ⚠️ UNSCANNED
```

## Troubleshooting

### Video Not Found
```python
import os
if not os.path.exists("videos/output.mp4"):
    print("Please ensure output.mp4 exists in videos/ directory")
```

### Wrong Frame Range
```python
# Check total frames
cap = cv2.VideoCapture("videos/output.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames / fps
print(f"Total duration: {duration} seconds")

# Use time-based (0-1.0) for easier calculation
frame_range = (0, 0.3)  # First 30%
```

### Missing Dependencies
```bash
# Core dependencies
pip install opencv-python opencv-contrib-python

# For advanced features
pip install numpy scipy pillow
```

## Best Practices

1. **Frame Range Accuracy**: Use time-based ranges for easier calculation
2. **Item Descriptions**: Include detailed descriptions for clarity
3. **Report Review**: Always review HTML report before archival
4. **Evidence Preservation**: Keep original videos unchanged
5. **Documentation**: Archive both video and reports together

## Integration with Detection Pipeline

The scan discrepancy evidence system integrates with TillGuardAI's detection pipeline:

```
Detection Pipeline → Fraud Detection → Evidence Recorder
                                           ↓
                              (generates fraud_report.json)
                                           ↓
                    Scan Discrepancy Annotator
                                           ↓
                    (creates annotated evidence)
```

## API Reference

### ScanDiscrepancyAnnotator

```python
class ScanDiscrepancyAnnotator:
    def __init__(video_path, output_dir="evidence", fps=30, width=1920, height=1080)
    def add_discrepancy(item_name, actual_quantity, scanned_quantity, frame_range, description="")
    def create_annotated_video(output_video=None) -> str
    def create_evidence_report(output_file=None) -> str
    def print_summary()
```

### ProfessionalEvidenceReport

```python
class ProfessionalEvidenceReport:
    def __init__(discrepancy_data: Dict, output_dir: str)
    def generate_html_report(output_file=None) -> str
    def generate_text_report(output_file=None) -> str
```

### create_evidence_package (Standalone)

```python
def create_evidence_package(
    video_path: str,
    discrepancies: list,
    output_dir: str = "evidence"
) -> dict
```

Returns:
```python
{
    'status': 'SUCCESS',
    'evidence_dir': '/path/to/evidence',
    'files': {
        'video': '/path/to/annotated_evidence.mp4',
        'json': '/path/to/report.json',
        'html': '/path/to/report.html',
        'text': '/path/to/report.txt'
    },
    'statistics': {
        'total_expected': 4,
        'total_scanned': 2,
        'total_missing': 2
    }
}
```

---

**Version**: 1.0  
**Date**: January 28, 2026  
**System**: TillGuardAI v2.0
