# TillGuardAI Scan Discrepancy Evidence System - Complete Navigation Guide

## 📋 Project Summary

A comprehensive video evidence analysis and professional documentation system for identifying and reporting scan discrepancies in retail point-of-sale transactions.

**Status**: ✅ COMPLETE  
**Date**: January 28, 2026  
**System**: TillGuardAI v2.0

---

## 📁 File Organization

### 📚 Documentation Files (Root Directory)

| File | Purpose | Content |
|------|---------|---------|
| **PROJECT_COMPLETION_REPORT.md** | Executive summary and final report | Findings, deliverables, recommendations |
| **CODEBASE_ANALYSIS.md** | Complete system architecture analysis | Components, data flow, integration |
| **EVIDENCE_REPORT.md** | Detailed evidence findings and analysis | Visual specs, timestamps, recommendations |
| **QUICK_REFERENCE.md** | Usage guide and API reference | Examples, troubleshooting, API docs |
| **NAVIGATION_GUIDE.md** (this file) | Navigation and file organization | Finding files, learning path |

### 🐍 Python Tools

#### Root Directory
- **generate_evidence.py** - Standalone processor (~350 lines, no dependencies)
- **process_evidence.py** - Integrated pipeline (~200 lines)

#### Scripts Directory (`scripts/`)
- **scan_discrepancy_annotator.py** - Video annotation tool (~500 lines)
- **evidence_report_generator.py** - Report generation (~400 lines)

---

## 🎯 Identified Evidence

### Scan Discrepancies Found

**Incident 1: Malt Drink**
- Expected: 2 items | Scanned: 1 item | Missing: 1 item
- Time: 0:00 - 0:54 (54 seconds)
- Severity: HIGH

**Incident 2: Kinder Joy**
- Expected: 2 items | Scanned: 1 item | Missing: 1 item
- Time: 0:54 - 1:48 (54 seconds)
- Severity: HIGH

### Statistics
```
Total Items Expected:     4
Total Items Scanned:      2
Total Items Missing:      2
Discrepancy Rate:        50.00%
```

---

## 🚀 Quick Start (3 Options)

### Option 1: Standalone (Easiest)
```bash
python generate_evidence.py
```
Output: Annotated video + 3 reports in `evidence/` directory

### Option 2: With Full Features
```bash
python process_evidence.py
```
Output: Same as above, with integrated pipeline

### Option 3: Custom Processing
```python
from generate_evidence import create_evidence_package

result = create_evidence_package(
    video_path="videos/output.mp4",
    discrepancies=[...],
    output_dir="evidence"
)
```

---

## 📖 Documentation Reading Path

### For Different Audiences

**Managers/Investigators**
→ Read: PROJECT_COMPLETION_REPORT.md (~15 min)
→ Then: EVIDENCE_REPORT.md (~30 min)

**Developers/Operators**
→ Read: QUICK_REFERENCE.md (~10 min)
→ Then: generate_evidence.py (code review)

**Architects/Tech Leads**
→ Read: CODEBASE_ANALYSIS.md (~20 min)
→ Then: All Python files for implementation details

**Complete Understanding**
1. PROJECT_COMPLETION_REPORT.md (15 min) - Overview
2. EVIDENCE_REPORT.md (30 min) - Details
3. QUICK_REFERENCE.md (10 min) - Usage
4. CODEBASE_ANALYSIS.md (20 min) - Architecture
5. Code files (60+ min) - Implementation

---

## 🔍 Finding What You Need

### By Use Case

| Task | Start With | Then Read |
|------|-----------|-----------|
| Understand findings | EVIDENCE_REPORT.md | PROJECT_COMPLETION_REPORT.md |
| Generate evidence | QUICK_REFERENCE.md | generate_evidence.py |
| Review architecture | CODEBASE_ANALYSIS.md | Specific component docs |
| Integrate with system | QUICK_REFERENCE.md (API) | process_evidence.py |
| Customize visuals | QUICK_REFERENCE.md | scan_discrepancy_annotator.py |

### By Document Content

| Document | Key Sections |
|----------|--------------|
| PROJECT_COMPLETION_REPORT.md | Summary, Findings, Recommendations, Conclusion |
| EVIDENCE_REPORT.md | Visual annotations, Timestamps, Report specs, Findings |
| QUICK_REFERENCE.md | Tools overview, Usage examples, API reference, Troubleshooting |
| CODEBASE_ANALYSIS.md | Components, Data flow, Features, Integration points |
| generate_evidence.py | Complete standalone implementation |

---

## 📊 Generated Evidence Structure

```
evidence/
└── output_YYYYMMDD_HHMMSS/
    ├── annotated_evidence.mp4           (Video with annotations)
    ├── scan_discrepancy_report.json     (Machine-readable data)
    ├── SCAN_DISCREPANCY_REPORT.html     (Professional HTML report)
    ├── SCAN_DISCREPANCY_REPORT.txt      (Text documentation)
    └── processing_summary.json          (Metadata)
```

---

## 🔧 Tool Reference

### generate_evidence.py (Recommended for Simplicity)

**Features**:
- Works standalone (no dependencies except OpenCV)
- Simple API
- Complete output (video + 3 reports)
- Ideal for quick processing

**Usage**:
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
        }
    ]
)
```

---

### scan_discrepancy_annotator.py (Advanced Features)

**Features**:
- Customizable visual elements
- Professional formatting control
- Flexible timestamp generation
- Better for large-scale operations

**Usage**:
```python
from scripts.scan_discrepancy_annotator import ScanDiscrepancyAnnotator

annotator = ScanDiscrepancyAnnotator("videos/output.mp4")
annotator.add_discrepancy(...)
video = annotator.create_annotated_video()
```

---

### evidence_report_generator.py (Report Generation)

**Features**:
- HTML reports (responsive, professional)
- Text reports (plain documentation)
- Customizable styling

**Usage**:
```python
from scripts.evidence_report_generator import ProfessionalEvidenceReport

generator = ProfessionalEvidenceReport(json_data, "evidence")
html = generator.generate_html_report()
text = generator.generate_text_report()
```

---

### process_evidence.py (Full Integration)

**Features**:
- Complete pipeline
- All tools integrated
- Professional output

**Usage**:
```python
from process_evidence import process_scan_discrepancy_evidence

results = process_scan_discrepancy_evidence(
    video_path="videos/output.mp4",
    discrepancies_data=[...]
)
```

---

## 📈 Learning Path

### Beginner (30 minutes total)
1. **Read** (5 min): PROJECT_COMPLETION_REPORT.md intro
2. **Review** (10 min): EVIDENCE_REPORT.md "Key Findings"
3. **Run** (10 min): `python generate_evidence.py`
4. **Explore** (5 min): Generated evidence/ directory

### Intermediate (90 minutes total)
1. **Read** (15 min): PROJECT_COMPLETION_REPORT.md full
2. **Read** (30 min): EVIDENCE_REPORT.md full
3. **Study** (15 min): QUICK_REFERENCE.md
4. **Run** (15 min): Try different tool options
5. **Experiment** (15 min): Customize discrepancies

### Advanced (180+ minutes total)
1. **Read** (20 min): CODEBASE_ANALYSIS.md
2. **Study** (30 min): generate_evidence.py code
3. **Study** (30 min): scan_discrepancy_annotator.py code
4. **Study** (30 min): evidence_report_generator.py code
5. **Plan** (30 min): Integration approach
6. **Implement** (60+ min): Custom implementation

---

## 📋 Visual Annotations Reference

### Alert Banner
```
Position: Top of frame
Color: RED (0, 0, 255)
Text: "⚠️ SCAN DISCREPANCY DETECTED"
Format: White text on red background
```

### Item Details
```
ITEM: [Product Name]
Expected: [Count] | Scanned: [Count] | Missing: [Count]

Quantity Boxes:
  GREEN ▢▢    = Expected items (green boxes)
  BLUE  ▢✓▢   = Scanned items (blue with checkmark)
  RED   ▢✗    = Missing items (red with X)
```

### Timestamp
```
Position: Bottom-right corner
Format: MM:SS.mmm (minutes:seconds.milliseconds)
Example: Time: 00:54.000
Color: Green text
```

---

## ✅ What's Included

### Documentation (5 files)
- ✅ PROJECT_COMPLETION_REPORT.md
- ✅ CODEBASE_ANALYSIS.md
- ✅ EVIDENCE_REPORT.md
- ✅ QUICK_REFERENCE.md
- ✅ NAVIGATION_GUIDE.md (this file)

### Tools (4 files)
- ✅ generate_evidence.py
- ✅ process_evidence.py
- ✅ scripts/scan_discrepancy_annotator.py
- ✅ scripts/evidence_report_generator.py

### Evidence Output
- ✅ Annotated video (MP4)
- ✅ JSON report (structured data)
- ✅ HTML report (professional)
- ✅ Text report (documentation)
- ✅ Metadata (processing info)

---

## 🎓 Key Concepts

### Frame Ranges
Can be specified as:
- **Time-based** (0.0-1.0): `frame_range=(0, 0.3)` = first 30%
- **Frame numbers**: `frame_range=(0, 1620)` = frames 0-1620
- Automatically converted based on video FPS

### Timestamps
- **Format**: MM:SS.mmm (minutes:seconds.milliseconds)
- **Precision**: ±1 frame (at 30fps = 33.33ms)
- **Display**: Overlaid on video in bottom-right

### Severity Levels
- **HIGH**: Items missing from receipt (fraud confirmed)
- **MEDIUM**: Items in packing area without scan
- **LOW**: Suspicious activity, not yet confirmed

---

## 🔗 Cross-Reference Map

```
PROJECT_COMPLETION_REPORT.md
  ├─→ EVIDENCE_REPORT.md (detailed findings)
  ├─→ QUICK_REFERENCE.md (tools overview)
  └─→ CODEBASE_ANALYSIS.md (architecture)

EVIDENCE_REPORT.md
  ├─→ Visual marker specs (in this file)
  ├─→ Report format examples (in this file)
  └─→ QUICK_REFERENCE.md (timestamp usage)

QUICK_REFERENCE.md
  ├─→ Tool documentation (in this file)
  ├─→ API reference (in this file)
  ├─→ Usage examples (in this file)
  └─→ Code files (implementation)

CODEBASE_ANALYSIS.md
  ├─→ Component details
  ├─→ Data flow diagrams
  └─→ Integration guidelines
```

---

## 💾 File Locations

### Root Directory
```
F:\Development\TIllGuardAI\visionai1\
├── PROJECT_COMPLETION_REPORT.md
├── CODEBASE_ANALYSIS.md
├── EVIDENCE_REPORT.md
├── QUICK_REFERENCE.md
├── NAVIGATION_GUIDE.md (this file)
├── generate_evidence.py
├── process_evidence.py
└── videos/
    └── output.mp4
```

### Scripts Directory
```
F:\Development\TIllGuardAI\visionai1\scripts\
├── scan_discrepancy_annotator.py
├── evidence_report_generator.py
└── [other existing files]
```

### Evidence Output
```
F:\Development\TIllGuardAI\visionai1\evidence\
└── output_20260128_HHMMSS/
    ├── annotated_evidence.mp4
    ├── scan_discrepancy_report.json
    ├── SCAN_DISCREPANCY_REPORT.html
    ├── SCAN_DISCREPANCY_REPORT.txt
    └── processing_summary.json
```

---

## 🎯 Common Tasks

### Task: Understand the Evidence
**Time**: 30 minutes
1. Read: PROJECT_COMPLETION_REPORT.md
2. Read: EVIDENCE_REPORT.md (sections 1-3)
3. Review: Generated evidence files

### Task: Generate New Evidence
**Time**: 15 minutes
1. Read: QUICK_REFERENCE.md (first section)
2. Run: `python generate_evidence.py`
3. Review: Generated files in evidence/

### Task: Integrate With System
**Time**: 60 minutes
1. Read: QUICK_REFERENCE.md (API section)
2. Study: generate_evidence.py (code)
3. Create: Custom integration script
4. Test: With your data

### Task: Understand Architecture
**Time**: 60 minutes
1. Read: CODEBASE_ANALYSIS.md (full)
2. Study: All Python files
3. Review: Data flow diagrams
4. Map: Integration points

---

## 📞 Troubleshooting

### Can't Find Something?
1. Check this file's table of contents
2. Use Ctrl+F to search document
3. See "🔍 Finding What You Need" section

### Don't Know Where to Start?
1. Are you a developer? → QUICK_REFERENCE.md
2. Are you an investigator? → EVIDENCE_REPORT.md
3. Are you managing? → PROJECT_COMPLETION_REPORT.md
4. Are you technical? → CODEBASE_ANALYSIS.md

### Need Code Examples?
→ QUICK_REFERENCE.md (Usage Examples section)

### Need to Troubleshoot?
→ QUICK_REFERENCE.md (Troubleshooting section)

---

## 📊 Quick Stats

### Evidence Data
- Total items analyzed: **4**
- Discrepancies found: **2**
- Items missing: **2**
- Discrepancy rate: **50%**

### System Metrics
- Video resolution: **1920x1080**
- Frame rate: **30 FPS**
- Duration: **~3 minutes**
- Timestamp precision: **±1 frame**

### Documentation
- Total documents: **5**
- Total code files: **4**
- Code examples: **20+**
- Estimated read time: **75 minutes** (all docs)

---

## ✨ Key Features

✓ **Professional Evidence**: Legal-grade documentation  
✓ **Multiple Formats**: Video, JSON, HTML, Text  
✓ **Frame Accuracy**: ±1 frame (33.33ms at 30fps)  
✓ **Visual Markers**: Professional color-coded indicators  
✓ **Easy Integration**: Simple Python API  
✓ **No Dependencies**: Standalone tools available  
✓ **Complete Documentation**: 5 comprehensive guides  
✓ **Working Examples**: All tools are production-ready  

---

## 🎓 Next Steps

### Immediate (Today)
1. Read PROJECT_COMPLETION_REPORT.md
2. Review EVIDENCE_REPORT.md
3. Run generate_evidence.py

### Short-term (This Week)
1. Study QUICK_REFERENCE.md
2. Review all generated evidence files
3. Archive evidence package

### Medium-term (This Month)
1. Review CODEBASE_ANALYSIS.md
2. Plan system integration
3. Implement custom tools as needed

---

## 📝 Document Info

**Created**: January 28, 2026  
**System**: TillGuardAI v2.0  
**Status**: Production Ready ✅  
**Version**: 1.0  

---

## 🎯 Summary

This system provides a **complete solution** for:
- ✅ Identifying scan discrepancies
- ✅ Creating professional video evidence
- ✅ Generating multiple report formats
- ✅ Documenting findings
- ✅ Supporting legal proceedings

**All tools are ready to use and well-documented.**

---

*For navigation help, use the table of contents above.*  
*For quick access to tools, see the "🔧 Tool Reference" section.*  
*For reading recommendations, see the "📖 Documentation Reading Path" section.*

**Happy analyzing! 📊**
