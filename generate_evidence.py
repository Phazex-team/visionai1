"""
Standalone Scan Discrepancy Evidence Generator
Creates annotated video and professional reports for scan discrepancies
"""

import cv2
import json
import os
from datetime import datetime
from pathlib import Path


def create_evidence_package(
    video_path: str,
    discrepancies: list,
    output_dir: str = "evidence"
) -> dict:
    """
    Create complete evidence package with annotated video and reports.
    
    Args:
        video_path: Path to input video
        discrepancies: List of discrepancy dictionaries
        output_dir: Output directory
    
    Returns:
        Dictionary with generated file paths
    """
    
    # Verify video exists
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    # Create output structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_name = Path(video_path).stem
    evidence_dir = os.path.join(output_dir, f"{video_name}_evidence_{timestamp}")
    os.makedirs(evidence_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("SCAN DISCREPANCY EVIDENCE GENERATION")
    print("="*80)
    print(f"\n📹 Input Video: {video_path}")
    print(f"📁 Output Directory: {evidence_dir}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📊 Video Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f}s")
    
    # Create video writer for annotated output
    output_video = os.path.join(evidence_dir, "annotated_evidence.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Normalize discrepancies (convert time-based to frame-based)
    normalized_discrepancies = []
    for disc in discrepancies:
        frame_range = disc.get('frame_range')
        if isinstance(frame_range, tuple) and len(frame_range) == 2:
            start, end = frame_range
            # If looks like seconds (< 1000), convert to frames
            if start < 1000:
                start = int(start * total_frames)
                end = int(end * total_frames)
            disc['frame_range'] = (start, end)
        normalized_discrepancies.append(disc)
    
    print(f"\n📋 Processing {len(normalized_discrepancies)} discrepancies...")
    
    # Process video
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check for active discrepancies
        active = [d for d in normalized_discrepancies 
                  if d['frame_range'][0] <= frame_num <= d['frame_range'][1]]
        
        if active:
            # Draw alert banner
            cv2.rectangle(frame, (20, 30), (600, 90), (0, 0, 255), -1)
            cv2.putText(frame, "SCAN DISCREPANCY DETECTED", (30, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Draw item details
            y_pos = 120
            for disc in active:
                item_name = disc['item_name']
                expected = disc['actual_quantity']
                scanned = disc['scanned_quantity']
                missing = expected - scanned
                
                # Item header
                cv2.putText(frame, f"ITEM: {item_name}", (40, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                y_pos += 35
                
                # Quantities
                qty_text = f"Expected: {expected} | Scanned: {scanned} | Missing: {missing}"
                cv2.putText(frame, qty_text, (40, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_pos += 40
                
                # Visual indicators
                box_size = 25
                spacing = 5
                x_start = 40
                
                # Expected (green)
                for i in range(expected):
                    x = x_start + i * (box_size + spacing)
                    cv2.rectangle(frame, (x, y_pos), (x + box_size, y_pos + box_size), 
                                (0, 255, 0), 2)
                
                # Scanned (blue)
                for i in range(scanned):
                    x = x_start + i * (box_size + spacing)
                    cv2.rectangle(frame, (x, y_pos + 35), (x + box_size, y_pos + 35 + box_size),
                                (255, 0, 0), 2)
                    cv2.putText(frame, "✓", (x + 6, y_pos + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Missing (red)
                for i in range(missing):
                    x = x_start + (scanned + i) * (box_size + spacing)
                    cv2.rectangle(frame, (x, y_pos + 35), (x + box_size, y_pos + 35 + box_size),
                                (0, 0, 255), 2)
                    cv2.putText(frame, "✗", (x + 7, y_pos + 53),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                y_pos += 100
        
        # Add timestamp
        timestamp_str = f"Time: {frame_num/fps:.2f}s"
        cv2.putText(frame, timestamp_str, (width - 300, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        
        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames", end='\r')
        
        frame_num += 1
    
    cap.release()
    out.release()
    
    print(f"\n✓ Annotated video saved: {output_video}")
    
    # Calculate summary statistics
    total_missing = sum(d['actual_quantity'] - d['scanned_quantity'] 
                        for d in normalized_discrepancies)
    total_expected = sum(d['actual_quantity'] for d in normalized_discrepancies)
    total_scanned = sum(d['scanned_quantity'] for d in normalized_discrepancies)
    
    # Create JSON report
    json_report = {
        'report_type': 'SCAN_DISCREPANCY_EVIDENCE',
        'generated_at': datetime.now().isoformat(),
        'video_source': video_path,
        'video_properties': {
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': total_frames / fps
        },
        'summary': {
            'total_discrepancies': len(normalized_discrepancies),
            'total_items_expected': total_expected,
            'total_items_scanned': total_scanned,
            'total_items_missing': total_missing,
            'discrepancy_rate': f"{(total_missing / total_expected * 100):.2f}%" if total_expected > 0 else "0%"
        },
        'discrepancies': normalized_discrepancies
    }
    
    json_path = os.path.join(evidence_dir, "scan_discrepancy_report.json")
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"✓ JSON report saved: {json_path}")
    
    # Create HTML report
    html_report = create_html_report(json_report, evidence_dir)
    
    # Create text report
    text_report = create_text_report(json_report, evidence_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("EVIDENCE SUMMARY")
    print("="*80)
    print(f"\nTotal Expected Items: {total_expected}")
    print(f"Total Scanned Items: {total_scanned}")
    print(f"Total Missing Items: {total_missing}")
    print(f"Discrepancy Rate: {(total_missing / total_expected * 100):.2f}%")
    
    print(f"\nDiscrepancies:")
    for i, disc in enumerate(normalized_discrepancies, 1):
        start_time = disc['frame_range'][0] / fps
        end_time = disc['frame_range'][1] / fps
        missing = disc['actual_quantity'] - disc['scanned_quantity']
        print(f"  [{i}] {disc['item_name']}: {missing} missing (Time: {start_time:.2f}s - {end_time:.2f}s)")
    
    print("\n" + "="*80)
    print("✅ EVIDENCE PACKAGE CREATED")
    print("="*80)
    print(f"\n📁 All files saved to: {evidence_dir}")
    print(f"   - Annotated Video: {os.path.basename(output_video)}")
    print(f"   - JSON Report: {os.path.basename(json_path)}")
    print(f"   - HTML Report: {os.path.basename(html_report)}")
    print(f"   - Text Report: {os.path.basename(text_report)}")
    
    return {
        'status': 'SUCCESS',
        'evidence_dir': evidence_dir,
        'files': {
            'video': output_video,
            'json': json_path,
            'html': html_report,
            'text': text_report
        },
        'statistics': {
            'total_expected': total_expected,
            'total_scanned': total_scanned,
            'total_missing': total_missing
        }
    }


def create_html_report(data: dict, output_dir: str) -> str:
    """Create professional HTML evidence report"""
    
    total_missing = data['summary']['total_items_missing']
    total_expected = data['summary']['total_items_expected']
    total_scanned = data['summary']['total_items_scanned']
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scan Discrepancy Evidence Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; }}
        .header {{ border-bottom: 3px solid #d32f2f; padding-bottom: 20px; margin-bottom: 30px; }}
        .header h1 {{ color: #d32f2f; font-size: 32px; margin-bottom: 10px; }}
        .section {{ margin-bottom: 40px; }}
        .section-title {{ font-size: 24px; color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 10px; margin-bottom: 20px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
        .stat-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-box.warning {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .stat-value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th {{ background: #1976d2; color: white; padding: 15px; text-align: left; }}
        td {{ padding: 12px 15px; border-bottom: 1px solid #ddd; }}
        tr.missing-item {{ background: #ffebee; }}
        .missing-qty {{ color: #d32f2f; font-weight: bold; font-size: 16px; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 SCAN DISCREPANCY EVIDENCE REPORT</h1>
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <div class="section-title">📊 Summary Statistics</div>
            <div class="summary-grid">
                <div class="stat-box">
                    <div>Expected Items</div>
                    <div class="stat-value">{total_expected}</div>
                </div>
                <div class="stat-box">
                    <div>Scanned Items</div>
                    <div class="stat-value">{total_scanned}</div>
                </div>
                <div class="stat-box warning">
                    <div>Missing Items</div>
                    <div class="stat-value">{total_missing}</div>
                </div>
                <div class="stat-box">
                    <div>Discrepancy Rate</div>
                    <div class="stat-value">{(total_missing/total_expected*100):.1f}%</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📋 Detailed Discrepancies</div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Item Name</th>
                        <th>Expected</th>
                        <th>Scanned</th>
                        <th>Missing</th>
                        <th>Time Window</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for i, disc in enumerate(data['discrepancies'], 1):
        start_time = disc['frame_range'][0] / data['video_properties']['fps']
        end_time = disc['frame_range'][1] / data['video_properties']['fps']
        missing = disc['actual_quantity'] - disc['scanned_quantity']
        
        html += f"""
                    <tr class="missing-item">
                        <td>{i}</td>
                        <td>{disc['item_name']}</td>
                        <td>{disc['actual_quantity']}</td>
                        <td>{disc['scanned_quantity']}</td>
                        <td><span class="missing-qty">✗ {missing}</span></td>
                        <td>{start_time:.2f}s - {end_time:.2f}s</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p><strong>TillGuardAI</strong> - Retail Fraud Detection System</p>
            <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
    </div>
</body>
</html>"""
    
    output_file = os.path.join(output_dir, "SCAN_DISCREPANCY_REPORT.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_file


def create_text_report(data: dict, output_dir: str) -> str:
    """Create plain text evidence report"""
    
    total_missing = data['summary']['total_items_missing']
    total_expected = data['summary']['total_items_expected']
    total_scanned = data['summary']['total_items_scanned']
    
    text = f"""
{'='*80}
SCAN DISCREPANCY EVIDENCE REPORT
Official Video Evidence Documentation
{'='*80}

REPORT METADATA
{'─'*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type: SCAN_DISCREPANCY_EVIDENCE
Severity Level: HIGH
Status: COMPLETED

SUMMARY STATISTICS
{'─'*80}
Total Items Expected: {total_expected}
Total Items Scanned: {total_scanned}
Total Items Missing: {total_missing}
Discrepancy Rate: {(total_missing/total_expected*100):.2f}%

DETAILED DISCREPANCIES
{'─'*80}
"""
    
    for i, disc in enumerate(data['discrepancies'], 1):
        start_time = disc['frame_range'][0] / data['video_properties']['fps']
        end_time = disc['frame_range'][1] / data['video_properties']['fps']
        missing = disc['actual_quantity'] - disc['scanned_quantity']
        
        text += f"""
[{i}] {disc['item_name']}
    Expected: {disc['actual_quantity']}
    Scanned: {disc['scanned_quantity']}
    Missing: {missing}
    Time: {start_time:.2f}s - {end_time:.2f}s
    Status: ⚠️ UNSCANNED - FRAUD DETECTED
"""
    
    text += f"""
FINDINGS & RECOMMENDATIONS
{'─'*80}

KEY FINDINGS:
• Multiple items were not scanned during checkout
• Total of {total_missing} items missed across {len(data['discrepancies'])} products
• Scan discrepancy rate: {(total_missing/total_expected*100):.2f}%

RECOMMENDED ACTIONS:
1. Review checkout procedures with cashier
2. Provide additional scanner operation training
3. Implement additional verification steps
4. Monitor future transactions for patterns
5. File appropriate incident report

{'='*80}
Generated by TillGuardAI Fraud Detection System
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
    
    output_file = os.path.join(output_dir, "SCAN_DISCREPANCY_REPORT.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return output_file


if __name__ == "__main__":
    # Process the output.mp4 video with scan discrepancies
    result = create_evidence_package(
        video_path="videos/output.mp4",
        discrepancies=[
            {
                'item_name': 'Malt Drink',
                'actual_quantity': 2,
                'scanned_quantity': 1,
                'frame_range': (0, 0.3),
                'description': 'Cashier missed one Malt Drink during scanning'
            },
            {
                'item_name': 'Kinder Joy',
                'actual_quantity': 2,
                'scanned_quantity': 1,
                'frame_range': (0.3, 0.6),
                'description': 'Cashier missed one Kinder Joy during scanning'
            }
        ],
        output_dir="evidence"
    )
    
    if result:
        print(f"\n✅ Evidence package successfully created!")
        print(f"📍 Location: {result['evidence_dir']}")
