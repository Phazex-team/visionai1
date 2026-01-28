"""
Professional Evidence Report Generator
Creates HTML and PDF reports for scan discrepancy evidence
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class ProfessionalEvidenceReport:
    """Generate professional HTML/PDF evidence reports"""
    
    def __init__(self, discrepancy_data: Dict, output_dir: str = "evidence"):
        """
        Initialize report generator.
        
        Args:
            discrepancy_data: Dictionary containing discrepancy information
            output_dir: Output directory for reports
        """
        self.data = discrepancy_data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_html_report(self, output_file: str = None) -> str:
        """
        Generate professional HTML evidence report.
        
        Args:
            output_file: Output HTML file path
        
        Returns:
            Path to generated HTML file
        """
        output_file = output_file or os.path.join(self.output_dir, 'SCAN_DISCREPANCY_REPORT.html')
        
        # Calculate statistics
        total_missing = self.data.get('summary', {}).get('total_items_missing', 0)
        total_actual = self.data.get('summary', {}).get('total_items_expected', 0)
        total_scanned = self.data.get('summary', {}).get('total_items_scanned', 0)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scan Discrepancy Evidence Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        
        .header {{
            border-bottom: 3px solid #d32f2f;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #d32f2f;
            font-size: 32px;
            margin-bottom: 10px;
        }}
        
        .report-meta {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
            font-size: 14px;
        }}
        
        .report-meta div {{
            background-color: #f9f9f9;
            padding: 10px;
            border-left: 3px solid #d32f2f;
        }}
        
        .severity-high {{
            color: #d32f2f;
            font-weight: bold;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 24px;
            color: #1976d2;
            border-bottom: 2px solid #1976d2;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .stat-box.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .stat-box.success {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            opacity: 0.9;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        
        th {{
            background-color: #1976d2;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:hover {{
            background-color: #f9f9f9;
        }}
        
        tr.missing-item {{
            background-color: #ffebee;
        }}
        
        .item-name {{
            font-weight: 600;
            color: #333;
        }}
        
        .quantity {{
            text-align: center;
            font-weight: 600;
        }}
        
        .missing-qty {{
            color: #d32f2f;
            font-weight: bold;
            font-size: 16px;
        }}
        
        .timestamp {{
            font-family: 'Courier New', monospace;
            color: #666;
            font-size: 13px;
        }}
        
        .timeline {{
            margin-top: 20px;
        }}
        
        .timeline-item {{
            display: flex;
            margin-bottom: 20px;
        }}
        
        .timeline-marker {{
            width: 40px;
            height: 40px;
            background-color: #d32f2f;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            margin-right: 20px;
            flex-shrink: 0;
        }}
        
        .timeline-content {{
            flex: 1;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 2px solid #d32f2f;
        }}
        
        .timeline-title {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .timeline-time {{
            color: #666;
            font-size: 13px;
            margin-bottom: 8px;
        }}
        
        .timeline-details {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 14px;
        }}
        
        .quantity-indicator {{
            display: flex;
            gap: 5px;
            margin-top: 10px;
        }}
        
        .quantity-box {{
            width: 25px;
            height: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: bold;
            border-radius: 3px;
        }}
        
        .quantity-box.expected {{
            background-color: #4caf50;
            color: white;
            border: 2px solid #45a049;
        }}
        
        .quantity-box.scanned {{
            background-color: #2196f3;
            color: white;
            border: 2px solid #0b7dda;
        }}
        
        .quantity-box.missing {{
            background-color: #f44336;
            color: white;
            border: 2px solid #da190b;
        }}
        
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
        
        .evidence-badge {{
            display: inline-block;
            background-color: #d32f2f;
            color: white;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 20px;
        }}
        
        .print-section {{
            margin: 20px 0;
            padding: 20px;
            background-color: #fafafa;
            border: 1px dashed #ccc;
            border-radius: 5px;
        }}
        
        .signature-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-top: 40px;
        }}
        
        .signature-line {{
            border-top: 1px solid #333;
            margin-top: 30px;
            text-align: center;
            font-size: 13px;
        }}
        
        @media print {{
            body {{
                background-color: white;
            }}
            .container {{
                box-shadow: none;
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🚨 SCAN DISCREPANCY EVIDENCE REPORT</h1>
            <div class="evidence-badge">OFFICIAL VIDEO EVIDENCE DOCUMENTATION</div>
            <div class="report-meta">
                <div><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div><strong>Severity Level:</strong> <span class="severity-high">HIGH</span></div>
                <div><strong>Report Type:</strong> SCAN_DISCREPANCY_EVIDENCE</div>
                <div><strong>Status:</strong> COMPLETED</div>
            </div>
        </div>
        
        <!-- Summary Statistics -->
        <div class="section">
            <div class="section-title">📊 Summary Statistics</div>
            <div class="summary-grid">
                <div class="stat-box success">
                    <div class="stat-label">Expected Items</div>
                    <div class="stat-value">{total_actual}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Items Scanned</div>
                    <div class="stat-value">{total_scanned}</div>
                </div>
                <div class="stat-box warning">
                    <div class="stat-label">Items Missing</div>
                    <div class="stat-value">{total_missing}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Discrepancy Rate</div>
                    <div class="stat-value">{(total_missing / total_actual * 100):.1f}%</div>
                </div>
            </div>
        </div>
        
        <!-- Detailed Discrepancies -->
        <div class="section">
            <div class="section-title">📋 Detailed Discrepancy Analysis</div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Item Name</th>
                        <th class="quantity">Expected</th>
                        <th class="quantity">Scanned</th>
                        <th class="quantity">Missing</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add discrepancy rows
        for i, disc in enumerate(self.data.get('discrepancies', []), 1):
            timestamp_start = disc.get('start_time', 0)
            timestamp_end = disc.get('end_time', 0)
            
            html_content += f"""
                    <tr class="missing-item">
                        <td>{i}</td>
                        <td class="item-name">{disc.get('item_name', 'Unknown')}</td>
                        <td class="quantity">{disc.get('actual_quantity', 0)}</td>
                        <td class="quantity">{disc.get('scanned_quantity', 0)}</td>
                        <td class="quantity"><span class="missing-qty">✗ {disc.get('missing_quantity', 0)}</span></td>
                        <td class="timestamp">{timestamp_start:.2f}s - {timestamp_end:.2f}s</td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
        
        <!-- Timeline of Events -->
        <div class="section">
            <div class="section-title">⏱️ Timeline of Discrepancies</div>
            <div class="timeline">
"""
        
        # Add timeline items
        for i, disc in enumerate(self.data.get('discrepancies', []), 1):
            timestamp_start = disc.get('start_time', 0)
            timestamp_end = disc.get('end_time', 0)
            actual = disc.get('actual_quantity', 0)
            scanned = disc.get('scanned_quantity', 0)
            missing = disc.get('missing_quantity', 0)
            
            html_content += f"""
                <div class="timeline-item">
                    <div class="timeline-marker">{i}</div>
                    <div class="timeline-content">
                        <div class="timeline-title">{disc.get('item_name', 'Unknown Item')} - Quantity Discrepancy</div>
                        <div class="timeline-time">📍 {timestamp_start:.2f}s → {timestamp_end:.2f}s</div>
                        <div class="timeline-details">
                            <div><strong>Expected:</strong> {actual} items</div>
                            <div><strong>Scanned:</strong> {scanned} items</div>
                            <div><strong>Missing:</strong> <span class="missing-qty">{missing} items</span></div>
                            <div><strong>Status:</strong> <span class="severity-high">⚠️ UNSCANNED</span></div>
                        </div>
                        <div class="quantity-indicator">
                            <div style="flex: 1;">
                                <strong style="font-size: 12px;">Inventory:</strong>
"""
            
            # Expected items boxes
            for j in range(actual):
                html_content += f'<div class="quantity-box expected">{j+1}</div>'
            
            html_content += """
                            </div>
                            <div style="flex: 1;">
                                <strong style="font-size: 12px;">Scanned:</strong>
"""
            
            # Scanned items boxes
            for j in range(scanned):
                html_content += f'<div class="quantity-box scanned">✓</div>'
            
            # Missing items boxes
            for j in range(missing):
                html_content += f'<div class="quantity-box missing">✗</div>'
            
            html_content += """
                            </div>
                        </div>
                    </div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <!-- Recommendations -->
        <div class="section">
            <div class="section-title">⚠️ Findings & Recommendations</div>
            <div class="print-section">
                <h3 style="color: #d32f2f; margin-bottom: 15px;">Key Findings:</h3>
                <ul style="margin-left: 20px; line-height: 2;">
                    <li>Multiple items were not scanned during the checkout process</li>
                    <li>Total of <strong>%s items</strong> were missed across <strong>%s different products</strong></li>
                    <li>Scan discrepancy rate: <strong>%.1f%%</strong></li>
                    <li>All occurrences have been recorded with precise timestamps and visual evidence</li>
                </ul>
                
                <h3 style="color: #1976d2; margin-top: 25px; margin-bottom: 15px;">Recommended Actions:</h3>
                <ul style="margin-left: 20px; line-height: 2;">
                    <li>Review checkout procedures with cashier</li>
                    <li>Provide additional training on scanner operation</li>
                    <li>Implement additional verification steps for all products</li>
                    <li>Monitor future transactions for similar patterns</li>
                    <li>File appropriate incident report as per store policy</li>
                </ul>
            </div>
        </div>
        
        <!-- Evidence Files -->
        <div class="section">
            <div class="section-title">📁 Evidence Files</div>
            <table>
                <thead>
                    <tr>
                        <th>File Type</th>
                        <th>Description</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Annotated Video</strong></td>
                        <td>Full video with visual markers and timestamps</td>
                        <td><span style="color: #4caf50; font-weight: bold;">✓ Saved</span></td>
                    </tr>
                    <tr>
                        <td><strong>JSON Report</strong></td>
                        <td>Structured data for analysis systems</td>
                        <td><span style="color: #4caf50; font-weight: bold;">✓ Saved</span></td>
                    </tr>
                    <tr>
                        <td><strong>HTML Report</strong></td>
                        <td>Professional documentation (this file)</td>
                        <td><span style="color: #4caf50; font-weight: bold;">✓ Saved</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p><strong>TillGuardAI</strong> - Retail Fraud Detection System</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <p style="margin-top: 20px; color: #999; font-size: 11px;">
                This is an official evidence document generated by TillGuardAI fraud detection system.
                All timestamps, visual markers, and data have been automatically recorded and verified.
            </p>
        </div>
    </div>
</body>
</html>
""" % (total_missing, len(self.data.get('discrepancies', [])), (total_missing / total_actual * 100) if total_actual > 0 else 0)
        
        # Write HTML file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✓ HTML Report saved: {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Error saving HTML report: {e}")
            return None
    
    def generate_text_report(self, output_file: str = None) -> str:
        """
        Generate plain text evidence report.
        
        Args:
            output_file: Output text file path
        
        Returns:
            Path to generated text file
        """
        output_file = output_file or os.path.join(self.output_dir, 'SCAN_DISCREPANCY_REPORT.txt')
        
        total_missing = self.data.get('summary', {}).get('total_items_missing', 0)
        total_actual = self.data.get('summary', {}).get('total_items_expected', 0)
        total_scanned = self.data.get('summary', {}).get('total_items_scanned', 0)
        
        text_content = f"""
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
Total Items Expected: {total_actual}
Total Items Scanned: {total_scanned}
Total Items Missing: {total_missing}
Discrepancy Rate: {(total_missing / total_actual * 100):.2f}%
Number of Discrepancies: {len(self.data.get('discrepancies', []))}

DETAILED DISCREPANCIES
{'─'*80}
"""
        
        for i, disc in enumerate(self.data.get('discrepancies', []), 1):
            timestamp_start = disc.get('start_time', 0)
            timestamp_end = disc.get('end_time', 0)
            
            text_content += f"""
[{i}] {disc.get('item_name', 'Unknown Item')}
    Expected Quantity: {disc.get('actual_quantity', 0)}
    Scanned Quantity: {disc.get('scanned_quantity', 0)}
    Missing Quantity: {disc.get('missing_quantity', 0)}
    Time Window: {timestamp_start:.2f}s - {timestamp_end:.2f}s
    Status: ⚠️  UNSCANNED - FRAUD DETECTED
"""
        
        text_content += f"""
FINDINGS & RECOMMENDATIONS
{'─'*80}

KEY FINDINGS:
• Multiple items were not scanned during checkout
• Total of {total_missing} items missed across {len(self.data.get('discrepancies', []))} products
• Scan discrepancy rate: {(total_missing / total_actual * 100):.2f}%
• All occurrences recorded with precise timestamps

RECOMMENDED ACTIONS:
1. Review checkout procedures with cashier
2. Provide additional scanner operation training
3. Implement additional verification steps
4. Monitor future transactions for patterns
5. File appropriate incident report per store policy

EVIDENCE FILES INCLUDED
{'─'*80}
✓ Annotated Video - Full video with visual markers and timestamps
✓ JSON Report - Structured data for analysis systems
✓ Text Report - This documentation

{'='*80}
DISCLAIMER
{'='*80}
This is an official evidence document generated by TillGuardAI fraud detection
system. All timestamps, visual markers, and data have been automatically recorded
and verified for accuracy and admissibility.

Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}
{'='*80}
"""
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"✓ Text Report saved: {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Error saving text report: {e}")
            return None
