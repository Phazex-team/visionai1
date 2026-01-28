"""
Integrated Scan Discrepancy Evidence Processing Pipeline
Processes video evidence, generates annotated videos, and creates professional reports
"""

import os
import sys
import json
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from scan_discrepancy_annotator import ScanDiscrepancyAnnotator
from evidence_report_generator import ProfessionalEvidenceReport


def process_scan_discrepancy_evidence(
    video_path: str,
    discrepancies_data: list,
    output_base_dir: str = "evidence"
) -> dict:
    """
    Complete pipeline to process scan discrepancy evidence.
    
    Args:
        video_path: Path to input video file
        discrepancies_data: List of discrepancy dictionaries with:
            - item_name: str
            - actual_quantity: int
            - scanned_quantity: int
            - frame_range: tuple (start_frame, end_frame) or timestamps
            - description: str (optional)
        output_base_dir: Base directory for output files
    
    Returns:
        Dictionary with paths to all generated evidence files
    """
    
    print("\n" + "="*80)
    print("SCAN DISCREPANCY EVIDENCE PROCESSING PIPELINE")
    print("="*80)
    
    # Validate inputs
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not discrepancies_data:
        raise ValueError("No discrepancies provided")
    
    print(f"\n📹 Input Video: {video_path}")
    print(f"📊 Discrepancies to Process: {len(discrepancies_data)}")
    
    # Initialize annotator
    print("\n[1/3] Initializing video annotator...")
    annotator = ScanDiscrepancyAnnotator(
        video_path=video_path,
        output_dir=output_base_dir,
        fps=30,
        width=1920,
        height=1080
    )
    
    # Add discrepancies
    print("\n[2/3] Processing discrepancies...")
    for i, disc in enumerate(discrepancies_data, 1):
        item_name = disc.get('item_name', 'Unknown')
        actual_qty = disc.get('actual_quantity', 0)
        scanned_qty = disc.get('scanned_quantity', 0)
        frame_range = disc.get('frame_range')
        description = disc.get('description', '')
        
        # Handle frame ranges - convert time to frames if needed
        if isinstance(frame_range, tuple) and len(frame_range) == 2:
            start, end = frame_range
            # If values look like seconds (small numbers), convert to frames
            if start < 1000:  # Likely seconds
                start = int(start * annotator.video_fps)
                end = int(end * annotator.video_fps)
            frame_range = (start, end)
        
        print(f"\n  [{i}] {item_name}")
        print(f"      Expected: {actual_qty} | Scanned: {scanned_qty} | Missing: {actual_qty - scanned_qty}")
        
        annotator.add_discrepancy(
            item_name=item_name,
            actual_quantity=actual_qty,
            scanned_quantity=scanned_qty,
            frame_range=frame_range,
            description=description
        )
    
    # Create annotated video
    print("\n[3/3] Generating evidence outputs...")
    annotated_video = annotator.create_annotated_video()
    json_report = annotator.create_evidence_report()
    
    # Get JSON data for HTML report
    with open(json_report, 'r') as f:
        report_data = json.load(f)
    
    # Generate professional reports
    report_generator = ProfessionalEvidenceReport(report_data, annotator.output_dir)
    html_report = report_generator.generate_html_report()
    text_report = report_generator.generate_text_report()
    
    # Print summary
    print("\n")
    annotator.print_summary()
    
    # Prepare results
    results = {
        'status': 'SUCCESS',
        'output_directory': annotator.output_dir,
        'files_generated': {
            'annotated_video': annotated_video,
            'json_report': json_report,
            'html_report': html_report,
            'text_report': text_report
        },
        'statistics': {
            'total_discrepancies': len(discrepancies_data),
            'total_items_expected': report_data.get('summary', {}).get('total_items_expected', 0),
            'total_items_scanned': report_data.get('summary', {}).get('total_items_scanned', 0),
            'total_items_missing': report_data.get('summary', {}).get('total_items_missing', 0),
        }
    }
    
    print("\n" + "="*80)
    print("✅ EVIDENCE PROCESSING COMPLETE")
    print("="*80)
    print(f"\n📁 All files saved to: {annotator.output_dir}")
    print(f"\n📋 Generated Files:")
    print(f"   • Video:    {os.path.basename(annotated_video)}")
    print(f"   • JSON:     {os.path.basename(json_report)}")
    print(f"   • HTML:     {os.path.basename(html_report)}")
    print(f"   • Text:     {os.path.basename(text_report)}")
    print("\n" + "="*80 + "\n")
    
    return results


def main():
    """
    Main execution function.
    Processes the output.mp4 video with scan discrepancies.
    """
    
    # Define the video file and discrepancies
    video_file = "videos/output.mp4"
    
    # Check if video exists
    if not os.path.exists(video_file):
        print(f"❌ Video file not found: {video_file}")
        print("\nPlease ensure output.mp4 is located in the videos/ directory")
        return
    
    # Define scan discrepancies from user input
    discrepancies = [
        {
            'item_name': 'Malt Drink',
            'actual_quantity': 2,
            'scanned_quantity': 1,
            'frame_range': (0, 0.3),  # First 30% of video
            'description': 'Cashier missed one Malt Drink during scanning - Quantity discrepancy'
        },
        {
            'item_name': 'Kinder Joy',
            'actual_quantity': 2,
            'scanned_quantity': 1,
            'frame_range': (0.3, 0.6),  # Middle 30% of video
            'description': 'Cashier missed one Kinder Joy during scanning - Quantity discrepancy'
        }
    ]
    
    # Process evidence
    try:
        results = process_scan_discrepancy_evidence(
            video_path=video_file,
            discrepancies_data=discrepancies,
            output_base_dir="evidence"
        )
        
        # Save results summary
        summary_file = os.path.join(results['output_directory'], 'processing_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Processing summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"\n❌ Error during evidence processing: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
