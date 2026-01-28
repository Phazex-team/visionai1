"""
Scan Discrepancy Evidence Annotator
Creates professional annotated video evidence showing scanning discrepancies
with timestamps and visual markers for missed items
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ScanDiscrepancyAnnotator:
    """
    Annotates video evidence with scan discrepancies.
    Adds visual markers, timestamps, and professional annotations.
    """
    
    # Color scheme for professional evidence video
    COLORS = {
        'missed_item': (0, 0, 255),      # Red for missed items
        'alert_bg': (0, 0, 255),          # Red background
        'text_white': (255, 255, 255),    # White text
        'scan_box': (0, 165, 255),        # Orange for scanned items
        'correct': (0, 255, 0),           # Green for correct items
        'warning': (0, 165, 255),         # Orange for warnings
        'text_dark': (0, 0, 0),           # Black text
    }
    
    def __init__(
        self,
        video_path: str,
        output_dir: str = None,
        fps: int = 30,
        width: int = 1920,
        height: int = 1080
    ):
        """
        Initialize the annotator.
        
        Args:
            video_path: Path to input video file
            output_dir: Output directory for annotated video and report
            fps: Frames per second for output video
            width: Output video width
            height: Output video height
        """
        self.video_path = video_path
        self.output_dir = output_dir or self._create_output_dir()
        self.fps = fps
        self.width = width
        self.height = height
        
        # Validate video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video properties
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n📹 Video Properties:")
        print(f"   - Resolution: {self.video_width}x{self.video_height}")
        print(f"   - FPS: {self.video_fps}")
        print(f"   - Total Frames: {self.total_frames}")
        print(f"   - Duration: {self.total_frames / self.video_fps:.2f}s")
        
        # Discrepancy data
        self.discrepancies: List[Dict] = []
        
    def _create_output_dir(self) -> str:
        """Create output directory for evidence"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = Path(self.video_path).stem
        output_dir = f"evidence/{video_name}_scan_discrepancy_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n📁 Output directory: {output_dir}")
        return output_dir
    
    def add_discrepancy(
        self,
        item_name: str,
        actual_quantity: int,
        scanned_quantity: int,
        frame_range: Tuple[int, int],
        description: str = ""
    ):
        """
        Add a scanning discrepancy to be annotated.
        
        Args:
            item_name: Name of the item (e.g., "Malt Drink")
            actual_quantity: Actual quantity in basket
            scanned_quantity: Quantity scanned by cashier
            frame_range: (start_frame, end_frame) for this discrepancy
            description: Optional description
        """
        missing_qty = actual_quantity - scanned_quantity
        
        discrepancy = {
            'item_name': item_name,
            'actual_quantity': actual_quantity,
            'scanned_quantity': scanned_quantity,
            'missing_quantity': missing_qty,
            'frame_range': frame_range,
            'start_time': frame_range[0] / self.video_fps,
            'end_time': frame_range[1] / self.video_fps,
            'description': description or f"{missing_qty} {item_name} missed scan",
            'severity': 'HIGH' if missing_qty > 0 else 'LOW'
        }
        
        self.discrepancies.append(discrepancy)
        print(f"\n✓ Added discrepancy: {item_name}")
        print(f"  Actual: {actual_quantity}, Scanned: {scanned_quantity}, Missing: {missing_qty}")
        print(f"  Frames: {frame_range[0]} - {frame_range[1]} ({discrepancy['start_time']:.2f}s - {discrepancy['end_time']:.2f}s)")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS.ms format"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    def _frame_to_timestamp(self, frame_num: int) -> str:
        """Convert frame number to timestamp string"""
        seconds = frame_num / self.video_fps
        return self._format_timestamp(seconds)
    
    def _draw_alert_banner(
        self,
        frame: np.ndarray,
        text: str,
        y_position: int = 50,
        severity: str = 'HIGH'
    ) -> np.ndarray:
        """
        Draw professional alert banner on frame.
        
        Args:
            frame: Input frame
            text: Alert text
            y_position: Y position for banner
            severity: 'HIGH', 'MEDIUM', or 'LOW'
        
        Returns:
            Annotated frame
        """
        # Color based on severity
        if severity == 'HIGH':
            bg_color = (0, 0, 255)  # Red
            text_color = (255, 255, 255)  # White
        elif severity == 'MEDIUM':
            bg_color = (0, 165, 255)  # Orange
            text_color = (255, 255, 255)  # White
        else:
            bg_color = (0, 255, 255)  # Yellow
            text_color = (0, 0, 0)  # Black
        
        # Create banner background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        margin_x = 20
        margin_y = 15
        cv2.rectangle(
            frame,
            (margin_x, y_position - text_h - margin_y),
            (margin_x + text_w + margin_x, y_position + baseline + margin_y),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (margin_x + 10, y_position + baseline),
            font,
            font_scale,
            text_color,
            thickness
        )
        
        return frame
    
    def _draw_item_marker(
        self,
        frame: np.ndarray,
        item_info: Dict,
        y_offset: int = 150
    ) -> np.ndarray:
        """
        Draw item discrepancy marker on frame.
        
        Args:
            frame: Input frame
            item_info: Item discrepancy info
            y_offset: Y offset from top
        
        Returns:
            Annotated frame
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        
        # Item name
        item_text = f"ITEM: {item_info['item_name']}"
        cv2.putText(frame, item_text, (40, y_offset), font, font_scale, (0, 165, 255), thickness)
        
        # Quantities
        qty_text = f"Expected: {item_info['actual_quantity']} | Scanned: {item_info['scanned_quantity']} | Missing: {item_info['missing_quantity']}"
        cv2.putText(frame, qty_text, (40, y_offset + 40), font, 0.8, (0, 0, 255), thickness)
        
        # Visual quantity indicator
        box_width = 30
        box_height = 20
        spacing = 5
        x_start = 40
        y_start = y_offset + 80
        
        # Expected items (green boxes)
        for i in range(item_info['actual_quantity']):
            x = x_start + i * (box_width + spacing)
            cv2.rectangle(frame, (x, y_start), (x + box_width, y_start + box_height), (0, 255, 0), 2)
            cv2.putText(frame, str(i+1), (x + 8, y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Scanned items (blue boxes with X)
        scanned_color = (255, 0, 0)  # Blue
        for i in range(item_info['scanned_quantity']):
            x = x_start + i * (box_width + spacing)
            cv2.rectangle(frame, (x, y_start + 40), (x + box_width, y_start + 40 + box_height), scanned_color, 2)
            cv2.putText(frame, "✓", (x + 5, y_start + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, scanned_color, 2)
        
        # Missing items (red boxes with X)
        missing_color = (0, 0, 255)  # Red
        for i in range(item_info['missing_quantity']):
            x = x_start + (i + item_info['scanned_quantity']) * (box_width + spacing)
            cv2.rectangle(frame, (x, y_start + 40), (x + box_width, y_start + 40 + box_height), missing_color, 2)
            cv2.putText(frame, "✗", (x + 7, y_start + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.8, missing_color, 2)
        
        # Legend
        legend_y = y_start + 90
        cv2.putText(frame, "Green: Expected  Blue: Scanned  Red: Missing", (40, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        
        return frame
    
    def _draw_timestamp_overlay(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """
        Draw timestamp and frame info overlay.
        
        Args:
            frame: Input frame
            frame_num: Frame number
        
        Returns:
            Annotated frame
        """
        timestamp = self._frame_to_timestamp(frame_num)
        time_text = f"Time: {timestamp}"
        
        # Draw in bottom right corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(time_text, font, 0.7, 2)
        
        x = self.width - text_w - 20
        y = self.height - 20
        
        # Background
        cv2.rectangle(frame, (x - 10, y - text_h - 10), (self.width - 10, self.height - 5), (0, 0, 0), -1)
        
        # Text
        cv2.putText(frame, time_text, (x, y), font, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def create_annotated_video(self, output_video: str = None) -> str:
        """
        Create annotated video with discrepancy markers.
        
        Args:
            output_video: Output video path (default: output_dir/annotated_evidence.mp4)
        
        Returns:
            Path to annotated video
        """
        if not self.discrepancies:
            print("⚠️  No discrepancies to annotate. Add discrepancies first.")
            return None
        
        output_video = output_video or os.path.join(self.output_dir, 'annotated_evidence.mp4')
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, self.video_fps, (self.width, self.height))
        
        if not out.isOpened():
            print(f"❌ Failed to create video writer for {output_video}")
            return None
        
        print(f"\n🎬 Creating annotated video: {output_video}")
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Resize frame to output dimensions if needed
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Check which discrepancies apply to this frame
                active_discrepancies = [
                    d for d in self.discrepancies
                    if d['frame_range'][0] <= frame_count <= d['frame_range'][1]
                ]
                
                # Draw overlays if discrepancies are active
                if active_discrepancies:
                    # Draw main alert banner
                    banner_text = "⚠️  SCAN DISCREPANCY DETECTED"
                    frame = self._draw_alert_banner(frame, banner_text, y_position=50, severity='HIGH')
                    
                    # Draw details for each active discrepancy
                    for idx, discrepancy in enumerate(active_discrepancies):
                        y_offset = 150 + idx * 180
                        frame = self._draw_item_marker(frame, discrepancy, y_offset)
                
                # Draw timestamp
                frame = self._draw_timestamp_overlay(frame, frame_count)
                
                # Write frame
                out.write(frame)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"  Processed {frame_count}/{self.total_frames} frames")
        
        finally:
            out.release()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        print(f"✓ Annotated video saved: {output_video}")
        return output_video
    
    def create_evidence_report(self, output_file: str = None) -> str:
        """
        Create professional evidence report.
        
        Args:
            output_file: Output JSON file path
        
        Returns:
            Path to report file
        """
        output_file = output_file or os.path.join(self.output_dir, 'scan_discrepancy_report.json')
        
        # Calculate summary statistics
        total_missing = sum(d['missing_quantity'] for d in self.discrepancies)
        total_actual = sum(d['actual_quantity'] for d in self.discrepancies)
        total_scanned = sum(d['scanned_quantity'] for d in self.discrepancies)
        
        report = {
            'report_type': 'SCAN_DISCREPANCY_EVIDENCE',
            'generated_at': datetime.now().isoformat(),
            'video_source': self.video_path,
            'video_properties': {
                'resolution': f"{self.video_width}x{self.video_height}",
                'fps': self.video_fps,
                'total_frames': self.total_frames,
                'duration_seconds': self.total_frames / self.video_fps
            },
            'summary': {
                'total_discrepancies': len(self.discrepancies),
                'total_items_expected': total_actual,
                'total_items_scanned': total_scanned,
                'total_items_missing': total_missing,
                'discrepancy_rate': f"{(total_missing / total_actual * 100):.2f}%" if total_actual > 0 else "0%",
                'severity': 'HIGH' if total_missing > 0 else 'LOW'
            },
            'discrepancies': self.discrepancies,
            'evidence_files': {
                'annotated_video': os.path.join(self.output_dir, 'annotated_evidence.mp4'),
                'this_report': output_file
            }
        }
        
        # Save report
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✓ Report saved: {output_file}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return None
        
        return output_file
    
    def print_summary(self):
        """Print evidence summary to console"""
        if not self.discrepancies:
            print("No discrepancies recorded.")
            return
        
        print("\n" + "="*80)
        print("SCAN DISCREPANCY EVIDENCE SUMMARY")
        print("="*80)
        
        total_missing = sum(d['missing_quantity'] for d in self.discrepancies)
        total_actual = sum(d['actual_quantity'] for d in self.discrepancies)
        total_scanned = sum(d['scanned_quantity'] for d in self.discrepancies)
        
        print(f"\n📊 STATISTICS:")
        print(f"  - Total Items Expected: {total_actual}")
        print(f"  - Total Items Scanned: {total_scanned}")
        print(f"  - Total Items Missing: {total_missing}")
        print(f"  - Discrepancy Rate: {(total_missing / total_actual * 100):.2f}%")
        print(f"  - Number of Items with Discrepancies: {len(self.discrepancies)}")
        
        print(f"\n📋 DETAILED DISCREPANCIES:")
        for i, d in enumerate(self.discrepancies, 1):
            print(f"\n  [{i}] {d['item_name']}")
            print(f"      Expected: {d['actual_quantity']}")
            print(f"      Scanned: {d['scanned_quantity']}")
            print(f"      Missing: {d['missing_quantity']}")
            print(f"      Time: {self._frame_to_timestamp(d['frame_range'][0])} - {self._frame_to_timestamp(d['frame_range'][1])}")
            print(f"      Severity: {d['severity']}")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main function to demonstrate usage"""
    import sys
    
    # Check if video file exists in workspace
    video_path = "videos/output.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please ensure output.mp4 is in the videos/ directory")
        return
    
    # Create annotator
    annotator = ScanDiscrepancyAnnotator(
        video_path=video_path,
        output_dir="evidence",
        fps=30,
        width=1920,
        height=1080
    )
    
    # Add discrepancies from user report
    # Malt Drink: 2 actual, 1 scanned, 1 missing
    # Frames 0-500 (approximate, adjust based on actual video)
    annotator.add_discrepancy(
        item_name="Malt Drink",
        actual_quantity=2,
        scanned_quantity=1,
        frame_range=(0, int(annotator.total_frames * 0.3)),
        description="Cashier missed one Malt Drink during scanning"
    )
    
    # Kinder Joy: 2 actual, 1 scanned, 1 missing
    # Frames 500-1000 (approximate, adjust based on actual video)
    annotator.add_discrepancy(
        item_name="Kinder Joy",
        actual_quantity=2,
        scanned_quantity=1,
        frame_range=(int(annotator.total_frames * 0.3), int(annotator.total_frames * 0.6)),
        description="Cashier missed one Kinder Joy during scanning"
    )
    
    # Create annotated video
    video_output = annotator.create_annotated_video()
    
    # Create report
    report_output = annotator.create_evidence_report()
    
    # Print summary
    annotator.print_summary()
    
    print(f"\n✅ Evidence package created in: {annotator.output_dir}")
    print(f"   - Video: {video_output}")
    print(f"   - Report: {report_output}")


if __name__ == "__main__":
    main()
