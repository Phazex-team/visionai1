"""
Evidence Recorder for Fraud Detection
Records frames, video clips, and fraud reports for evidence
"""
import os
import json
import cv2
import numpy as np
from datetime import datetime
from collections import deque
from typing import List, Dict, Optional, Tuple
from config_models import EvidenceConfig, FaceMaskingConfig
from face_masking import FaceMasker, FaceMaskingConfig as FMConfig


class EvidenceRecorder:
    """
    Record and save evidence for fraud detection.
    Handles frame buffers, video clips, and fraud reports.
    """
    
    def __init__(
        self, 
        config: EvidenceConfig, 
        video_metadata: Dict, 
        enable_face_masking: Optional[bool] = None,
        face_masking_config: FaceMaskingConfig = None
    ):
        """
        Args:
            config: EvidenceConfig with output paths, buffer times, save modes
            video_metadata: Dict with fps, frame_width, frame_height, video_path
            enable_face_masking: Whether to blur faces in evidence (for privacy). If None, uses config.enable_face_masking.
            face_masking_config: Optional FaceMaskingConfig for advanced face masking settings
        """
        self.config = config
        self.metadata = video_metadata
        self.evidence_dir = None
        self.fraud_records = []
        self.after_frame_count = int(video_metadata['fps'] * config.buffer_seconds_after)
        self.pending_after_events = []
        
        # Initialize face masker for privacy protection
        self.face_masker = None
        self.enable_face_masking = config.enable_face_masking if enable_face_masking is None else enable_face_masking
        self._frame_num = 0  # Track frame numbers for face tracking
        
        if self.enable_face_masking:
            try:
                # Create FaceMasker with config (async MediaPipe by default)
                if face_masking_config is not None:
                    fm_config = FMConfig(
                        enabled=face_masking_config.enabled,
                        async_enabled=face_masking_config.async_enabled,
                        detector_type=face_masking_config.detector_type,
                        mask_type=face_masking_config.mask_type,
                        blur_strength=face_masking_config.blur_strength,
                        min_detection_confidence=face_masking_config.min_detection_confidence,
                        persistence_frames=face_masking_config.persistence_frames,
                        detection_interval_frames=face_masking_config.detection_interval_frames,
                        enable_profile_detection=face_masking_config.enable_profile_detection,
                        model_selection=face_masking_config.model_selection
                    )
                else:
                    # Default async MediaPipe config for evidence
                    fm_config = FMConfig(
                        async_enabled=True,
                        detector_type="mediapipe",
                        mask_type="blur",
                        blur_strength=51,
                        persistence_frames=15,
                        detection_interval_frames=3
                    )
                self.face_masker = FaceMasker(config=fm_config)
                print("[Evidence] Face masking ENABLED (async MediaPipe) for privacy protection")
            except Exception as e:
                print(f"[Evidence] WARNING: Could not initialize face masker: {e}")
                self.face_masker = None
        else:
            print("[Evidence] Face masking DISABLED by configuration")
        
        # Pending events waiting for after-frames to accumulate
        self.pending_after_events: List[Dict] = []
        
        # Frame buffer for recording before/after fraud
        max_buffer_frames = int(video_metadata['fps'] * config.buffer_seconds_before)
        self.frame_buffer = deque(maxlen=max_buffer_frames)
        self.frame_num_buffer = deque(maxlen=max_buffer_frames)
        self.detections_buffer = deque(maxlen=max_buffer_frames)  # Store detections with frames
        
        self._setup_evidence_dir()
    
    def _setup_evidence_dir(self):
        """Create evidence output directory structure"""
        if not self.config.enabled:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.basename(self.metadata.get('video_path', 'video'))
        video_name = os.path.splitext(base_name)[0]
        
        self.evidence_dir = os.path.join(
            self.config.output_dir,
            f"{video_name}_fraud_evidence_{timestamp}"
        )
        
        os.makedirs(self.evidence_dir, exist_ok=True)
        os.makedirs(os.path.join(self.evidence_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(self.evidence_dir, 'clips'), exist_ok=True)
        
        print(f"\n📁 Evidence directory created: {self.evidence_dir}")
    
    def _draw_all_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Draw all detection bounding boxes on frame.
        
        Args:
            frame: Input frame to annotate
            detections: Detection results with boxes, labels, scores
            
        Returns:
            Annotated frame with all bounding boxes
        """
        if not detections:
            return frame
        
        boxes = detections.get('boxes', [])
        labels = detections.get('labels', [])
        scores = detections.get('scores', [])
        
        # Color scheme for different detection types
        colors = {
            'person': (255, 0, 0),      # Blue
            'bottle': (0, 255, 0),      # Green
            'box': (0, 255, 255),       # Yellow
            'bag': (255, 0, 255),       # Magenta
            'item': (0, 165, 255),      # Orange
            'product': (255, 255, 0),   # Cyan
            'hand': (128, 0, 255),      # Purple
            'cart': (255, 128, 0),      # Light blue
            'trolley': (255, 128, 0),   # Light blue
            'basket': (0, 128, 255),    # Orange-red
            'default': (0, 255, 0)      # Green
        }
        
        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = [int(v) for v in box]
                label = labels[i] if i < len(labels) else 'item'
                score = scores[i] if i < len(scores) else 0.0
                
                # Get color based on label
                color = colors.get(label.lower(), colors['default'])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label_text = f"{label}: {score:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w + 5, y1), color, -1)
                cv2.putText(frame, label_text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception:
                pass
        
        return frame

    def _apply_face_masking(self, frame: np.ndarray, frame_num: int = None) -> np.ndarray:
        """
        Apply face masking to frame for privacy protection.
        
        Args:
            frame: Input frame (will NOT be modified)
            frame_num: Optional frame number for temporal tracking
            
        Returns:
            Frame with faces masked/blurred
        """
        if not self.enable_face_masking or self.face_masker is None:
            return frame
        
        try:
            # Use provided frame_num or auto-increment
            if frame_num is None:
                frame_num = self._frame_num
                self._frame_num += 1
            return self.face_masker.mask_faces(frame, frame_num=frame_num)
        except Exception as e:
            # If face masking fails, return original frame
            return frame

    def record_frame(self, frame: np.ndarray, frame_num: int, detections: Dict = None):
        """
        Buffer frame for potential evidence recording.
        
        Args:
            frame: Input frame
            frame_num: Frame number in video
            detections: Detection results for this frame (boxes, labels, scores)
        """
        if not self.config.enabled:
            return
        
        self.frame_buffer.append(frame.copy())
        self.frame_num_buffer.append(frame_num)
        self.detections_buffer.append(detections or {})
        
        # Attach frame to any pending events collecting after-frames
        if self.pending_after_events:
            remaining_events = []
            for event in self.pending_after_events:
                if event['remaining_after'] > 0:
                    event['frames'].append((frame.copy(), frame_num, detections or {}))
                    event['remaining_after'] -= 1
                
                if event['remaining_after'] <= 0:
                    self._finalize_pending_event(event)
                else:
                    remaining_events.append(event)
            self.pending_after_events = remaining_events
    
    def save_fraud_evidence(
        self,
        fraud_event: Dict,
        future_frames: Optional[List[Tuple[np.ndarray, int]]] = None
    ) -> Optional[Dict]:
        """
        Save evidence for a fraud event (before + during + after frames).
        
        Args:
            fraud_event: Dict with fraud details (type, label, frame_num, description, etc.)
            future_frames: List of (frame, frame_num) tuples captured after fraud
            
        Returns:
            Fraud record with evidence file paths, or None if not saved
        """
        if not self.config.enabled:
            return None
        
        # Skip evidence if ROI validation explicitly failed
        if fraud_event.get('roi_confirmed') is False:
            print(f"[Evidence] Skipping evidence for fraud event outside ROI: {fraud_event.get('type')}")
            return None
        
        if not self.evidence_dir:
            return None
        
        # Build initial evidence frames (before/during)
        before_frames = list(zip(self.frame_buffer, self.frame_num_buffer, self.detections_buffer))
        all_frames = before_frames.copy()
        
        # If future frames explicitly provided, save immediately
        if future_frames:
            all_frames.extend(list(future_frames))
            return self._save_evidence_from_frames(fraud_event, all_frames)
        
        # Otherwise, collect after-frames asynchronously
        if self.after_frame_count > 0:
            pending = {
                'fraud_event': fraud_event,
                'frames': all_frames,
                'remaining_after': self.after_frame_count
            }
            self.pending_after_events.append(pending)
            return None
        
        # No after-frames requested; save now
        return self._save_evidence_from_frames(fraud_event, all_frames)
    
    def _finalize_pending_event(self, pending_event: Dict):
        """Finalize a pending evidence bundle once after-frames are collected"""
        fraud_event = pending_event['fraud_event']
        frames_with_nums = pending_event['frames']
        self._save_evidence_from_frames(fraud_event, frames_with_nums)
    
    def _save_evidence_from_frames(self, fraud_event: Dict, frames_with_nums: List) -> Optional[Dict]:
        """Core routine to save frames + clip + record"""
        fraud_id = len(self.fraud_records)
        
        if not frames_with_nums:
            print(f"[Evidence] No frames available for fraud {fraud_id}")
            return None
        
        all_evidence = frames_with_nums
        frame_paths = []
        
        # 1. Save individual evidence frames
        if self.config.save_frames:
            fraud_bbox = fraud_event.get('bbox')
            fraud_label = fraud_event.get('label', 'fraud')
            fraud_type = fraud_event.get('type', 'fraud')
            
            for i, item in enumerate(all_evidence):
                if len(item) == 3:
                    frame, fn, frame_detections = item
                else:
                    frame, fn = item
                    frame_detections = {}
                
                frame_path = os.path.join(
                    self.evidence_dir,
                    'frames',
                    f"fraud_{fraud_id:03d}_frame_{fn:06d}_{i:03d}.jpg"
                )
                
                display_frame = frame.copy()
                display_frame = self._apply_face_masking(display_frame, frame_num=fn)
                display_frame = self._draw_all_detections(display_frame, frame_detections)
                
                if fraud_bbox is not None:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in fraud_bbox]
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        label_text = f"FRAUD: {fraud_label}"
                        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(display_frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 0, 255), -1)
                        cv2.putText(display_frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"[Evidence] Error drawing bbox: {e}")
                
                if self.config.frame_text_overlay:
                    cv2.putText(display_frame, f"Frame: {fn}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    fraud_banner = f"[{fraud_type.upper()}] {fraud_event.get('description', '')}"
                    cv2.putText(display_frame, fraud_banner, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imwrite(frame_path, display_frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality])
                frame_paths.append(frame_path)
        
        # 2. Save video clip
        video_path = None
        if self.config.save_video_clips and len(all_evidence) > 0:
            video_path = self._save_video_clip(fraud_id, all_evidence, fraud_event)
        
        # 3. Create fraud record
        evidence_frame_nums = []
        for item in all_evidence:
            if len(item) >= 2:
                evidence_frame_nums.append(item[1])
        
        fraud_record = {
            'fraud_id': fraud_id,
            'type': fraud_event.get('type', 'unknown'),
            'label': fraud_event.get('label', 'unknown'),
            'frame_num': fraud_event.get('frame_num', -1),
            'timestamp': fraud_event.get('timestamp', datetime.now().isoformat()),
            'description': fraud_event.get('description', ''),
            'confidence': fraud_event.get('confidence', 0.0),
            'evidence_frames': evidence_frame_nums,
            'frame_count': len(all_evidence),
            'buffer_before_seconds': self.config.buffer_seconds_before,
            'buffer_after_seconds': self.config.buffer_seconds_after,
            'frame_files': frame_paths,
            'video_file': video_path,
            'saved_at': datetime.now().isoformat(),
            'roi_zone': fraud_event.get('roi_zone'),
            'roi_confirmed': fraud_event.get('roi_confirmed'),
            'owner_id': fraud_event.get('owner_id'),
            'owner_role': fraud_event.get('owner_role'),
            'ownership_history': fraud_event.get('ownership_history')
        }
        
        self.fraud_records.append(fraud_record)
        print(f"[Evidence] Saved fraud #{fraud_id}: {fraud_event.get('description', 'fraud')} "
              f"({len(all_evidence)} frames)")
        return fraud_record
    
    def _create_video_writer(self, video_path: str, fps: float, width: int, height: int):
        """
        Create a VideoWriter with fallback codecs to avoid FFmpeg encoder errors.
        Returns (writer, used_codec) or (None, None) on failure.
        """
        codec_candidates = []
        if self.config.video_codec:
            codec_candidates.append(self.config.video_codec)
        codec_candidates.extend(['avc1', 'H264', 'X264', 'mp4v'])
        
        tried = []
        for codec in codec_candidates:
            if codec in tried:
                continue
            tried.append(codec)
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            if writer.isOpened():
                if codec != self.config.video_codec:
                    print(f"[Evidence] Video writer fallback to codec '{codec}' (requested '{self.config.video_codec}')")
                return writer, codec
            writer.release()
        
        print(f"[Evidence] Failed to create video writer for {video_path} (tried {tried})")
        return None, None
    
    def _save_video_clip(
        self,
        fraud_id: int,
        frames_with_nums: List,
        fraud_event: Dict = None
    ) -> Optional[str]:
        """
        Create and save video clip of fraud evidence with annotations.
        
        Args:
            fraud_id: Fraud event ID
            frames_with_nums: List of (frame, frame_num) or (frame, frame_num, detections) tuples
            fraud_event: Fraud event details (for drawing bbox)
            
        Returns:
            Path to saved video file, or None if failed
        """
        try:
            if not frames_with_nums:
                return None
            
            # Get frame dimensions from first frame
            first_frame = frames_with_nums[0][0]
            height, width = first_frame.shape[:2]
            
            video_path = os.path.join(
                self.evidence_dir,
                'clips',
                f"fraud_{fraud_id:03d}.mp4"
            )
            
            out, used_codec = self._create_video_writer(
                video_path,
                self.metadata['fps'],
                width,
                height
            )
            if out is None:
                return None
            
            # Get fraud bbox and label if available
            fraud_bbox = fraud_event.get('bbox') if fraud_event else None
            fraud_label = fraud_event.get('label', 'fraud') if fraud_event else 'fraud'
            fraud_type = fraud_event.get('type', 'fraud') if fraud_event else 'fraud'
            
            # Write all frames with annotations
            for item in frames_with_nums:
                # Handle both old format (frame, fn) and new format (frame, fn, detections)
                if len(item) == 3:
                    frame, frame_num, frame_detections = item
                else:
                    frame, frame_num = item
                    frame_detections = {}
                
                # Ensure frame is the right size
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                # Create annotated frame
                display_frame = frame.copy()
                
                # Apply face masking FIRST for privacy (pass frame_num for tracking)
                display_frame = self._apply_face_masking(display_frame, frame_num=frame_num)
                
                # Draw ALL detected items first (green/yellow boxes)
                display_frame = self._draw_all_detections(display_frame, frame_detections)
                
                # Draw fraud item bounding box ON TOP (in RED)
                if fraud_bbox is not None:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in fraud_bbox]
                        # Draw red bounding box for fraud item
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        # Draw label
                        label_text = f"FRAUD: {fraud_label}"
                        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(display_frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 0, 255), -1)
                        cv2.putText(display_frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    except Exception as e:
                        pass
                
                # Add frame info overlay
                cv2.putText(display_frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"[{fraud_type.upper()}]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                out.write(display_frame)
            
            out.release()
            fraud_event['video_codec'] = used_codec or self.config.video_codec
            print(f"[Evidence] Saved video clip: {video_path} (codec={fraud_event.get('video_codec')})")
            return video_path
        
        except Exception as e:
            print(f"[Evidence] Error saving video clip: {e}")
            return None
    
    def finalize_report(self) -> Optional[str]:
        """
        Finalize and save fraud report as JSON.
        
        Returns:
            Path to saved report file, or None if not saved
        """
        if not self.config.enabled or not self.config.save_fraud_report:
            return None
        
        if not self.evidence_dir:
            return None
        
        # Finalize any pending events with whatever frames have been collected
        if self.pending_after_events:
            for pending in list(self.pending_after_events):
                self._finalize_pending_event(pending)
            self.pending_after_events = []
        
        report_path = os.path.join(self.evidence_dir, 'fraud_report.json')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'video': self.metadata.get('video_path', 'unknown'),
            'fps': self.metadata.get('fps', 25),
            'total_frauds': len(self.fraud_records),
            'fraud_records': self.fraud_records,
            'summary': {
                'by_type': self._count_by_type(),
                'by_label': self._count_by_label(),
                'total_evidence_frames': sum(r.get('frame_count', 0) for r in self.fraud_records)
            }
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"[Evidence] Saved fraud report: {report_path}")
            return report_path
        except Exception as e:
            print(f"[Evidence] Error saving report: {e}")
            return None
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count frauds by type"""
        counts = {}
        for record in self.fraud_records:
            fraud_type = record.get('type', 'unknown')
            counts[fraud_type] = counts.get(fraud_type, 0) + 1
        return counts
    
    def _count_by_label(self) -> Dict[str, int]:
        """Count frauds by detected item label"""
        counts = {}
        for record in self.fraud_records:
            label = record.get('label', 'unknown')
            counts[label] = counts.get(label, 0) + 1
        return counts
    
    def get_summary(self) -> Dict:
        """Get summary of recorded evidence"""
        return {
            'evidence_dir': self.evidence_dir,
            'total_frauds': len(self.fraud_records),
            'total_evidence_frames': sum(r.get('frame_count', 0) for r in self.fraud_records),
            'frauds_by_type': self._count_by_type(),
            'frauds_by_label': self._count_by_label()
        }
    
    def cleanup(self):
        """Cleanup resources including face masker"""
        if self.face_masker is not None:
            try:
                self.face_masker.stop()
            except Exception:
                pass
            self.face_masker = None
