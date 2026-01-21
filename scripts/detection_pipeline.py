"""
Detection Pipeline - Main Orchestrator
Coordinates model loading, optimization, POS processing, fraud detection, and evidence recording
"""
import cv2
import numpy as np
import yaml
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from config_models import ApplicationConfig, DetectionConfig, OptimizationConfig
from model_factory import ModelFactory, ModelRegistry
from optimization_manager import OptimizationManager
from pos_processor import POSProcessor
from evidence_recorder import EvidenceRecorder
from fraud_detector_v2 import FraudDetectorV2


class DetectionPipeline:
    """
    Main orchestrator for the complete detection pipeline.
    Coordinates:
    - Model loading via ModelFactory
    - Performance optimization (ROI, resize, GPU memory)
    - POS data loading and matching
    - Fraud detection via FraudDetectorV2
    - Evidence recording
    """
    
    def __init__(self, config: ApplicationConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: ApplicationConfig instance
        """
        self.config = config
        config.validate()
        
        self.model_registry = ModelRegistry()
        
        print(f"\n{'='*70}")
        print(f"DETECTION PIPELINE INITIALIZATION")
        print(f"{'='*70}")
        
        # 1. Create model with factory
        self.model = ModelFactory.create_model(config.detection, config.optimization)
        self.model_registry.register_model('primary', self.model, config.detection, config.optimization)
        
        # 1b. Set detection classes on the model
        detection_classes = []
        detection_classes.extend(config.classes.retail_classes)
        detection_classes.extend(config.classes.person_classes)
        detection_classes.extend(config.classes.scanner_classes)
        
        if detection_classes:
            print(f"\n📋 Setting detection classes: {detection_classes}")
            self.model.set_classes(detection_classes)
        else:
            print(f"\n⚠️  Warning: No detection classes configured!")
        
        # 2. Initialize optimization manager
        self.optimization_mgr = OptimizationManager(config.optimization)
        
        # 3. Initialize POS processor
        self.pos_processor = POSProcessor(
            config.pos,
            fps=config.fps,
            video_start_time=self._get_video_start_time()
        )
        
        if self.config.pos.enabled:
            print(f"\n📋 POS Data Summary:")
            print(json.dumps(self.pos_processor.get_pos_summary(), indent=2, default=str))
        
        # 4. Initialize evidence recorder
        self.evidence_recorder = EvidenceRecorder(config.evidence, {
            'fps': config.fps,
            'frame_width': config.frame_width,
            'frame_height': config.frame_height,
            'video_path': config.video_path
        }, enable_face_masking=config.evidence.enable_face_masking)
        
        # 5. Initialize fraud detector
        self.fraud_detector = FraudDetectorV2(
            fps=config.fps,
            frame_width=config.frame_width,
            frame_height=config.frame_height
        )
        self._load_zones()
        
        print(f"{'='*70}\n")
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'total_detections': 0,
            'fraud_events': 0,
            'average_inference_time': 0.0,
            'pos_matches': 0,
            'pos_mismatches': 0
        }
    
    def _get_video_start_time(self) -> datetime:
        """Extract video start time from metadata or use current time"""
        if self.config.metadata.get('video_start_time'):
            return datetime.fromisoformat(self.config.metadata['video_start_time'])
        return datetime.now()
    
    def _load_zones(self):
        """Load zone definitions into fraud detector"""
        zones = self.config.zones.to_dict()
        for zone_name, points in zones.items():
            if points:  # Only set if points exist
                self.fraud_detector.zones[zone_name] = np.array(points)
    
    def _draw_detections(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input frame to annotate
            result: Detection result from process_frame
            
        Returns:
            Annotated frame with bounding boxes
        """
        if not result or 'detections' not in result:
            return frame
        
        detections = result['detections']
        boxes = detections.get('boxes', [])
        labels = detections.get('labels', [])
        scores = detections.get('scores', [])
        tracker_ids = detections.get('tracker_ids', [])
        
        # Color scheme for different detection types
        colors = {
            'person': (255, 0, 0),      # Blue
            'bottle': (0, 255, 0),      # Green
            'box': (0, 255, 255),       # Yellow
            'bag': (255, 0, 255),       # Magenta
            'item': (0, 165, 255),      # Orange
            'product': (255, 255, 0),   # Cyan
            'default': (0, 255, 0)      # Green
        }
        
        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = [int(v) for v in box]
                label = labels[i] if i < len(labels) else 'item'
                score = scores[i] if i < len(scores) else 0.0
                track_id = tracker_ids[i] if i < len(tracker_ids) else None
                
                # Get color based on label
                color = colors.get(label.lower(), colors['default'])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence and ID
                if track_id is not None:
                    label_text = f"{label} #{track_id}: {score:.2f}"
                else:
                    label_text = f"{label}: {score:.2f}"
                    
                (label_w, label_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w + 5, y1), color, -1)
                cv2.putText(frame, label_text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                pass
        
        # Draw zones overlay
        zone_colors = {
            'counter': (0, 255, 0),      # Green
            'scanner': (255, 0, 0),      # Blue
            'exit': (0, 0, 255),         # Red
            'packing_area': (0, 255, 255), # Yellow
            'customer_area': (255, 255, 0), # Cyan
        }
        
        for zone_name, zone_pts in self.fraud_detector.zones.items():
            if zone_pts is not None and len(zone_pts) > 0:
                pts = zone_pts.astype(np.int32).reshape((-1, 1, 2))
                color = zone_colors.get(zone_name, (128, 128, 128))
                cv2.polylines(frame, [pts], True, color, 2)
                # Draw zone label
                cx = int(np.mean(zone_pts[:, 0]))
                cy = int(np.mean(zone_pts[:, 1]))
                cv2.putText(frame, zone_name.upper(), (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw frame number
        cv2.putText(frame, f"Frame: {result['frame_num']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw fraud alerts if any
        fraud_result = result.get('fraud', {})
        if fraud_result.get('fraud_events'):
            cv2.putText(frame, "⚠️ FRAUD DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame

    def process_frame(self, frame: np.ndarray, frame_num: int) -> Dict:
        """
        Process a single frame through the entire pipeline.
        
        Args:
            frame: Input frame (BGR, from OpenCV)
            frame_num: Frame number in video
            
        Returns:
            Dict with detection results, POS matching, fraud detection
        """
        self.stats['total_frames'] += 1
        
        # 1. Check if frame should be skipped
        if self.optimization_mgr.should_skip_frame(frame_num):
            self.stats['skipped_frames'] += 1
            return None
        
        self.stats['processed_frames'] += 1
        
        # 2. Prepare frame (ROI crop, resize)
        prep_frame, prep_metadata = self.optimization_mgr.prepare_frame(frame)
        
        # 3. Run model inference
        detections = self.model.predict(prep_frame)
        
        # 4. Map boxes back to original frame coordinates
        if 'boxes' in detections and len(detections['boxes']) > 0:
            detections['boxes'] = self.optimization_mgr.map_boxes_back(
                detections['boxes'],
                prep_metadata
            )
        
        self.stats['total_detections'] += len(detections.get('labels', []))
        
        # Record frame for evidence WITH detections (for drawing bboxes in evidence clips)
        self.evidence_recorder.record_frame(frame, frame_num, detections)
        
        # 5. Match detections with POS data
        pos_match = None
        if self.config.pos.enabled:
            pos_match = self.pos_processor.match_detections_to_pos(
                frame_num,
                detections.get('labels', [])
            )
            self.stats['pos_matches'] += pos_match['matched_count']
            self.stats['pos_mismatches'] += pos_match['unmatched_pos_items']
        
        # 6. Run fraud detection
        fraud_result = self.fraud_detector.process_frame(
            frame,
            frame_num,
            detections,
            pos_match
        )
        
        if fraud_result.get('fraud_events'):
            self.stats['fraud_events'] += len(fraud_result['fraud_events'])
            
            # Save evidence for fraud events
            for event in fraud_result['fraud_events']:
                self.evidence_recorder.save_fraud_evidence(event)
        
        # 7. Manage GPU memory
        self.optimization_mgr.manage_gpu_memory()
        
        return {
            'frame_num': frame_num,
            'detections': detections,
            'pos_match': pos_match,
            'fraud': fraud_result
        }
    
    def run_video(self, max_frames: Optional[int] = None, save_output: bool = True):
        """Process entire video from config
        
        Args:
            max_frames: Maximum frames to process (None for all)
            save_output: If True, save annotated output video with bounding boxes
        """
        cap = cv2.VideoCapture(self.config.video_path)
        frame_idx = 0
        
        # Setup output video writer if saving
        output_writer = None
        if save_output:
            output_path = self.config.video_path.replace('.mp4', '_output.mp4')
            if not output_path.endswith('_output.mp4'):
                output_path = 'output_video.mp4'
            
            fps = cap.get(cv2.CAP_PROP_FPS) or self.config.fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"📹 Saving annotated output to: {output_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break
            
            # Process frame
            result = self.process_frame(frame, frame_idx)
            
            # Draw bounding boxes on output video
            if save_output and output_writer and result:
                annotated_frame = self._draw_detections(frame.copy(), result)
                output_writer.write(annotated_frame)
            
            frame_idx += 1
            
        cap.release()
        if output_writer:
            output_writer.release()
            print(f"✅ Output video saved!")
        
        # Finalize report and return stats
        self.evidence_recorder.finalize_report()
        
        # Gather summary statistics
        summary = self.fraud_detector.get_summary()
        
        # Add POS stats if enabled
        if self.config.pos.enabled:
            pos_summary = self.pos_processor.get_pos_summary()
            summary['pos_total_items'] = pos_summary['total_items']
            summary['pos_unique_items'] = pos_summary['unique_items']
            summary['pos_matched_events'] = self.stats['pos_matches']
            summary['pos_unmatched_events'] = self.stats['pos_mismatches']
            
        # Add processing stats
        summary['processed_frames'] = frame_idx
        summary['total_detections'] = self.stats['total_detections']
        
        return summary
    
    def _finalize(self):
        """Finalize processing and save reports"""
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        
        # Save evidence report
        evidence_report = self.evidence_recorder.finalize_report()
        
        # Print statistics
        print(f"\n📊 STATISTICS:")
        print(f"  Total frames: {self.stats['total_frames']}")
        print(f"  Processed frames: {self.stats['processed_frames']}")
        print(f"  Skipped frames: {self.stats['skipped_frames']}")
        print(f"  Total detections: {self.stats['total_detections']}")
        print(f"  Fraud events: {self.stats['fraud_events']}")
        if self.config.pos.enabled:
            print(f"  POS matches: {self.stats['pos_matches']}")
            print(f"  POS mismatches: {self.stats['pos_mismatches']}")
        
        # Model metrics
        metrics = self.model.get_metrics_summary()
        print(f"\n🔍 MODEL METRICS:")
        print(f"  Model: {metrics['model_name']}")
        print(f"  Avg inference time: {metrics['avg_inference_time_ms']:.1f}ms")
        print(f"  Avg detections/frame: {metrics['avg_detections_per_frame']:.1f}")
        
        # Evidence summary
        if self.config.evidence.enabled:
            evidence_summary = self.evidence_recorder.get_summary()
            print(f"\n📁 EVIDENCE RECORDED:")
            print(f"  Directory: {evidence_summary['evidence_dir']}")
            print(f"  Total fraud events: {evidence_summary['total_frauds']}")
            print(f"  Total evidence frames: {evidence_summary['total_evidence_frames']}")
        
        print(f"{'='*70}\n")
    
    def get_statistics(self) -> Dict:
        """Get current pipeline statistics"""
        return self.stats.copy()


def load_config_from_file(config_path: str) -> ApplicationConfig:
    """
    Load ApplicationConfig from YAML or JSON file.
    
    Args:
        config_path: Path to config file (.yaml or .json)
        
    Returns:
        ApplicationConfig instance
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return ApplicationConfig.from_dict(config_dict)


def create_example_config(output_path: str = 'config_example.yaml'):
    """
    Create an example configuration file.
    
    Args:
        output_path: Where to save the example config
    """
    example_config = {
        'video_path': 'videos/NVR_ch10_main_20260109095150_20260109095555.mp4',
        'fps': 25,
        'frame_width': 1280,
        'frame_height': 720,
        'detection': {
            'model_name': 'yoloworld',
            'model_type': 'cnn',
            'confidence_threshold': 0.15,
            'iou_threshold': 0.5,
            'device': 'cuda'
        },
        'optimization': {
            'model_name': 'yoloworld',
            'enable_roi_crop': False,
            'roi_bounds': None,
            'input_width': 1280,
            'input_height': 720,
            'max_dim': 1280,
            'skip_every_n_frames': 1,
            'target_fps': 30,
            'clear_gpu_after_n_frames': 30
        },
        'pos': {
            'enabled': True,
            'xml_path': 'pos_data.xml',
            'data_format': 'xml',
            'timezone_offset': 0,
            'match_strategy': 'fuzzy',
            'min_match_confidence': 0.7
        },
        'evidence': {
            'enabled': True,
            'output_dir': 'evidence',
            'record_mode': 'fraud_with_buffer',
            'buffer_seconds_before': 3.0,
            'buffer_seconds_after': 5.0,
            'save_frames': True,
            'save_video_clips': True,
            'save_fraud_report': True,
            'image_quality': 95
        },
        'zones': {
            'counter': [[100, 150], [1200, 150], [1200, 700], [100, 700]],
            'scanner': [[400, 200], [800, 200], [800, 600], [400, 600]],
            'trolley': [[50, 600], [1230, 600], [1230, 720], [50, 720]],
            'exit': [[1150, 0], [1280, 0], [1280, 720], [1150, 720]]
        },
        'classes': {
            'retail_classes': ['bottle', 'box', 'bag', 'item', 'product'],
            'person_classes': ['person'],
            'scanner_classes': ['scanner']
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Example config saved to: {output_path}")
